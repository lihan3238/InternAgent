import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from datasets import Dataset
from torch.utils.data import TensorDataset, DataLoader
import random
import os
import traceback
import sys
from tqdm import tqdm
import pandas as pd
from sklearn.utils import shuffle
import time
from torch.nn.utils import clip_grad_norm_
import json
import datetime

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from deepspeed import get_accelerator
import logging
import deepspeed
import subprocess
import wandb
from sklearn.metrics import r2_score
import numpy as np

import pathlib

import os
import re
import shutil


def save_model_and_losses(
    global_rank,
    losses=None, 
    base_model_save_path=None, 
    lora_adapter_save_path=None, 
    model=None, 
    use_lora=False, 
    save_root=None, 
    logger=None
):

    model_to_save = model.module if hasattr(model, 'module') else model

    model_to_save.llama.base_model.save_pretrained(base_model_save_path)
    print(f"Base model saved to: {base_model_save_path}")
    logger.info(f"Base model saved to: {base_model_save_path}")
    
    if use_lora and lora_adapter_save_path is not None:
        model_to_save.llama.save_pretrained(lora_adapter_save_path)
        print(f"LoRA adapter saved to: {lora_adapter_save_path}")
        logger.info(f"LoRA adapter saved to: {lora_adapter_save_path}")
    
    if isinstance(model_to_save.predictor, torch.nn.Module) and save_root is not None:
        pathlib.Path(save_root).mkdir(parents=True, exist_ok=True) 
        torch.save(model_to_save.predictor.state_dict(), os.path.join(save_root, 'predictor.pt'))


def parse_filename(filename):
    match = re.match(r'(\d+)-(\d+)-(\d+\.?\d*)', filename)
    if match:
        epoch = int(match.group(1))
        step = int(match.group(2))
        loss = float(match.group(3))
        return epoch, step, loss
    return None


def delete_files_wrt_loss(save_path, max_save_files=5, reverse=True):
    files = os.listdir(save_path)
    parsed_files = []

    for file in files:
        parsed = parse_filename(file)
        if parsed:
            parsed_files.append((file, *parsed))

    if not parsed_files:
        return

    parsed_files.sort(key=lambda x: x[3], reverse=reverse)
    print(parsed_files)

    max_save_files = min(max_save_files, len(parsed_files))
    best_files = parsed_files[:max_save_files]


    for file, _, _, _ in parsed_files[max_save_files:]:
        file_path = os.path.join(save_path, file)
        for subfile in os.listdir(file_path):
            os.remove(os.path.join(file_path, subfile))
        os.rmdir(file_path)

    print(f"Kept {len(best_files)} best files with the smallest loss.")

def save_ckpt_with_limited_files(model, save_path, epoch, step, loss, max_save_files=5, reverse=True):
    ckpt_id = f'{epoch}-{step}-{loss}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    if max_save_files is not None:
        if dist.get_rank()==0:
            delete_files_wrt_loss(save_path, max_save_files, reverse=reverse)
    model.save_checkpoint(save_path, tag=ckpt_id)


def read_reaction(path,data_name):
    dataset = Dataset.load_from_disk(os.path.join(path, data_name))
    # raw_data = pd.read_csv(os.path.join(path, data_name, data_name + ".csv"))
    known_yields = dataset['yield']
    known_conditions = dataset['condition']
    reactions = dataset['reaction']
    return known_conditions, known_yields, reactions



def cleanup():
    dist.destroy_process_group()

class YieldPredLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(YieldPredLayer, self).__init__()
        self.act = nn.SiLU()
        self.predictor = nn.Sequential(
                            nn.Linear(input_size, hidden_size),
                            # self.act,
                            # nn.Linear(hidden_size, hidden_size//4),
                            # # self.act,
                            nn.Linear(hidden_size, 1),
                        )
    def forward(self, x):
        pred = self.predictor(x)
        # print(f'pred:{pred.view(-1)}')
        # print(f'y: {y.view(-1)}')
        return pred

class LlamaWithLoss(nn.Module):
    def __init__(self, llama, predictor):
        super(LlamaWithLoss, self).__init__()
        self.llama = llama
        self.loss_func = torch.nn.MSELoss()

        self.predictor = predictor
        
    def forward(self, inputs, y, pooling_method='last_token', return_loss=True):
        outputs = self.llama(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state
        if pooling_method == 'mean':
            embeddings = last_hidden_state.mean(dim=1)  # Mean pooling to get sentence-level embeddings
        elif pooling_method=='last_token':
            embeddings = last_hidden_state[:,-1,:]
        if return_loss:
            pred = self.predictor(embeddings)
            loss = self.loss_func(pred.view(-1),y.view(-1))
            return embeddings, loss
        else:
            pred = self.predictor(embeddings)
            return embeddings, pred

def read_data_from_csv(path):
    data_df = pd.read_csv(path)
    dataset = Dataset.from_dict(data_df)
    return dataset



def train(args):
    os.environ["WANDB_SILENT"] = "true"  
    os.environ["WANDB_ERROR_REPORTING"] = "false"  
    os.makedirs(args.out_dir, exist_ok=True)
    if args.local_rank == 0:
        tb_log_dir =os.getenv("TENSORBOARD_LOG_PATH", "/tensorboard_logs/")
        #tb_log_dir = os.path.join(args.out_dir, "logs")  
        writer = SummaryWriter(log_dir=tb_log_dir)
        #wandb.init( 
            #project="chemical_predict", 
            #name=datetime.datetime.now().strftime('%Y-%m-%d--%H:%M')+'-'+args.data_name
            #)

    # Load the model and tokenizer
    pretrained_model_path = args.pretrained_model_path
    num_epoch = args.num_epoch
    batch_size= args.per_device_train_batch_size
    yield_predictor_path = args.yield_predictor_path
    lr=args.lr
    max_length = args.max_length

    data_path=args.data_path
    data_name = args.data_name
    # Save the base model
    lora_adapter_path = args.lora_adapter_path

    load_ds_dir = args.load_ds_dir
    load_ds_ckpt_id = args.load_ds_ckpt_id

    use_lora = args.use_lora

    log_path = os.path.join(args.out_dir, args.log_file)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
    )
    logger = logging.getLogger()

    # args.global_rank = torch.distributed.get_rank()
    get_accelerator().set_device(args.local_rank)
    device = torch.device(get_accelerator().device_name(), args.local_rank)
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    # torch.distributed.init_process_group(backend='nccl')
    deepspeed.init_distributed()

    print('using device', device)

    print('Load model...')
    logger.info('Load model...')


    if use_lora:

        # Define LoRA configuration
        if not os.path.exists(lora_adapter_path):
            model = AutoModel.from_pretrained(pretrained_model_path)
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
            print(model)
            logger.info(model)
            # Apply LoRA to the model\
            print('LoRA configuring...')
            logger.info('LoRA configuring...')
            lora_config = LoraConfig(
                r=8,  # Rank of the low-rank matrix
                lora_alpha=16,
                lora_dropout=0.1,
                # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                target_modules="all-linear",
                # bias="none",
                # modules_to_save=["classifier"]
                )
            model = get_peft_model(model,lora_config)
        else:
            print(f"Load LoRA from {lora_adapter_path}")
            logger.info(f"Load LoRA from {lora_adapter_path}")
            lora_config = PeftConfig.from_pretrained(lora_adapter_path)
            model = AutoModel.from_pretrained(lora_config.base_model_name_or_path)
            model = PeftModel.from_pretrained(model, lora_adapter_path, is_trainable=True)
        model.print_trainable_parameters()  # Print the number of trainable parameters to confirm LoRA is applied
    
    world_size = torch.distributed.get_world_size()
    rank = args.local_rank


    predictor = YieldPredLayer(4096,1024,1).to(device).train()  # 4096 => 1024 => 256 => 1, act= relu
    if os.path.exists(yield_predictor_path):
        predictor.load_state_dict(torch.load(yield_predictor_path))
    model = LlamaWithLoss(model, predictor)


    print(model)


    # Data
    print('Load data from...', os.path.join(data_path, data_name, 'train.csv'))
    logger.info(f'Load data ...')
    # train_data = Dataset.load_from_disk(os.path.join(data_path, data_name))
    train_data = read_data_from_csv(os.path.join(data_path, data_name, 'train.csv'))
    # DDP sampler
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
    trainloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)



    # 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    train_batch_size = args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
    print("Train_batch_size: ", train_batch_size)
    logger.info(f'Train_batch_size: {train_batch_size}')

    with open(args.deepspeed_config, 'r') as f:
        ds_config = json.load(f)
    ds_config['train_batch_size'] = train_batch_size
    ds_config['scheduler']['params']['total_num_steps'] = num_epoch * len(trainloader) / (args.per_device_train_batch_size * args.gradient_accumulation_steps / world_size)
    model, optimizer, _, _ = deepspeed.initialize(
        # args=args,
        model=model,
        optimizer=optimizer,
        config_params=ds_config,
        # model_parameters=all_parameters,
        # dist_init_required=True,
    )

    if load_ds_dir is not None and os.path.exists(load_ds_dir):
        print(f"Load deepspeed checkpoint from {load_ds_dir}")
        logger.info(f'Load deepspeed checkpoint from {load_ds_dir}')
        model.load_checkpoint(load_ds_dir, load_ds_ckpt_id)


    

    test_data = read_data_from_csv(os.path.join(data_path, data_name, 'test.csv'))
    test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False)
    testloader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler)

    best_eval_loss = torch.inf
    
    for epoch in range(num_epoch):
        print(f'Training Epoch {epoch}:')
        logger.info(f'Training Epoch {epoch}:')
        # DDP set epoch
        trainloader.sampler.set_epoch(epoch)

        total_loss = torch.scalar_tensor(0)
        losses = []
        run_time = 0
        for i, batch_data in enumerate(trainloader):
            model.train()
            model.llama.train()
            model.predictor.train()

            start_time = time.time()

            # y = batch_data['yield'].to(torch.float).to(device)
            # reaction = batch_data['reaction']
            # condition = batch_data['condition']
            # prompts = [reaction[k] + condition[k] for k in range(len(reaction))]
            prompts = batch_data['instruction']
            y = batch_data['output'].to(torch.float).to(device)

            inputs = tokenizer(prompts, max_length=max_length, padding='longest', truncation=True, return_tensors="pt").to(device)

            # Get embeddings
            if model.fp16_enabled():
                y = y.half() 

            _, loss = model(inputs, y)
            losses.append(loss.cpu().item())


            # Backpropagation and optimization
            # optimizer.zero_grad()
            model.backward(loss)
            model.step()
            # clip_grad_norm_(model.parameters(), max_grad_norm)
            # optimizer.step()
            # scheduler.step()

            end_time = time.time()
            run_time += (end_time-start_time)/ (60*60)
            print(f"Rank {rank}, Epoch {epoch}:{i+1}/{len(trainloader)}-step, \tloss:{loss}, \truning time:{end_time-start_time}s, \tleft_time:{(run_time/(i+1)) *(len(trainloader)-i-1)}h ")
            logger.info(f"Rank {rank}, Epoch {epoch}:{i+1}/{len(trainloader)}-step, \tloss:{loss}, \truning time:{end_time-start_time}s, \tleft_time:{(run_time/(i+1)) *(len(trainloader)-i-1)}h ")

            if args.local_rank == 0: 
                #wandb.log({'train_loss': loss,'current_lr': optimizer.param_groups[0]['lr']})
                global_step = epoch * len(trainloader) + i  
                writer.add_scalar('train_loss', loss.item(), global_step)  
                writer.add_scalar('train_lr', optimizer.param_groups[0]['lr'], global_step)

            if ((i+1)+(epoch*len(trainloader))) % 5000 == 0:
                ckpt_path = os.path.join(args.save_root, 'checkpoints')

                save_ckpt_with_limited_files(model,ckpt_path,epoch,i,(i+1)+(epoch*len(trainloader)))
                print(f"Model saved to: {ckpt_path}")
                logger.info(f"Model saved to: {ckpt_path}")

        if (epoch) % 50 == 0:
            # Evaluate model
            model.eval()
            model.llama.eval()
            model.predictor.eval()
            eval_losses = []
            pred_all = []
            target_all = []
            best_r2_score = -1
            print(f"Evaluation ...")
            logger.info(f"Evaluation ...")
            with torch.no_grad():
                testloader.sampler.set_epoch(epoch)
                if args.local_rank == 0:
                    test_iterator = tqdm(testloader, desc="Testing", file=sys.stdout, position=0)
                else:
                    test_iterator = testloader
                for batch_data in test_iterator:
                    prompts = batch_data['instruction']
                    y_true = batch_data['output'].to(torch.float).to(device)

                    inputs = tokenizer(prompts, max_length=max_length, padding='longest', truncation=True, return_tensors="pt").to(device)

                    # Get embeddings
                    if model.fp16_enabled():
                        y_true = y_true.half()

                    embeddings, pred = model(inputs, y_true, return_loss=False)
                    pred_all.append(pred.cpu().numpy())
                    target_all.append(y_true.cpu().numpy())
                    loss = torch.nn.functional.mse_loss(pred.view(-1), y_true.view(-1))
                    eval_losses.append(loss.cpu().item())


            pred_all = np.concatenate(pred_all).reshape(-1)
            target_all = np.concatenate(target_all).reshape(-1)

            r2 = r2_score(target_all.astype(np.float64), pred_all.astype(np.float64))
            r2 = torch.tensor(r2).to(torch.cuda.current_device())
            
            avg_eval_loss = sum(eval_losses) / len(eval_losses)
            print(f"Avg Eval Loss: {avg_eval_loss}, R2: {r2}")
            logger.info(f"Avg Eval Loss: {avg_eval_loss}, R2: {r2}")
            dist.all_reduce(r2, op=dist.ReduceOp.SUM)

            if r2 > best_r2_score:
                best_r2_score = r2


            if args.local_rank == 0: 
                print(r2)
                r2 = r2 / world_size
                #wandb.log({'eval_loss':avg_eval_loss, 'eval R2': r2})
                writer.add_scalar('eval_loss', avg_eval_loss, epoch)  
                writer.add_scalar('eval_R2', r2, epoch)
                if avg_eval_loss < best_eval_loss:
                    best_eval_loss = avg_eval_loss
            


        print(f"Avg  Loss on Epoch {epoch}: {sum(losses) / len(losses)}")
        logger.info(f"Avg Loss on Epoch {epoch}: {sum(losses) / len(losses)}")
        



    ckpt_path = os.path.join(args.out_dir, 'ckpt')
    model.save_checkpoint(ckpt_path, tag='ckpt', save_latest=False)
    print(f"Model saved to: {ckpt_path}")

    logger.info(f"Model saved to: {ckpt_path}")

    if args.local_rank == 0: 
        #wandb.log({'train_loss_on_epoch': total_loss.item() / (i+1)})
        #wandb.finish()
        writer.close()

        final_loss = best_eval_loss
        final_r2 = r2
        print(f"Final loss: {final_loss}, R2: {final_r2}")
    
        final_infos = {
            "Chemical prediction":{
                "means":{
                    "loss": final_loss,
                    "r2_score": float(final_r2)
                }
            }
        }
    
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, 'final_info.json'), 'w') as f:
            json.dump(final_infos, f, indent=4) 
    
        
        

def main(args):
    
    # world_size = args.world_size
    print('start training')


    # deepspeed.launcher.executable.main(train, args=(world_size, args), nprocs=world_size)
    # mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
    # os.environ['TOKENIZERS_PARALLELISM='] = "true"
    # os.environ['RANK'] = os.environ['SLURM_PROCID']
    # os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    # os.environ['MASTER_PORT'] = str(random.randint(1024, 65535))
    # os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
    train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distributed Data Parallel Training")
    
    parser.add_argument("--pretrained_model_path", default='/mnt/cache/Chemllm/reaction_condition_recommendation/src/step1_llama3_8b_0916_yearly_pistachio_ep3')
    # parser.add_argument("--pretrained_model_path", default='/mnt/hwfile/ai4chem/share/jianpeng/llama3_8b_0916_lora_yield_pred_ds')
    parser.add_argument("--lora_adapter_path", default="/mnt/hwfile/ai4chem/share/jianpeng/llama3_8b_0916_lora_yield_pred_ds/lora_adapter")
    parser.add_argument("--yield_predictor_path", default="/mnt/hwfile/ai4chem/share/jianpeng/llama3_8b_0916_lora_yield_pred_ds/predictor.pt")
    parser.add_argument("--num_epoch", type=int, default=2)
    parser.add_argument("--local_rank", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--data_path", default='/mnt/petrelfs/chenjianpeng/cjp/LLaMA-Factory/train_regression/data4regression')
    parser.add_argument("--data_name", default='suzuki_miyaura_600')
    parser.add_argument("--save_root", default="/mnt/hwfile/ai4chem/share/jianpeng/llama3_8b_0916_lora_yield_pred_ds")
    parser.add_argument("--base_model_save_path", default="base_model")
    parser.add_argument("--lora_adapter_save_path", default="lora_adapter")
    parser.add_argument('--use_lora', type=int, default=1)
    parser.add_argument('--log_file', type=str, default="training_ds.log")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=3000)
    parser.add_argument('--load_ds_dir', type=str, default=None)
    parser.add_argument('--load_ds_ckpt_id', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default="run_0")
    

    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print("Origin error in main process:", flush=True)
        traceback.print_exc(file=open(os.path.join(args.out_dir, "traceback.log"), "w"))
        raise
