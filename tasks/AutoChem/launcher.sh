
deepspeed experiment.py \
   --pretrained_model_path '/baseline_ckpt/step1_llama3_8b_0916_yearly_pistachio_ep3' \
   --num_epoch 301 \
   --data_path '/datasets/data4regression' \
   --data_name 'suzuki_miyaura_fg_changes_60' \
   --per_device_train_batch_size 2 \
   --save_root 'result_60' \
   --gradient_accumulation_steps 4 \
   --use_lora 1 \
   --log_file "training_ds.log" \
   --deepspeed_config 'ds_config.json' \
   --out_dir $1
