<<<<<<< HEAD
#!/bin/bash
out_dir=$1
rseed=2024
ROOT=.

# Default data path - you may need to download the dataset manually
DATA_PATH="datasets/household_power_consumption.txt"

python $ROOT/experiment.py \
--out_dir ${out_dir} \
--data_path ${DATA_PATH} \
--seq_length 48 \
--pred_length 10 \
--batch_size 2048 \
--max_epochs 5 \
--patience 10 \
--val_interval 1
=======
#!/bin/bash
out_dir=$1
rseed=2024
ROOT=.

# Default data path - you may need to download the dataset manually
DATA_PATH="datasets/household_power_consumption.txt"

python $ROOT/experiment.py \
--out_dir ${out_dir} \
--data_path ${DATA_PATH} \
--seq_length 48 \
--pred_length 10 \
--batch_size 2048 \
--max_epochs 5 \
--patience 10 \
--val_interval 1
>>>>>>> ssy
