# Modified from AcademiCodec

#!/bin/bash
# qsub -I -q GPU-S -l select=1:ngpus=2

# source path.sh
# set -e

# .lst save the wav path.
input_training_file="train-all.lst"
input_validation_file="dev-all.lst"
other="test_vastai"
log_file="20250218"

log_root="checkpoints/${other}/${log_file}"

#mode=debug
mode=train

if [ "${mode}" == "debug" ]; then
  ## debug
  echo "Debug"
  log_root=${log_root}_debug
  export CUDA_VISIBLE_DEVICES=0
  python ${BIN_DIR}/train.py \
    --config ${config} \
    --checkpoint_path ${log_root} \
    --input_training_file ${input_training_file} \
    --input_validation_file ${input_validation_file} \
    --checkpoint_interval 100 \
    --summary_interval 10 \
    --validation_interval 100 \

elif [ "$mode" == "train" ]; then
  ## train
  echo "Train model..."
  export CUDA_VISIBLE_DEVICES=0
  python ${BIN_DIR}/train_bestckpt.py > ${log_file}.log \
    --config ${config} \
    --checkpoint_path ${log_root} \
    --input_training_file ${input_training_file} \
    --input_validation_file ${input_validation_file} \
    --summary_interval 50 \
    --validation_interval 100 \
    --training_epochs 10
fi

# elif [ "$mode" == "train" ]; then
#   ## train
#   echo "Train model..."
#   # export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#   export CUDA_VISIBLE_DEVICES=0,1
#   python ${BIN_DIR}/train.py > ${log_file}.log \
#     --config ${config} \
#     --checkpoint_path ${log_root} \
#     --input_training_file ${input_training_file} \
#     --input_validation_file ${input_validation_file} \
#     --checkpoint_interval 10 \
#     --validation_interval 5
# fi

  # python ${BIN_DIR}/train_no_tensorboard.py > ${log_file}.log \

tensorboard --logdir ${log_root}/logs/ --port=6006 --bind_all