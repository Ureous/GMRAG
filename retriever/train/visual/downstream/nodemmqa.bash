GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=172.17.0.5 # your master address
MASTER_PORT=6001 # your master port
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/root/data1/Code/Multimodalqa/ # the directory of query_train.jsonl
SAVE_PATH=/root/data1/Code/Multimodalqa/flod_save_models/ # your saving path 
IMAGE_PATH=/root/data1/Code/Multimodalqa/MMQA/final_dataset_images/ # the training image directory
EPOCH=50
RESUME_PATH=/root/data2/BAAI.bge-visualized/Visualized_base_en_v1.5.pth # pre-trained visualized bge weights
SAVE_STEPS=100
GROUP_SIZE=5 # = one (positive sample) + number (of hard negative samples)
BSZ_PERGPU=30
LR=2e-5

Training_Dir=/root/data1/Code/Mumu/FlagEmbedding/FlagEmbedding/visual  #your training dir
DeepSpeedConfig=/root/data1/Code/Mumu/FlagEmbedding/FlagEmbedding/visual/downstream/deepspeedconfig.json #your deepspeed config file
cd $Training_Dir
# Data and model


mkdir $SAVE_PATH
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

export LAUNCHER="torchrun \
    $DISTRIBUTED_ARGS \
    "

full_options="
  --output_dir $SAVE_PATH \
  --bge_model_name_or_path  /root/data2/BAAI.bge-base-en-v1.5 \
  --visual_model_name_or_path  EVA02-CLIP-B-16 \
  --dataloader_num_workers 1  \
  --train_data $DATA_PATH \
  --train_data_image $IMAGE_PATH \
  --train_group_size $GROUP_SIZE
  --learning_rate $LR \
  --fp16 \
  --per_device_train_batch_size $BSZ_PERGPU \
  --dataloader_drop_last True \
  --normlized True \
  --temperature 0.02 \
  --logging_steps 10 \
  --num_train_epochs $EPOCH \
  --negatives_cross_device \
  --train_text_tower True  \
  --train_vision_tower True \
  --resume_path $RESUME_PATH \
  --save_steps $SAVE_STEPS \
  --deepspeed $DeepSpeedConfig \
  "
  # --gradient_checkpointing \
run_cmd="$LAUNCHER -m downstream.run_mmqa ${full_options}"
echo ${run_cmd}
eval ${run_cmd} 2>&1 | tee $SAVE_PATH/output_$NODE_RANK.log



set +x

