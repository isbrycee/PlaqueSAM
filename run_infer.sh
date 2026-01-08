if [ "$CPU" == "1" ]; then
    export CUDA_VISIBLE_DEVICES=-1
    NUM_GPUS=0
else
    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    NUM_GPUS=1
fi

python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_t+_MOSE_finetune_infer.yaml \
    --use-cluster 0 \
    --num-gpus $NUM_GPUS