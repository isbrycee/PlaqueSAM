export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_t+_MOSE_finetune_1st_eval.yaml \
    --use-cluster 0 \
    --num-gpus 1