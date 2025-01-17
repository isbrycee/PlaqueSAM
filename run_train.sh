export CUDA_VISIBLE_DEVICES=0


python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_t+_MOSE_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 1
