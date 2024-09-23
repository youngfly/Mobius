set -xe

export NCCL_IB_DISABLE=1 
export NCCL_DEBUG=INFO 
export NCCL_P2P_DISABLE=1

# export HOME=/apdcephfs/share_47076/cheryldeng/
# export HF_HOME=/apdcephfs/share_47076/cheryldeng/.cache/huggingface/
# export HUGGINGFACE_HUB_CACHE=/apdcephfs/share_47076/cheryldeng/.cache/huggingface/hub
# # export TRANSFORMERS_CACHE=/apdcephfs/share_47076/cheryldeng/.cache/huggingface/transformers
# export CONDA=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/cheryldeng/soft/miniconda3/bin/conda


export HOME=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/cheryldeng/
export HF_HOME=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/cheryldeng/.cache/huggingface/
export HUGGINGFACE_HUB_CACHE=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/cheryldeng/.cache/huggingface/hub
# export TRANSFORMERS_CACHE=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/cheryldeng/.cache/huggingface/transformers
export CONDA=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/cheryldeng/soft/miniconda3/bin/conda

echo "NODE_RANK: $RANK"
echo "GPU_NUM: $GPU_NUM"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"


# 单机多卡
echo "begin to train the model"
# $CONDA run -n diffusers \
conda run -n diffusers \
accelerate launch --multi_gpu --num_processes=2 --main_process_port=29000 --num_machines=1 \
    train_t2v_adapter.py \
    --init_from_pretrained_2d=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/cheryldeng/ckpts/jp-chinese_stable_diffusion_v1.0/ \
    --dataloader_num_workers=1 \
    --resolution=256 \
    --train_batch_size=2 \
    --max_train_steps=50000 \
    --gradient_accumulation_steps=2 \
    --resume_from_checkpoint=latest \
    --checkpointing_steps=500 \
    --dataset_file ../mydata/emoticon_caption_llava.finetune.json \
    --log_file logs/adapter.txt \
    --output_dir=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/acceleryang/yyr/ckpts/text-to-video-synthesis/sta_adapter_debug/output


