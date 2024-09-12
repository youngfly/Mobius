set -xe

export NCCL_IB_DISABLE=1 
export NCCL_DEBUG=INFO 
export NCCL_P2P_DISABLE=1

export HOME=/apdcephfs/share_47076/cheryldeng/
export HF_HOME=/apdcephfs/share_47076/cheryldeng/.cache/huggingface/
export HUGGINGFACE_HUB_CACHE=/apdcephfs/share_47076/cheryldeng/.cache/huggingface/hub
# export TRANSFORMERS_CACHE=/apdcephfs/share_47076/cheryldeng/.cache/huggingface/transformers
export CONDA=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/cheryldeng/soft/miniconda3/bin/conda


echo "NODE_RANK: $RANK"
echo "GPU_NUM: $GPU_NUM"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"



# 多机多卡
$CONDA run -n diffusers \
python -m torch.distributed.run \
   --nproc_per_node ${GPU_NUM} --nnodes ${WORLD_SIZE} --master_addr ${MASTER_ADDR} \
   --master_port ${MASTER_PORT} --node_rank ${RANK} --max_restarts=1 \
   train_text_to_video.py \
   --init_from_pretrained_2d=/apdcephfs/share_47076/cheryldeng/ckpts/jp-chinese_stable_diffusion_v1.0 \
   --dataloader_num_workers=8 \
   --resolution=256 \
   --train_batch_size=2 \
   --max_train_steps=1000000 \
   --gradient_accumulation_steps=2 \
   --resume_from_checkpoint=latest \
   --checkpointing_steps=500 \
   --dataset_file ../mydata/emoticon_caption_gif_nframe.json \
   --log_file logs/log.txt \
   --output_dir=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/cheryldeng/ckpts/text-to-video-synthesis/cn_spatial_frozen/output

