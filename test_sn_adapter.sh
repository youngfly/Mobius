set -xe

export NCCL_IB_DISABLE=1 
export NCCL_DEBUG=INFO 
export NCCL_P2P_DISABLE=1

export HOME=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/cheryldeng/
export HF_HOME=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/cheryldeng/.cache/huggingface/
export HUGGINGFACE_HUB_CACHE=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/cheryldeng/.cache/huggingface/hub
# export TRANSFORMERS_CACHE=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/cheryldeng/.cache/huggingface/transformers

echo "NODE_RANK: $RANK"
echo "GPU_NUM: $GPU_NUM"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"


echo "begin to inference, 2gpus,"
conda run -n diffusers \
accelerate launch --multi_gpu --num_processes=2 --num_machines=1 \
    test_t2v_adapter.py \
    --init_from_pretrained_2d=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/cheryldeng/ckpts/jp-chinese_stable_diffusion_v1.0/ \
    --resume_from_checkpoint=checkpoint-210000 \
    --ckpt_resume_dir=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/acceleryang/yyr/ckpts/text-to-video-synthesis/adapter_1213_100wstep_batch16_st/output \
    --prompt_file=./my_prompt_cn1.txt \
    --prompt_suffix=", 卡通表情, 纯色背景" \
    --save_img_dir=../vis/cn_spacetime_ft/28W
    # --convert_to_hf_prerained=../infer/data/cn_spatial_frozen_mini_hf