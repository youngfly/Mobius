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
    test_t2v.py \
    --init_from_pretrained_2d=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/cheryldeng/ckpts/jp-chinese_stable_diffusion_v1.0/ \
    --ckpt_resume_dir=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/acceleryang/yyr/.cache/huggingface/hub/models--damo-vilab--text-to-video-ms-1.7b/snapshots/338683c3175cde59b407a699af8fe68f4a2b8f74 \
    --prompt_file=./prompts_cn.txt \
    --prompt_suffix="." \
    --save_img_dir=../vis/cn_cs_damo
    # --convert_to_hf_prerained=../infer/data/cn_spatial_frozen_mini_hf

# conda run -n diffusers \
# accelerate launch --multi_gpu --num_processes=2 --num_machines=1 \
#     test_t2v.py \
#     --init_from_pretrained_2d=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/acceleryang/yyr/.cache/huggingface/hub/models--damo-vilab--text-to-video-ms-1.7b-chinese/snapshots/338683c3175cde59b407a699af8fe68f4a2b8f74 \
#     --resume_from_checkpoint="None" \
#     --ckpt_resume_dir=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/acceleryang/yyr/.cache/huggingface/hub/models--damo-vilab--text-to-video-ms-1.7b-chinese/snapshots/338683c3175cde59b407a699af8fe68f4a2b8f74 \
#     --prompt_file=./prompts_reality.txt \
#     --prompt_suffix=", 纯色背景" \
#     --save_img_dir=../vis/cn_cs_damo
#     # --convert_to_hf_prerained=../infer/data/cn_spatial_frozen_mini_hf


# conda run -n diffusers \
# accelerate launch --multi_gpu --num_processes=2 --num_machines=1 \
#     test_t2v.py \
#     --init_from_pretrained_2d=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/cheryldeng/ckpts/jp-chinese_stable_diffusion_v1.0/ \
#     --resume_from_checkpoint=checkpoint-125000 \
#     --ckpt_resume_dir=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/acceleryang/yyr/ckpts/text-to-video-synthesis/base_1205_A_100wstep_batch16/output \
#     --prompt_file=./prompts_cn.txt \
#     --prompt_suffix=", 卡通表情, 纯色背景" \
#     --save_img_dir=../vis/cn_cs_damo
#     # --convert_to_hf_prerained=../infer/data/cn_spatial_frozen_mini_hf


