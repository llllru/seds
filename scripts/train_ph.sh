# learnable topology disabled:
# --learnable_topology --topology_res_scale 0.1
DATA_PATH=""
TIME_NOW=$(date +%Y%m%d%H%M%S)
CUDA_VISIBLE_DEVICES="0,1" PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29558 \
main_task_retrieval.py --do_train --num_thread_reader=64 \
--epochs=200 --batch_size=48 --n_display=10 \
--data_path data_ph \
--features_path "/media/hdd1/RTMW-POSE/phoenix_pose_rtmw" \
--output_dir result_train/ph \
--signbert --init_sign_model ckpts/pretrain_signbert.pth \
--fusion_type 'gloss_atten' \
--lr 1e-5 --sign_lr 1e-4 \
--max_words 32 --feature_len 64 --max_length_frames 300 \
--slide_windows 16 --windows_stride 1 \
--original_size_w 210 --original_size_h 260 \
--crop_size 256 --frames_threshold 0.1 --threshold 0.3 \
--batch_size_val 48 \
--text_aug true \
--datatype ph_pose --coef_lr 1. --freeze_layer_num 0 \
--linear_patch 2d --sim_header Filip \
--pretrained_clip_name ViT-B/32 \