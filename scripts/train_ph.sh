DATA_PATH=""
TIME_NOW=$(date +%Y%m%d%H%M%S)
CUDA_VISIBLE_DEVICES="1,2,3,4" python -m torch.distributed.launch --nproc_per_node=4 --master_port 29558 \
main_task_retrieval.py --do_train --num_thread_reader=64 \
--epochs=200 --batch_size=96 --n_display=10 \
--data_path data_ph \
--features_path "./PHOENIX-2014-T/features/RTM_Keypoints/" \
--output_dir result_train/ph \
--signbert --init_sign_model ckpts/pretrain_signbert.pth \
--fusion_type 'gloss_atten' \
--lr 1e-5 --sign_lr 1e-4 \
--max_words 32 --feature_len 64 --max_length_frames 300 \
--slide_windows 16 --windows_stride 1 \
--crop_size 256 --frames_threshold 0.1 --threshold 0.4 \
--batch_size_val 64 \
--text_aug false \
--datatype ph_pose --coef_lr 1. --freeze_layer_num 0 \
--linear_patch 2d --sim_header Filip \
--pretrained_clip_name ViT-B/32