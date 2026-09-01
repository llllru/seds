DATA_PATH=""
TIME_NOW=$(date +%Y%m%d%H%M%S)
CUDA_VISIBLE_DEVICES="4,5" python -m torch.distributed.launch --nproc_per_node=2 --master_port 28463 \
main_task_retrieval.py --do_train --num_thread_reader=64 \
--epochs=10 --batch_size=32 --n_display=10 \
--data_path data_h2 \
--features_path "/media/hdd2/lyr2025/seds_data/How2Sign/processed_videos_256/RTMpose/Pose_all_24rates/" \
--features_RGB_path "/media/hdd2/lyr2025/seds_data/How2Sign/processed_videos_256/I3D_features/" \
--output_dir result_train/h2s_posergbfusion_fullpath_gate_only_lr3e4 \
--signbert --init_model "/media/hdd2/lyr2025/seds_data/h2s_best_model.bin" \
--fusion_type 'gloss_atten' --dynamic_modal_gate --full_path_modal_gate --gate_hidden_dim 256 \
--gate_only_train --gate_lr 3e-4 \
--lr 1e-5 --sign_lr 1e-4 \
--max_words 32 --feature_len 64 --max_length_frames 300 \
--slide_windows 16 --windows_stride 1 --original_size_w 256 --original_size_h 256 \
--crop_size 256 --frames_threshold 0.1 --threshold 0.4 \
--batch_size_val 64 \
--datatype h2s_pose --coef_lr 1. --freeze_layer_num 0 \
--linear_patch 2d --sim_header Filip \
--pretrained_clip_name ViT-B/32
