CUDA_VISIBLE_DEVICES=1 python train.py ../train2.bin ../valid2.bin --gpus 1 --threads 8 --progress_bar_refresh_rate 20 --default_root_dir models --batch-size 16384 --architecture leiser
