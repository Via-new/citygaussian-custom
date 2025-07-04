get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

PROJECT=VGGT

declare -a scenes=(
    "data/MipNeRF360/garden"
    "data/MipNeRF360/flowers"
)

dir="data/MipNeRF360"

for data_path in $dir/*; do
# for data_path in "${scenes[@]}"; do
    while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Start running GS on '$data_path'"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$gpu_id python main.py fit \
                            --config configs/colmap_exp_ds4.yaml \
                            --data.path $data_path \
                            -n $(basename $data_path) \
                            --output outputs/$(basename $dir) \
                            --logger wandb \
                            --project $PROJECT \
                            --data.train_max_num_images_to_cache 1024 &
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 60
            break
        else
            echo "No GPU available at the moment. Retrying in 2 minute."
            sleep 60
        fi
    done
done
wait

for data_path in $dir/*; do
# for data_path in "${scenes[@]}"; do
    while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Start evaluating GS on '$data_path'"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
                            --config outputs/$(basename $dir)/$(basename $data_path)/config.yaml \
                            --save_val &
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 60
            break
        else
            echo "No GPU available at the moment. Retrying in 2 minute."
            sleep 60
        fi
    done
done
wait

python tools/gather_results.py outputs/$(basename $dir) --format_float
python tools/wandb_sync.py --output_path outputs/$(basename $dir)