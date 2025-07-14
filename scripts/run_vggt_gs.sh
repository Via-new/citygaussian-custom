get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

PROJECT=VGGT

declare -a scenes=(
    "data/MipNeRF360_vggt/garden"
    "data/MipNeRF360_vggt/flowers"
)

dir="data/MipNeRF360_vggt"
ref_dir="data/MipNeRF360"
post_fix="_mcmc_pose_opt_depth_w1"
config_path="configs/colmap_pose_opt_exp_ds4_mcmc_depth.yaml"

for data_path in $dir/*; do
# for data_path in "${scenes[@]}"; do
    while [ -d "$data_path" ]; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Start running GS on '$data_path'"
            # python utils/get_depth_scales.py $data_path
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$gpu_id python main.py fit \
                            --config $config_path \
                            --data.path $data_path \
                            -n $(basename $data_path)_vggt$post_fix \
                            --output outputs/$(basename $dir)$post_fix \
                            --logger wandb \
                            --project $PROJECT \
                            --data.train_max_num_images_to_cache 1024 &
                            # --data.parser.init_args.ref_path $ref_dir/$(basename $data_path) \
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

python tools/gather_wandb.py --output_path outputs/$(basename $dir)$post_fix

for data_path in $dir/*; do
# for data_path in "${scenes[@]}"; do
    while [ -d "$data_path" ]; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Start evaluating GS on '$data_path'"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
                            --config outputs/$(basename $dir)$post_fix/$(basename $data_path)_vggt$post_fix/config.yaml \
                            --save_val --val_train &
                            # --data.parser.init_args.ref_path $data_path \
                            # --model.metric internal.metrics.PoseOptMetrics \
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 30
            break
        else
            echo "No GPU available at the moment. Retrying in 2 minute."
            sleep 30
        fi
    done
done
wait

python tools/gather_results.py outputs/$(basename $dir)$post_fix --format_float
# python tools/wandb_sync.py --output_path outputs/$(basename $dir)_$post_fix