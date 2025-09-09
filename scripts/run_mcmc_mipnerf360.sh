get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

PROJECT=VGGT

dir="data/MipNeRF360_vggt_opt_gt_align"
ref_dir="$dir"
post_fix="_vggt_mcmc_x"
config_path="configs/colmap_exp_ds4_mcmc.yaml"

declare -a scenes=(
    "$dir/bicycle"
    "$dir/bonsai"
    "$dir/counter"
    "$dir/flowers"
    "$dir/garden"
    "$dir/kitchen"
    "$dir/room"
    "$dir/stump"
    "$dir/treehill"
)

declare -a cap_maxs=(
    6_543_652
    1_251_501
    1_185_185
    3_854_928
    5_838_855
    1_796_567
    1_566_821
    5_527_031
    3_897_537
)

declare -a ds_rates=(
    4
    2
    2
    4
    4
    2
    2
    4
    4
)

# for i in "${!scenes[@]}"; do
#     data_path=${scenes[$i]}
#     ds_rate=${ds_rates[$i]}
#     # python utils/get_depth_scales.py $data_path
#     rm -rf $data_path/images_$ds_rate
#     python utils/image_downsample.py $data_path/images --factor $ds_rate &
# done
# wait

for i in "${!scenes[@]}"; do
    data_path=${scenes[$i]}
    cap_max=${cap_maxs[$i]}
    ds_rate=${ds_rates[$i]}
    while [ -d "$data_path" ]; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Start running GS on '$data_path'"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$gpu_id python main.py fit \
                            --config $config_path \
                            --data.path $data_path \
                            --data.parser.init_args.down_sample_factor $ds_rate \
                            --model.density.init_args.cap_max $cap_max \
                            -n $(basename $data_path)$post_fix \
                            --output outputs/$(basename $dir)$post_fix \
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

# python tools/gather_wandb.py --output_path outputs/$(basename $dir)$post_fix

rm -rf outputs/$(basename $dir)${post_fix}/*/test
for data_path in $dir/*; do
# for data_path in "${scenes[@]}"; do
    while [ -d "$data_path" ]; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Start evaluating GS on '$data_path'"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
                            --config outputs/$(basename $dir)$post_fix/$(basename $data_path)$post_fix/config.yaml \
                            --save_val &
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