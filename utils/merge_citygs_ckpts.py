import os
import sys
import add_pypath
import argparse
import torch
import numpy as np
import logging
from tqdm import tqdm
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.citygs_partitioning_utils import CityGSPartitioning, PartitionCoordinates

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Path to the model output directory")
args = parser.parse_args()

# 主路径打印（确认输入路径是否正确）
print(f"===== 输入的模型输出目录：{args.path} =====")
checkpoint_dir = os.path.join(args.path, "blocks")
print(f"===== 分区根目录（blocks）：{checkpoint_dir} =====")

# 验证blocks目录是否存在
if not os.path.exists(checkpoint_dir):
    print(f"错误：blocks目录不存在！路径：{checkpoint_dir}")
    exit(1)
if not os.path.isdir(checkpoint_dir):
    print(f"错误：{checkpoint_dir} 不是目录！")
    exit(1)

# 日志配置
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler(os.path.join(os.path.dirname(checkpoint_dir), "merge.log"))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

# 替换原有的“搜索检查点”循环，直接手动查找.ckpt文件
logger.info("开始搜索检查点文件...")
max_iteration = -1
checkpoint_files = []

print(f"\n===== blocks目录下的内容：{os.listdir(checkpoint_dir)} =====")

for block_dir in os.listdir(checkpoint_dir):
    block_path = os.path.join(checkpoint_dir, block_dir)
    print(f"\n----- 正在处理分区文件夹：{block_path} -----")
    
    if not os.path.isdir(block_path):
        print(f"跳过 {block_path}：不是目录")
        continue
    
    checkpoints_subdir = os.path.join(block_path, "checkpoints")
    print(f"检查点子文件夹路径：{checkpoints_subdir}")
    
    if not os.path.exists(checkpoints_subdir) or not os.path.isdir(checkpoints_subdir):
        print(f"警告：{checkpoints_subdir} 不存在或不是目录，跳过")
        continue
    
    # 手动遍历checkpoints文件夹中的所有.ckpt文件
    subdir_files = os.listdir(checkpoints_subdir)
    ckpt_files_in_subdir = [f for f in subdir_files if f.endswith(".ckpt")]
    print(f"checkpoints子文件夹内的.ckpt文件列表：{ckpt_files_in_subdir}")
    
    if not ckpt_files_in_subdir:
        print(f"警告：{checkpoints_subdir} 中没有.ckpt文件，跳过")
        continue
    
    # 从所有.ckpt文件中提取最大step的文件
    max_step_in_block = -1
    best_ckpt_in_block = None
    for ckpt_file in ckpt_files_in_subdir:
        try:
            # 提取step（适配格式：epoch=xxx-step=xxxx.ckpt）
            step_str = ckpt_file.split("step=")[-1].split(".ckpt")[0]
            step = int(step_str)
            if step > max_step_in_block:
                max_step_in_block = step
                best_ckpt_in_block = os.path.join(checkpoints_subdir, ckpt_file)
        except Exception as e:
            print(f"警告：解析文件 {ckpt_file} 的step失败，错误：{e}，跳过该文件")
            continue
    
    if best_ckpt_in_block is None:
        print(f"警告：{checkpoints_subdir} 中没有有效step的.ckpt文件，跳过")
        continue
    
    print(f"分区 {block_dir} 中最大step的文件：{best_ckpt_in_block}（step={max_step_in_block}）")
    
    # 更新全局最大step和检查点列表
    if max_step_in_block > max_iteration:
        max_iteration = max_step_in_block
        checkpoint_files = [best_ckpt_in_block]
        print(f"更新全局最大step为：{max_iteration}")
    elif max_step_in_block == max_iteration:
        checkpoint_files.append(best_ckpt_in_block)
        print(f"添加到检查点列表：{best_ckpt_in_block}")

# 验证结果
checkpoint_files = sorted(checkpoint_files)
print(f"\n===== 最终收集到的检查点文件列表：{checkpoint_files} =====")
assert len(checkpoint_files) > 0, "未找到任何有效.ckpt检查点文件"

logger.info(f"找到的检查点文件：{checkpoint_files}")

logger.info(f"找到的检查点文件：{checkpoint_files}")

# 后续合并逻辑（保持不变）
import add_pypath
import torch
from internal.models.gaussian import Gaussian

is_new_model = True
param_list_key_by_name = {}
extra_param_list_key_by_name = {}
optimizer_state_exp_avg_list_key_by_index = {}
optimizer_state_exp_avg_sq_list_key_by_index = {}
density_controller_state_list_key_by_name = {}
number_of_gaussians = []

ckpt = torch.load(checkpoint_files[0], map_location="cpu")
dataparser_config = ckpt["datamodule_hyper_parameters"]["parser"]
partitions = torch.load(os.path.join(os.path.dirname(dataparser_config.image_list), "partitions.pt"))
partition_coordinates = PartitionCoordinates(
    id=partitions['partition_coordinates']['id'],
    xy=partitions['partition_coordinates']['xy'],
)
partition_bounding_boxes = partition_coordinates.get_bounding_boxes(partitions['scene_config']['partition_size'], enlarge=0.)

del ckpt, dataparser_config

for i in tqdm(checkpoint_files, desc="Loading checkpoints"):
    ckpt = torch.load(i, map_location="cpu")
    dataparser_config = ckpt["datamodule_hyper_parameters"]["parser"]
    xyz_gs = ckpt['state_dict']['gaussian_model.gaussians.means'] @ partitions['extra_data']['rotation_transform'][:3, :3].T

    if partitions['scene_config']['contract']:
        xyz_gs = CityGSPartitioning.contract_to_unisphere(xyz_gs[:, :2], partitions['scene_config']['aabb'], ord=torch.inf)

    mask_preserved = CityGSPartitioning.is_in_bounding_boxes(
        bounding_boxes=partition_bounding_boxes,
        coordinates=xyz_gs[:, :2],
    )[dataparser_config.block_id]
    
    property_names = []
    gaussian_property_dict_key_prefix = "gaussian_model.gaussians."
    density_controller_state_dict_key_prefix = "density_controller."
    for key, value in ckpt["state_dict"].items():
        if key.startswith(gaussian_property_dict_key_prefix):
            param_list_key_by_name.setdefault(key, []).append(value[mask_preserved])
            property_names.append(key[len(gaussian_property_dict_key_prefix):])
        elif key.startswith(density_controller_state_dict_key_prefix):
            param_list_key_by_name.setdefault(key, []).append(value)

    for optimizer_idx, optimizer in enumerate(ckpt["optimizer_states"]):
        for param_group_idx, param_group in enumerate(optimizer["param_groups"]):
            if param_group["name"] not in property_names:
                continue

            property_names.remove(param_group["name"])
            state = optimizer["state"][param_group_idx]

            optimizer_state_exp_avg_list_key_by_index.setdefault(optimizer_idx, {}).setdefault(param_group_idx, []).append(state["exp_avg"])
            optimizer_state_exp_avg_sq_list_key_by_index.setdefault(optimizer_idx, {}).setdefault(param_group_idx, []).append(state["exp_avg_sq"])

        if len(property_names) == 0:
            break

    number_of_gaussians.append(mask_preserved.sum().item())

logger.info("Merging Gaussians and density controller states...")
ckpt["datamodule_hyper_parameters"]["parser"] = torch.load(ckpt['hyper_parameters']['initialize_from'], map_location="cpu")["datamodule_hyper_parameters"]["parser"]
for i in param_list_key_by_name:
    ckpt["state_dict"][i] = torch.concat(param_list_key_by_name[i], dim=0)
if is_new_model is True:
    logger.info("Merging optimizers...")
    for optimizer_idx in optimizer_state_exp_avg_list_key_by_index.keys():
        for param_group_idx in optimizer_state_exp_avg_list_key_by_index[optimizer_idx].keys():
            ckpt["optimizer_states"][optimizer_idx]["state"][param_group_idx]["exp_avg"] = torch.concat(
                optimizer_state_exp_avg_list_key_by_index[optimizer_idx][param_group_idx],
                dim=0,
            )
            ckpt["optimizer_states"][optimizer_idx]["state"][param_group_idx]["exp_avg_sq"] = torch.concat(
                optimizer_state_exp_avg_sq_list_key_by_index[optimizer_idx][param_group_idx],
                dim=0,
            )
else:
    for i in extra_param_list_key_by_name:
        ckpt["gaussian_model_extra_state_dict"][i] = torch.concat(extra_param_list_key_by_name[i], dim=0)
    logger.info("Merging optimizers...")
    for i in optimizer_state_exp_avg_list_key_by_index.keys():
        ckpt["optimizer_states"][0]["state"][i]["exp_avg"] = torch.concat(optimizer_state_exp_avg_list_key_by_index[i], dim=0)
        ckpt["optimizer_states"][0]["state"][i]["exp_avg_sq"] = torch.concat(optimizer_state_exp_avg_sq_list_key_by_index[i], dim=0)


def rename_ddp_appearance_states():
    gaussian_property_dict_key_prefix = "renderer.appearance_model.module."
    for i in list(ckpt["state_dict"].keys()):
        if i.startswith(gaussian_property_dict_key_prefix) is False:
            continue
        new_key = "renderer.model.{}".format(i[len(gaussian_property_dict_key_prefix):])
        ckpt["state_dict"][new_key] = ckpt["state_dict"][i]
        del ckpt["state_dict"][i]

logger.info("number_of_gaussians=sum({})={}".format(number_of_gaussians, sum(number_of_gaussians)))
output_checkpoint_dir = os.path.join(os.path.dirname(checkpoint_dir), "checkpoints")
if not os.path.exists(output_checkpoint_dir):
    os.makedirs(output_checkpoint_dir)
output_path = os.path.join(output_checkpoint_dir, ckpt["hyper_parameters"]["initialize_from"].split('/')[-1])
logger.info(f"Saving to '{output_path}'")
torch.save(ckpt, output_path)
logger.info(f"Saved to '{output_path}'")