import os
import sys
import yaml
import torch
import torch_scatter
import imageio
import numpy as np
import re
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from internal.utils.general_utils import parse
from internal.utils.render_utils import generate_path, record_path, generate_static_path
from internal.utils.gaussian_model_loader import GaussianModelLoader

# 体素过滤函数（去除漂浮噪点）  通过将场景在 xy 平面划分为体素，统计每个体素内 z 坐标（高度）的分布，过滤掉高度异常的点（可能是重建错误的漂浮噪点）。
def voxel_filtering_no_gt(voxel_size, xy_range, target_xyz, std_ratio=2.0):
    assert len(xy_range) == 4, "Unrecognized xy_range format"
    with torch.no_grad():

        voxel_index = torch.div(torch.tensor(target_xyz[:, :2]).float() - xy_range[None, :2], voxel_size[None, :], rounding_mode='floor')
        voxel_coords = voxel_index * voxel_size[None, :] + xy_range[None, :2] + voxel_size[None, :] / 2

        new_coors, unq_inv, unq_cnt = torch.unique(voxel_coords, return_inverse=True, return_counts=True, dim=0)
        feat_mean = torch_scatter.scatter(target_xyz[:, 2], unq_inv, dim=0, reduce='mean')
        feat_std = torch_scatter.scatter_std(target_xyz[:, 2], unq_inv, dim=0)

        mask = target_xyz[:, 2] > feat_mean[unq_inv] + std_ratio * feat_std[unq_inv]

    return mask

def get_next_video_path(base_path):
    """
    获取下一个可用的视频文件路径（带序号，避免覆盖已有文件）
    
    Args:
        base_path: 基础文件路径（包含完整路径和扩展名，如 './videos/model_video.mp4'）
    
    Returns:
        新的文件路径，如果基础路径不存在则返回基础路径，否则返回带序号的新路径
    """
    # 如果基础路径不存在，直接返回
    if not os.path.exists(base_path):
        return base_path
    
    # 分离文件名和扩展名
    dir_name = os.path.dirname(base_path)  # 目录路径
    base_name = os.path.basename(base_path)  # 文件名（包含扩展名）
    name_without_ext, ext = os.path.splitext(base_name)  # 分离文件名和扩展名
    
    # 查找已存在的带序号的文件
    # 匹配模式：{name_without_ext}_XXX{ext}，其中XXX是数字（可能不是3位）
    pattern = re.compile(rf'^{re.escape(name_without_ext)}_(\d+){re.escape(ext)}$')
    
    existing_numbers = []
    if os.path.exists(dir_name):
        for filename in os.listdir(dir_name):
            match = pattern.match(filename)
            if match:
                existing_numbers.append(int(match.group(1)))
    
    # 找到下一个可用的序号
    # 如果基础文件存在，从001开始；如果已有带序号的文件，使用最大序号+1
    if existing_numbers:
        next_number = max(existing_numbers) + 1
    else:
        # 基础文件存在但没有带序号的文件，从001开始
        next_number = 1
    
    # 生成新的文件路径（序号格式化为3位数字，如 001, 002, ...）
    new_filename = f"{name_without_ext}_{next_number:03d}{ext}"
    new_path = os.path.join(dir_name, new_filename)
    
    return new_path

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--output_path', type=str, help='path of config', default=None)
    parser.add_argument('--data_path', type=str, help='path of data', default=None)
    parser.add_argument("--n_frames", type=int, help="number of frames", default=240)
    parser.add_argument("--train", action="store_true", help="whether to use train set as trajectories")
    parser.add_argument("--scale_percentile", type=int, help="trajectory radius percentile", default=99)
    parser.add_argument("--pitch", type=float, help="pitch in degree, 0 means no pitch changes", default=None)
    parser.add_argument("--x_shift", type=float, help="x-axis shift of ellipse center, 0 means no pitch changes", default=0)
    parser.add_argument("--y_shift", type=float, help="y-axis shift of ellipse center, 0 means no pitch changes", default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--filter", action="store_true", help="whether to filter out floaters")
    parser.add_argument("--vox_grid", type=int, help="number of voxelization grid", default=25)
    parser.add_argument("--std_ratio", type=float, help="used to control filtering threshold", default=2.0)
    parser.add_argument("--save_filtered_gs", action="store_true", help="whether to save filtered gaussian")
    parser.add_argument("--static", action="store_true", help="whether to use static trajectory (all frames use first camera pose)")
    args = parser.parse_args(sys.argv[1:])

    # 加载模型与数据
    loadable_file = GaussianModelLoader.search_load_file(args.output_path)
    ckpt = torch.load(loadable_file, map_location="cpu")

    # 解析数据配置（相机参数、数据路径等）
    dataparser_config = ckpt["datamodule_hyper_parameters"]["parser"]
    if args.data_path is not None:
        data_path = args.data_path
    else:
        data_path = ckpt["datamodule_hyper_parameters"]["path"]
    dataparser_outputs = dataparser_config.instantiate(
        path=data_path,
        output_path=os.getcwd(),
        global_rank=0,
    ).get_outputs()

    # 选择使用训练集或测试集的相机来生成轨迹
    if args.train:
        cameras = dataparser_outputs.train_set.cameras
    else:
        cameras = dataparser_outputs.test_set.cameras

    device = torch.device("cuda")  # 使用GPU渲染
    bkgd_color = torch.tensor(ckpt["hyper_parameters"]     ["background_color"], device=device)  # 背景颜色
    
    # 从检查点初始化高斯模型并冻结参数（仅用于渲染，不训练）
    model = GaussianModelLoader.initialize_model_from_checkpoint(
        ckpt,
        device=device,
    )
    model.freeze()
    model.pre_activate_all_properties()
    # 初始化渲染器（高斯渲染器，用于将3D高斯点投影为2D图像）
    renderer = GaussianModelLoader.initialize_renderer_from_checkpoint(
        ckpt,
        stage="validate",
        device=device,
    )
    print("Gaussian count: {}".format(model.get_xyz.shape[0])) # 打印高斯点数量

    traj_dir = os.path.join(ckpt["datamodule_hyper_parameters"]["path"], 'traj')  # 创建轨迹保存目录
    os.makedirs(traj_dir, exist_ok=True)

    # 根据参数选择生成静态轨迹还是椭圆轨迹
    if args.static:
        # 生成静态轨迹：所有帧都使用第一张相机的位姿
        if not args.filter:
            cam_traj = generate_static_path(cameras, traj_dir, n_frames=args.n_frames)
        else:
            # 静态轨迹不支持过滤（因为不需要坐标变换）
            print("Warning: --filter is not supported with --static. Generating static trajectory without filter.")
            cam_traj = generate_static_path(cameras, traj_dir, n_frames=args.n_frames, filter=False)
    else:
        # 生成椭圆轨迹（原有逻辑）
        if not args.filter:
            # 不过滤噪点：直接生成相机轨迹（基于输入相机的分布，生成平滑的椭圆或路径）
            cam_traj = generate_path(cameras, traj_dir, n_frames=args.n_frames, pitch=args.pitch, shift=[args.x_shift, args.y_shift], scale_percentile=args.scale_percentile)
        else:
            # 过滤噪点：生成轨迹时同时获取坐标变换矩阵，用于对齐高斯点和轨迹
            cam_traj, colmap_to_world_transform, pose_recenter = generate_path(cameras, traj_dir, n_frames=args.n_frames, pitch=args.pitch, 
                                                                               shift=[args.x_shift, args.y_shift], filter=True, 
                                                                               scale_percentile=args.scale_percentile)
            # 将高斯点坐标转换到轨迹对应的坐标系（便于体素过滤）
            xyz_homo = torch.cat((model.get_xyz, torch.zeros(model.get_xyz.shape[0], 1, device="cuda")), dim=-1)
            transformed_xyz = xyz_homo @ torch.tensor(colmap_to_world_transform, device="cuda", dtype=xyz_homo.dtype).T

            # 计算xy范围，确定体素大小
            x_min, x_max = transformed_xyz[:, 0].min(), transformed_xyz[:, 0].max()
            y_min, y_max = transformed_xyz[:, 1].min(), transformed_xyz[:, 1].max()
            voxel_size = torch.tensor([(x_max - x_min) / args.vox_grid, (y_max - y_min) / args.vox_grid], device="cuda")
            xy_range = torch.tensor([x_min, y_min, x_max, y_max], device="cuda")

            # 执行体素过滤，得到需要去除的点的掩码
            vox_mask = voxel_filtering_no_gt(voxel_size, xy_range, transformed_xyz, args.std_ratio).bool().cpu().numpy()
            model.opacities[vox_mask] = 0.0

            # 保存过滤后的模型（可选）
            if args.save_filtered_gs:
                ckpt['state_dict']['gaussian_model.gaussians.opacities'][vox_mask] = -13.8  #1e-6
                torch.save(ckpt, loadable_file.replace('.ckpt', '_filtered.ckpt'))
   
    # 渲染视频并保存
    print(f"Camera trajectory saved to {traj_dir}. Start rendering...")

    # 创建视频保存目录
    os.makedirs('./videos', exist_ok=True)
    # 生成基础视频路径
    base_video_path = os.path.join('./videos', f"{args.output_path.split('/')[-1]}_video.mp4")
    # 获取下一个可用的视频路径（带序号，避免覆盖已有文件）
    video_path = get_next_video_path(base_video_path)
    print(f"Video will be saved to: {video_path}")
    ##1109  # 初始化视频写入器（30帧/秒）
    video = imageio.get_writer(video_path,format='ffmpeg', fps=60) 
    #video = imageio.get_writer(video_path, fps=30)


    # 遍历轨迹中的每个相机，渲染图像并添加到视频
    for t in tqdm(range(len(cam_traj))):
        cam = cam_traj[t]  # 获取第t帧的相机参数
        cam.height = torch.tensor(cam.height, device=cam.R.device)
        cam.width = torch.tensor(cam.width, device=cam.R.device)
        img = renderer(cam, model, bkgd_color)['render']  # 调用渲染器生成图像
        # 转换图像格式（0-255 uint8）并添加到视频
        img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        video.append_data(img)
    video.close()  # 关闭视频写入器
    print(f"Video saved to {video_path}.")