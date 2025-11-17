import os
import sys
import yaml
import torch
import torch_scatter
import imageio
import copy
import numpy as np
import re
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from internal.utils.general_utils import parse
from internal.utils.render_utils import generate_path, record_path, generate_static_path,generate_path_custom,generate_single_pose,pad_poses
from internal.utils.gaussian_model_loader import GaussianModelLoader

import threading  # 新增：导入线程模块
import socket
import cv2

# --------------------------
# 新增：图片发送线程函数
# --------------------------
def send_image_thread(image, port=12345):
    """主进程会等待该线程发送完成后再退出，直接接收内存中的图像数组"""
    def send_image():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                s.listen(1)
                print(f"\n发送线程启动：等待本地连接（端口 {port}）...")
                conn, addr = s.accept()
                with conn:
                    print(f"已连接本地笔记本：{addr}")
                    
                    # 注意：OpenCV 默认处理 BGR 格式，而渲染的图像是 RGB，需要转换
                    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # 编码为JPG（直接处理内存中的图像）
                    _, img_encoded = cv2.imencode('.jpg', img_bgr)
                    img_bytes = img_encoded.tobytes()
                    
                    # 先发送图片大小，再发送数据
                    conn.sendall(len(img_bytes).to_bytes(4, byteorder='big'))
                    conn.sendall(img_bytes)
                    print(f"图像已成功发送到本地（未保存到服务器）")
        except Exception as e:
            print(f"发送线程错误：{str(e)}")

    thread = threading.Thread(target=send_image)  # 非守护线程
    thread.start()
    thread.join()  # 主进程等待发送完成

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
    parser.add_argument("--custom", action="store_true", help="whether to use custom trajectory (all frames use first camera pose)")
    parser.add_argument("--axis", type=str, help="Select the axis to translate", default=1.0)
    parser.add_argument("--distance", type=float, help="control the distance of the trajectory", default=1.0)
    parser.add_argument("--x_scale", type=float, help="control the x axis position", default=1)
    parser.add_argument("--y_scale", type=float, help="control the y axis position", default=1)
    parser.add_argument("--z_scale", type=float, help="control the z axis position", default=1)
    parser.add_argument("--pitch_custom", type=float, help="control the camera pitch", default=0)
    parser.add_argument("--yaw_custom", type=float, help="control the z camera yaw", default=0)
    parser.add_argument("--roll_custom", type=float, help="control the z camera roll", default=0)

    # 在原ArgumentParser中添加以下参数
    parser.add_argument("--single_image", action="store_true", help="是否只渲染单张图片（而非视频）")
    parser.add_argument("--x_offset", type=float, default=0.0, help="单张图片的x轴位置偏移（相对于第一张相机）")
    parser.add_argument("--y_offset", type=float, default=0.0, help="单张图片的y轴位置偏移")
    parser.add_argument("--z_offset", type=float, default=0.0, help="单张图片的z轴位置偏移")
    parser.add_argument("--pitch_single", type=float, default=0.0, help="单张图片的pitch角度（度）")
    parser.add_argument("--yaw_single", type=float, default=0.0, help="单张图片的yaw角度（度）")
    parser.add_argument("--roll_single", type=float, default=0.0, help="单张图片的roll角度（度）")
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

    # 新增：单张图片渲染逻辑
    if args.single_image:
        # 1. 提取第一张相机的参考位姿
        first_cam_pose = np.linalg.inv(np.asarray((cameras[0].world_to_camera.T).cpu().numpy()))
        # 坐标系转换（与generate_path_custom保持一致）
        first_cam_pose = np.diag([1, -1, 1, 1]) @ first_cam_pose

        # 2. 生成调整后的单一位姿
        single_pose = generate_single_pose(
            reference_pose=first_cam_pose,
            x_offset=args.x_offset,
            y_offset=args.y_offset,
            z_offset=args.z_offset,
            yaw=args.yaw_single,
            pitch=args.pitch_single,
            roll=args.roll_single
        )
        # 补齐为4x4矩阵（适配后续转换）
        single_pose = pad_poses(np.array([single_pose]))[0]  # 形状(4,4)

        # 3. 构造相机对象
        cam = copy.deepcopy(cameras[0]).to_device("cuda")
        cam.height = int(cam.height / 2) * 2
        cam.width = int(cam.width / 2) * 2
        # 转换位姿为世界到相机矩阵
        c2w = single_pose @ np.diag([1, -1, -1, 1])  # 转回COLMAP坐标系
        cam.world_to_camera = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
        cam.full_projection = (cam.world_to_camera.unsqueeze(0).bmm(cam.projection.unsqueeze(0))).squeeze(0)
        cam.camera_center = cam.world_to_camera.inverse()[3, :3]

        # 4. 应用噪点过滤（如果启用）
        if args.filter:
            # （复用之前的过滤逻辑，与轨迹模式一致）
            xyz_homo = torch.cat((model.get_xyz, torch.zeros(model.get_xyz.shape[0], 1, device="cuda")), dim=-1)
            transformed_xyz = xyz_homo @ torch.tensor(colmap_to_world_transform, device="cuda", dtype=xyz_homo.dtype).T
            x_min, x_max = transformed_xyz[:, 0].min(), transformed_xyz[:, 0].max()
            y_min, y_max = transformed_xyz[:, 1].min(), transformed_xyz[:, 1].max()
            voxel_size = torch.tensor([(x_max - x_min) / args.vox_grid, (y_max - y_min) / args.vox_grid], device="cuda")
            xy_range = torch.tensor([x_min, y_min, x_max, y_max], device="cuda")
            vox_mask = voxel_filtering_no_gt(voxel_size, xy_range, transformed_xyz, args.std_ratio).bool().cpu().numpy()
            model.opacities[vox_mask] = 0.0

        # 5. 渲染并保存单张图片
        # os.makedirs('./render_images', exist_ok=True)
        # img_path = os.path.join('./render_images', f"{args.output_path.split('/')[-1]}_single.png")
        # img = renderer(cam, model, bkgd_color)['render']
        # img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        # imageio.imwrite(img_path, img)
        # print(f"Single image saved to {img_path}")

        # 5. 渲染图像（仅在内存中，不保存到服务器）
        img = renderer(cam, model, bkgd_color)['render']
        # 转换为 numpy 数组（RGB格式，0-255 uint8）
        img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        print(f"图像渲染完成，准备发送到本地...")

        # 直接发送内存中的图像数组（不保存到服务器硬盘）
        send_image_thread(
            image=img,  # 传入内存中的图像数组
            port=12345
        )

        # 无需等待额外时间（send_image_thread 已通过 join 确保发送完成）