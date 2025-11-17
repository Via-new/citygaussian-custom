import os
import sys
import termios
import fcntl
import yaml
import torch
import torch_scatter
import imageio
import copy
import numpy as np
import re
import time
import signal  # 新增：用于信号处理
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from internal.utils.general_utils import parse
from internal.utils.render_utils import generate_path, record_path, generate_static_path, generate_path_custom, generate_single_pose, pad_poses
from internal.utils.gaussian_model_loader import GaussianModelLoader

import threading
import socket
import cv2

# --------------------------
# 全局变量
# --------------------------
offset = {'x': 0.0, 'y': 0.0, 'z': 0.0}
offset_lock = threading.Lock()
STEP = 10
KEYBOARD_THREAD_RUNNING = True
old_terminal_attr = None  # 全局保存终端原始属性，用于强制恢复

# --------------------------
# 终端模式切换工具函数（修复信号处理）
# --------------------------
def set_terminal_raw_mode():
    """设置终端为原始模式（保留信号处理，支持Ctrl+C）"""
    global old_terminal_attr
    fd = sys.stdin.fileno()
    old_terminal_attr = termios.tcgetattr(fd)  # 保存原始属性
    new_attr = termios.tcgetattr(fd)
    
    # 关键修改：只禁用行缓冲和回显，保留信号处理（不移除termios.ISIG）
    new_attr[3] &= ~(termios.ICANON | termios.ECHO)  # 仅屏蔽这两个标志
    termios.tcsetattr(fd, termios.TCSANOW, new_attr)
    
    # 设置非阻塞模式
    fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
    return old_terminal_attr

def restore_terminal_mode():
    """强制恢复终端原始属性（全局调用）"""
    global old_terminal_attr
    if old_terminal_attr is not None:
        fd = sys.stdin.fileno()
        termios.tcsetattr(fd, termios.TCSANOW, old_terminal_attr)
        old_terminal_attr = None  # 恢复后清空

# --------------------------
# 信号处理函数（捕获Ctrl+C）
# --------------------------
def handle_sigint(signum, frame):
    """捕获SIGINT信号（Ctrl+C），确保终端恢复并退出"""
    print("\n🛑 捕获到Ctrl+C，正在退出...")
    global KEYBOARD_THREAD_RUNNING
    KEYBOARD_THREAD_RUNNING = False
    restore_terminal_mode()  # 强制恢复终端
    sys.exit(0)  # 退出程序

# 注册信号处理（确保Ctrl+C被捕获）
signal.signal(signal.SIGINT, handle_sigint)

# --------------------------
# 键盘监听线程（优化退出逻辑）
# --------------------------
def keyboard_listener_thread():
    """实时捕获单键输入，支持Ctrl+C退出"""
    print("键盘监听线程已启动（按W/S/A/D/Q/E控制偏移，无需回车；按Ctrl+C退出）")
    set_terminal_raw_mode()  # 设置终端模式
    try:
        while KEYBOARD_THREAD_RUNNING:
            try:
                key = sys.stdin.read(1)  # 读取单键（非阻塞）
                if not key:
                    time.sleep(0.01)
                    continue
                
                key = key.lower()
                with offset_lock:
                    if key == 'w':
                        offset['x'] += STEP
                        print(f"✅ X+ 偏移量更新：{offset['x']:.2f}")
                    elif key == 's':
                        offset['x'] -= STEP
                        print(f"✅ X- 偏移量更新：{offset['x']:.2f}")
                    elif key == 'a':
                        offset['y'] -= STEP
                        print(f"✅ Y- 偏移量更新：{offset['y']:.2f}")
                    elif key == 'd':
                        offset['y'] += STEP
                        print(f"✅ Y+ 偏移量更新：{offset['y']:.2f}")
                    elif key == 'q':
                        offset['z'] -= STEP
                        print(f"✅ Z- 偏移量更新：{offset['z']:.2f}")
                    elif key == 'e':
                        offset['z'] += STEP
                        print(f"✅ Z+ 偏移量更新：{offset['z']:.2f}")
            except Exception:
                continue
    finally:
        restore_terminal_mode()  # 线程退出时恢复终端

# --------------------------
# 图像发送线程（保持不变）
# --------------------------
def send_image_thread(image, port=12345):
    def send_image():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', port))
                s.settimeout(1.0)
                s.listen(1)
                try:
                    conn, addr = s.accept()
                except socket.timeout:
                    return
                with conn:
                    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    _, img_encoded = cv2.imencode('.jpg', img_bgr)
                    img_bytes = img_encoded.tobytes()
                    conn.sendall(len(img_bytes).to_bytes(4, byteorder='big'))
                    conn.sendall(img_bytes)
        except Exception as e:
            print(f"发送线程错误：{str(e)}")

    thread = threading.Thread(target=send_image)
    thread.start()
    thread.join(timeout=2.0)

# --------------------------
# 其他工具函数（保持不变）
# --------------------------
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
    if not os.path.exists(base_path):
        return base_path
    dir_name = os.path.dirname(base_path)
    base_name = os.path.basename(base_path)
    name_without_ext, ext = os.path.splitext(base_name)
    pattern = re.compile(rf'^{re.escape(name_without_ext)}_(\d+){re.escape(ext)}$')
    existing_numbers = []
    if os.path.exists(dir_name):
        for filename in os.listdir(dir_name):
            match = pattern.match(filename)
            if match:
                existing_numbers.append(int(match.group(1)))
    next_number = max(existing_numbers) + 1 if existing_numbers else 1
    new_filename = f"{name_without_ext}_{next_number:03d}{ext}"
    return os.path.join(dir_name, new_filename)

# --------------------------
# 主程序
# --------------------------
if __name__ == "__main__":
    # 参数解析
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--output_path', type=str, help='path of config', default=None)
    parser.add_argument('--data_path', type=str, help='path of data', default=None)
    parser.add_argument("--n_frames", type=int, default=240)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--scale_percentile", type=int, default=99)
    parser.add_argument("--pitch", type=float, default=None)
    parser.add_argument("--x_shift", type=float, default=0)
    parser.add_argument("--y_shift", type=float, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--vox_grid", type=int, default=25)
    parser.add_argument("--std_ratio", type=float, default=2.0)
    parser.add_argument("--save_filtered_gs", action="store_true")
    parser.add_argument("--custom", action="store_true")
    parser.add_argument("--single_image", action="store_true")
    parser.add_argument("--x_offset", type=float, default=0.0)
    parser.add_argument("--y_offset", type=float, default=0.0)
    parser.add_argument("--z_offset", type=float, default=0.0)
    parser.add_argument("--pitch_single", type=float, default=0.0)
    parser.add_argument("--yaw_single", type=float, default=0.0)
    parser.add_argument("--roll_single", type=float, default=0.0)
    args = parser.parse_args(sys.argv[1:])

    # 加载模型与数据
    loadable_file = GaussianModelLoader.search_load_file(args.output_path)
    ckpt = torch.load(loadable_file, map_location="cpu")
    dataparser_config = ckpt["datamodule_hyper_parameters"]["parser"]
    data_path = args.data_path if args.data_path else ckpt["datamodule_hyper_parameters"]["path"]
    dataparser_outputs = dataparser_config.instantiate(
        path=data_path, output_path=os.getcwd(), global_rank=0
    ).get_outputs()
    cameras = dataparser_outputs.train_set.cameras if args.train else dataparser_outputs.test_set.cameras
    device = torch.device("cuda")
    bkgd_color = torch.tensor(ckpt["hyper_parameters"]["background_color"], device=device)
    model = GaussianModelLoader.initialize_model_from_checkpoint(ckpt, device=device)
    model.freeze()
    model.pre_activate_all_properties()
    renderer = GaussianModelLoader.initialize_renderer_from_checkpoint(ckpt, stage="validate", device=device)
    print(f"Gaussian count: {model.get_xyz.shape[0]}")
    traj_dir = os.path.join(ckpt["datamodule_hyper_parameters"]["path"], 'traj')
    os.makedirs(traj_dir, exist_ok=True)

    # 循环渲染+键盘控制逻辑
    if args.single_image:
        # 1. 计算基准位姿
        first_cam_pose = np.linalg.inv(np.asarray((cameras[0].world_to_camera.T).cpu().numpy()))
        first_cam_pose = np.diag([1, -1, 1, 1]) @ first_cam_pose
        print("\n📌 已记录基准位姿，开始循环渲染（按Ctrl+C终止）")
        
        # 2. 启动键盘线程
        keyboard_thread = threading.Thread(target=keyboard_listener_thread, daemon=True)
        keyboard_thread.start()

        try:
            while True:
                # 3. 获取偏移量
                with offset_lock:
                    current_x = offset['x']
                    current_y = offset['y']
                    current_z = offset['z']

                # 4. 生成位姿
                single_pose = generate_single_pose(
                    reference_pose=first_cam_pose,
                    x_offset=current_x,
                    y_offset=current_y,
                    z_offset=current_z,
                    yaw=args.yaw_single,
                    pitch=args.pitch_single,
                    roll=args.roll_single
                )
                single_pose = pad_poses(np.array([single_pose]))[0]

                # 5. 构造相机
                cam = copy.deepcopy(cameras[0]).to_device("cuda")
                cam.height = int(cam.height / 2) * 2
                cam.width = int(cam.width / 2) * 2
                c2w = single_pose @ np.diag([1, -1, -1, 1])
                cam.world_to_camera = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
                cam.full_projection = (cam.world_to_camera.unsqueeze(0).bmm(cam.projection.unsqueeze(0))).squeeze(0)
                cam.camera_center = cam.world_to_camera.inverse()[3, :3]

                # 6. 渲染
                img = renderer(cam, model, bkgd_color)['render']
                img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                print(f"\n[{time.ctime()}] 渲染完成（当前偏移：X={current_x:.2f}, Y={current_y:.2f}, Z={current_z:.2f}）")

                # 7. 发送图像
                send_image_thread(image=img, port=12345)

                time.sleep(0.1)

        finally:
            # 无论如何退出，都强制恢复终端
            KEYBOARD_THREAD_RUNNING = False
            keyboard_thread.join()
            restore_terminal_mode()