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
import signal
from queue import Queue,Empty  # 新增：线程安全队列
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
# 全局变量（新增队列）
# --------------------------
offset = {'x': 0.0, 'y': 0.0, 'z': 0.0}
offset_lock = threading.Lock()
# 新增角度控制变量
yaw = 0.0         # 偏航角
pitch = 0.0       # 俯仰角
roll = 0.0        # 翻滚角
angle_lock = threading.Lock()  # 角度控制的线程锁
STEP = 0.1
STEP_ANGLE = 1.0  # 角度变化步长（10度）
KEYBOARD_THREAD_RUNNING = True
SEND_THREAD_RUNNING = True  # 发送线程运行标志
old_terminal_attr = None
IMAGE_QUEUE = Queue(maxsize=3)  # 图像缓冲队列（最大5张，避免内存堆积）

# --------------------------
# 终端模式切换工具函数
# --------------------------
def set_terminal_raw_mode():
    global old_terminal_attr
    fd = sys.stdin.fileno()
    old_terminal_attr = termios.tcgetattr(fd)
    new_attr = termios.tcgetattr(fd)
    new_attr[3] &= ~(termios.ICANON | termios.ECHO)
    termios.tcsetattr(fd, termios.TCSANOW, new_attr)
    fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
    return old_terminal_attr

def restore_terminal_mode():
    global old_terminal_attr
    if old_terminal_attr is not None:
        fd = sys.stdin.fileno()
        termios.tcsetattr(fd, termios.TCSANOW, old_terminal_attr)
        old_terminal_attr = None

# --------------------------
# 信号处理函数
# --------------------------
def handle_sigint(signum, frame):
    print("\n🛑 捕获到Ctrl+C，正在退出...")
    global KEYBOARD_THREAD_RUNNING, SEND_THREAD_RUNNING
    KEYBOARD_THREAD_RUNNING = False
    SEND_THREAD_RUNNING = False
    restore_terminal_mode()
    # 向队列放入None作为退出信号
    IMAGE_QUEUE.put(None)
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

# --------------------------
# 键盘监听线程（新增角度控制）
# --------------------------
def keyboard_listener_thread():
    # 更新提示信息，包含角度控制按键
    print("键盘监听线程已启动（按W/S/A/D/Q/E控制偏移；J/L/I/K/U/O控制角度；按Ctrl+C退出）")
    set_terminal_raw_mode()
    try:
        while KEYBOARD_THREAD_RUNNING:
            try:
                key = sys.stdin.read(1)
                if not key:
                    time.sleep(0.01)
                    continue
                key = key.lower()
                
                # 位置偏移控制（原有逻辑）
                with offset_lock:
                    if key == 'w':
                        offset['x'] += STEP
                        print(f"✅ X+ 偏移量更新：{offset['x']:.2f}")
                    elif key == 's':
                        offset['x'] -= STEP
                        print(f"✅ X- 偏移量更新：{offset['x']:.2f}")
                    elif key == 'a':
                        offset['y'] += STEP
                        print(f"✅ Y+ 偏移量更新：{offset['y']:.2f}")
                    elif key == 'd':
                        offset['y'] -= STEP
                        print(f"✅ Y- 偏移量更新：{offset['y']:.2f}")
                    elif key == 'q':
                        offset['z'] -= STEP
                        print(f"✅ Z- 偏移量更新：{offset['z']:.2f}")
                    elif key == 'e':
                        offset['z'] += STEP
                        print(f"✅ Z+ 偏移量更新：{offset['z']:.2f}")
                
                # 新增角度控制逻辑
                with angle_lock:
                    if key == 'j':  # 相机向左旋
                        global roll
                        roll -= STEP_ANGLE
                        print(f"🔄 Roll- 角度更新：{roll:.1f}°")
                    elif key == 'l':  # 相机向右旋
                        roll += STEP_ANGLE
                        print(f"🔄 Roll+ 角度更新：{roll:.1f}°")
                    elif key == 'i':  # 相机向上翻
                        global pitch
                        pitch += STEP_ANGLE
                        print(f"🔄 Pitch+ 角度更新：{pitch:.1f}°")
                    elif key == 'k':  # 相机向下翻
                        pitch -= STEP_ANGLE
                        print(f"🔄 Pitch- 角度更新：{pitch:.1f}°")
                    elif key == 'u':  # 相机向左翻
                        global yaw
                        yaw -= STEP_ANGLE
                        print(f"🔄 Yaw- 角度更新：{yaw:.1f}°")
                    elif key == 'o':  # 相机向右翻
                        yaw += STEP_ANGLE
                        print(f"🔄 Yaw+ 角度更新：{yaw:.1f}°")
                        
                        
            except Exception:
                continue
    finally:
        restore_terminal_mode()

# --------------------------
# 发送线程（修复阻塞退出）
# --------------------------
def send_thread(port=12345):
    """长期运行的发送线程，使用长连接持续发送图像（修复退出逻辑）"""
    print(f"发送线程已启动（端口 {port}），等待图像数据...")
    # 新增：保存当前连接的socket，用于退出时强制关闭
    current_socket = None
    current_conn = None

    while SEND_THREAD_RUNNING:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                current_socket = s  # 记录当前socket
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', port))
                s.settimeout(1.0)  # 缩短超时，加快退出响应
                s.listen(1)
                print("等待接收端连接...")
                
                try:
                    conn, addr = s.accept()
                    current_conn = conn  # 记录当前连接
                    print(f"已与接收端 {addr} 建立长连接")
                except socket.timeout:
                    continue  # 超时后重新循环，检查是否需要退出

                with conn:
                    while SEND_THREAD_RUNNING:  # 依赖SEND_THREAD_RUNNING退出
                        try:
                            image = IMAGE_QUEUE.get(timeout=0.05)
                            if image is None:
                                break
                            
                            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            _, img_encoded = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            img_bytes = img_encoded.tobytes()
                            conn.sendall(len(img_bytes).to_bytes(4, byteorder='big'))
                            conn.sendall(img_bytes)
                            IMAGE_QUEUE.task_done()
                        except Empty:
                            continue
                        except Exception as e:
                            print(f"长连接发送错误：{str(e)}，尝试重连...")
                            break
        except Exception as e:
            print(f"连接建立失败：{str(e)}，1秒后重试...")
            time.sleep(1)
            continue
        finally:
            # 退出前强制关闭连接，避免阻塞
            if current_conn:
                try:
                    current_conn.close()
                except:
                    pass
            if current_socket:
                try:
                    current_socket.close()
                except:
                    pass

    print("发送线程：已退出")

# --------------------------
# 其他工具函数（不变）
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
# 主程序（修改图像发送逻辑）
# --------------------------
if __name__ == "__main__":
    # 参数解析（不变）
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

    # 加载模型与数据（不变）
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
        
        # 2. 启动键盘线程和发送线程（发送线程长期运行）
        keyboard_thread = threading.Thread(target=keyboard_listener_thread, daemon=True)
        keyboard_thread.start()
        
        send_thread_obj = threading.Thread(target=send_thread, args=(12345,), daemon=True)
        send_thread_obj.start()

        try:
            while True:
                # 3. 获取偏移量和角度
                with offset_lock:
                    current_x = offset['x']
                    current_y = offset['y']
                    current_z = offset['z']
                with angle_lock:
                    current_yaw = yaw
                    current_pitch = pitch
                    current_roll = roll

                # 4. 生成位姿+构造相机（使用实时角度）
                single_pose = generate_single_pose(
                    reference_pose=first_cam_pose,
                    x_offset=current_x,
                    y_offset=current_y,
                    z_offset=current_z,
                    yaw=current_yaw,  # 使用实时yaw
                    pitch=current_pitch,  # 使用实时pitch
                    roll=current_roll  # 使用实时roll
                )
                single_pose = pad_poses(np.array([single_pose]))[0]

                #原代码
                cam = copy.deepcopy(cameras[0]).to_device("cuda")
                cam.height = int(cam.height / 2) * 2
                cam.width = int(cam.width / 2) * 2
                c2w = single_pose @ np.diag([1, -1, -1, 1])
                cam.world_to_camera = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
                cam.full_projection = (cam.world_to_camera.unsqueeze(0).bmm(cam.projection.unsqueeze(0))).squeeze(0)
                cam.camera_center = cam.world_to_camera.inverse()[3, :3]

                # 5. 渲染
                img = renderer(cam, model, bkgd_color)['render']
                img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()

                # 新增：缩小图像尺寸（关键优化）
                # 例如：缩小到原尺寸的50%（根据需求调整scale值）
                scale = 0.95  # 缩放比例（图片质量太大）
                height, width = img.shape[:2]
                new_height = int(height * scale)
                new_width = int(width * scale)
                # 使用INTER_AREA插值，缩小图像时质量更优
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)


                # 打印信息，包含角度
                frame_count = 0  # 在循环外定义
                frame_count += 1
                if frame_count % 25 == 0:  # 每25帧打印一次
                    print(f"[{time.ctime()}] 渲染完成（位置：X={current_x:.2f}, Y={current_y:.2f}, Z={current_z:.2f}；角度：Yaw={current_yaw:.1f}°, Pitch={current_pitch:.1f}°, Roll={current_roll:.1f}°）")

                # 6. 将图像放入队列（而非创建新线程）
                if not IMAGE_QUEUE.full():
                    IMAGE_QUEUE.put(img)  # 队列未满则放入
                    # print("图像已加入发送队列")
                else:
                    print("⚠️  发送队列已满，丢弃当前图像")

                # 关键：缩短延迟至0.02秒（理论支持50fps，预留渲染时间）
                # 若渲染耗时较长，可去掉固定延迟，让循环自然由渲染时间控制
                #time.sleep(0.02)

        finally:
            # 1. 修正：将发送线程运行标志设为False
            KEYBOARD_THREAD_RUNNING = False
            SEND_THREAD_RUNNING = False  # 关键修复：改为False
            # 2. 向队列放入退出信号
            IMAGE_QUEUE.put(None)
            # 3. 等待键盘线程和发送线程完全退出
            keyboard_thread.join(timeout=2.0)  # 限时等待
            send_thread_obj.join(timeout=2.0)   # 新增：等待发送线程
            # 4. 恢复终端
            restore_terminal_mode()
            print("所有线程已退出，程序终止")