# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os  # 操作系统接口，用于文件路径操作
import enum  # 枚举类型支持
import types  # 类型检查相关
import copy  # 深拷贝功能，用于复制相机对象
import pickle  # 序列化/反序列化，用于保存相机轨迹
import numpy as np  # 数值计算库
import mediapy as media  # 视频处理库
from typing import List, Mapping, Optional, Text, Tuple, Union  # 类型提示
from PIL import Image  # 图像处理库
from matplotlib import cm  # 颜色映射，用于深度图可视化
from tqdm import tqdm  # 进度条显示
from typing import Literal

import torch  # PyTorch深度学习框架

def normalize(x: np.ndarray) -> np.ndarray:
  """归一化辅助函数：将向量归一化为单位向量"""
  return x / np.linalg.norm(x)  # 除以向量的L2范数（模长），得到单位向量

def pad_poses(p: np.ndarray) -> np.ndarray:
  """将 [..., 3, 4] 位姿矩阵填充为齐次坐标形式 [..., 4, 4]，底部添加 [0,0,0,1] 行"""
  # 创建底部行 [0, 0, 0, 1]，并广播到与输入矩阵匹配的形状
  bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
  # 在倒数第二个维度（行维度）上拼接，将3x4矩阵变为4x4齐次矩阵
  return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
  """从 [..., 4, 4] 齐次位姿矩阵中移除底部齐次行，返回 [..., 3, 4] 矩阵"""
  return p[..., :3, :4]  # 只取前3行，移除最后一行 [0,0,0,1]


def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """将位姿重新居中到原点：计算平均位姿，然后将所有位姿变换到以平均位姿为原点的坐标系"""
  cam2world = average_pose(poses)  # 计算所有位姿的平均位姿
  transform = np.linalg.inv(pad_poses(cam2world))  # 计算逆变换矩阵，用于将位姿移到原点
  poses = transform @ pad_poses(poses)  # 应用变换，将所有位姿相对于平均位姿重新居中
  return unpad_poses(poses), transform  # 返回变换后的位姿（移除齐次行）和变换矩阵


def average_pose(poses: np.ndarray) -> np.ndarray:
  """使用输入位姿的平均位置、z轴和上向量创建新的平均位姿"""
  position = poses[:, :3, 3].mean(0)  # 计算所有位姿位置的平均值（第4列是位置）
  z_axis = poses[:, :3, 2].mean(0)  # 计算所有位姿z轴（前向方向）的平均值（第3列）
  up = poses[:, :3, 1].mean(0)  # 计算所有位姿上向量（上方向）的平均值（第2列）
  cam2world = viewmatrix(z_axis, up, position)  # 使用平均的方向和位置构造视图矩阵
  return cam2world  # 返回相机到世界的变换矩阵

def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
  """构造lookat视图矩阵（相机到世界的变换矩阵）"""
  vec2 = normalize(lookdir)  # 归一化观察方向（z轴，相机前向）
  vec0 = normalize(np.cross(up, vec2))  # 右向量（x轴）：上向量与观察方向的叉积
  vec1 = normalize(np.cross(vec2, vec0))  # 上向量（y轴）：观察方向与右向量的叉积，确保正交
  m = np.stack([vec0, vec1, vec2, position], axis=1)  # 将三个基向量和位置堆叠成4x4矩阵
  return m  # 返回相机到世界的变换矩阵

def focus_point_fn(poses: np.ndarray) -> np.ndarray:
  """计算所有位姿焦点轴的最邻近点（所有相机视线的焦点）"""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]  # 提取所有相机的观察方向（z轴）和位置
  # 计算投影矩阵：将向量投影到垂直于观察方向的平面上
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m  # 计算投影矩阵的转置乘以自身
  # 使用最小二乘法找到最接近所有焦点轴的点
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt  # 返回焦点坐标

def transform_poses_pca(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """使用PCA（主成分分析）变换位姿，使主成分位于XYZ轴上

  Args:
    poses: 形状为 (N, 3, 4) 的数组，包含相机的相机到世界变换矩阵

  Returns:
    元组 (poses, transform)，包含变换后的位姿和应用的相机到世界变换矩阵
  """
  t = poses[:, :3, 3]  # 提取所有位姿的位置向量（第4列）
  t_mean = t.mean(axis=0)  # 计算位置的平均值
  t = t - t_mean  # 将位置中心化（减去均值）

  # 对中心化后的位置进行PCA：计算协方差矩阵的特征值和特征向量
  eigval, eigvec = np.linalg.eig(t.T @ t)
  # 按特征值从大到小排序特征向量
  inds = np.argsort(eigval)[::-1]
  eigvec = eigvec[:, inds]
  rot = eigvec.T  # 转置得到旋转矩阵（主成分作为新坐标轴）
  # 如果行列式为负，需要翻转一个轴以保持右手坐标系
  if np.linalg.det(rot) < 0:
    rot = np.diag(np.array([1, 1, -1])) @ rot

  # 构造完整的变换矩阵：旋转 + 平移（先旋转再平移）
  transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
  # 应用变换到所有位姿
  poses_recentered = unpad_poses(transform @ pad_poses(poses))
  # 将变换矩阵扩展为4x4齐次矩阵
  transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

  # 如果y轴的z分量为负，翻转坐标系（确保y轴向上）
  if poses_recentered.mean(axis=0)[2, 1] < 0:
    poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
    transform = np.diag(np.array([1, -1, -1, 1])) @ transform

  return poses_recentered, transform  # 返回变换后的位姿和变换矩阵
  # points = np.random.rand(3,100)
  # points_h = np.concatenate((points,np.ones_like(points[:1])), axis=0)
  # (poses_recentered @ points_h)[0]
  # (transform @ pad_poses(poses) @ points_h)[0,:3]
  # import pdb; pdb.set_trace()

  # # Just make sure it's it in the [-1, 1]^3 cube
  # scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
  # poses_recentered[:, :3, 3] *= scale_factor
  # transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

  # return poses_recentered, transform

def generate_ellipse_path(poses: np.ndarray,
                          n_frames: int = 120,
                          scale_percentile: int = 90,
                          shift: float = None,
                          pitch: float = None,
                          z_variation: float = 0.,
                          z_phase: float = 0.) -> np.ndarray:
  """基于给定位姿生成椭圆渲染路径"""
  # 计算路径的焦点（所有相机朝向这个点）
  center = focus_point_fn(poses)
  # 路径高度位于z=0（在零均值捕获模式的中间）
  offset = np.array([center[0]+shift[0], center[1]+shift[1], 0])  # 应用xy偏移，z设为0

  # 基于输入相机位置计算椭圆轴的缩放
  sc = np.percentile(np.abs(poses[:, :3, 3] - offset), scale_percentile, axis=0)
  # 使用关于焦点在xy平面上对称的椭圆
  low = -sc + offset  # 椭圆的下界（最小x, y）
  high = sc + offset  # 椭圆的上界（最大x, y）
  # 可选的高度变化不需要对称
  z_low = np.percentile((poses[:, :3, 3]), 100 - scale_percentile, axis=0)  # z轴下界
  z_high = np.percentile((poses[:, :3, 3]), scale_percentile, axis=0)  # z轴上界

  def get_positions(theta):
    """根据角度theta计算椭圆上的位置"""
    # 使用三角函数在边界之间插值，在x-y平面上得到椭圆
    # 可选地也在z上插值以沿路径改变相机高度
    return np.stack([
        low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),  # x坐标：cos函数映射到[0,1]再缩放到[low, high]
        low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),  # y坐标：sin函数映射到[0,1]再缩放到[low, high]
        z_variation * (z_low[2] + (z_high - z_low)[2] *  # z坐标：可选的高度变化
                       (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),  # 带相位偏移的cos函数
    ], -1)

  theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)  # 生成从0到2π的均匀角度
  positions = get_positions(theta)  # 计算所有角度对应的位置

  #if const_speed:  # 如果启用恒定速度（已注释）

  # # 重新采样theta角度，使速度更接近恒定
  # lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
  # theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
  # positions = get_positions(theta)

  # 丢弃重复的最后一个位置（因为endpoint=True会产生重复的0和2π位置）
  positions = positions[:-1]

  # 设置路径的上向量为最接近输入位姿平均上向量的轴
  avg_up = poses[:, :3, 1].mean(0)  # 计算所有位姿上向量的平均值
  avg_up = avg_up / np.linalg.norm(avg_up)  # 归一化
  ind_up = np.argmax(np.abs(avg_up))  # 找到绝对值最大的分量索引（确定主要方向）
  up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])  # 创建标准基向量，保持符号

  lookdir = positions - center[None, :]  # 计算从每个位置到焦点的观察方向
  if pitch is not None:
    pitch = np.deg2rad(pitch)  # 将俯仰角从度转换为弧度
    # 根据ind_up和pitch构造旋转矩阵：调整观察方向以应用俯仰角
    lookdir[:, ind_up] = np.linalg.norm(lookdir[:, ind_up-2:ind_up], axis=-1) * np.tan(pitch)

  # 为每个位置构造视图矩阵（相机到世界的变换矩阵）
  return np.stack([viewmatrix(lookdir[i], up, positions[i]) for i in range(positions.shape[0])])


def generate_translation_path(reference_pose: np.ndarray,
                              n_frames: int = 120,
                              axis: Literal['x', 'y','z'] = 'x',
                              distance: float = 1.0,
                              x_scale:float = 1.0,
                              y_scale:float = 1.0,
                              z_scale:float = 1.0,
                              yaw: float = 0.0,    # Yaw调整（度），正值向右倾，负值向左倾
                              pitch: float = 0.0,  # Pitch调整（度），正值抬头，负值低头
                              roll: float = 0.0) -> np.ndarray:  # Roll调整（度），正值右旋，负值左旋
    """
    基于单张参考相机位姿，生成沿x/y轴平移的轨迹，并支持Yaw/Pitch/Roll角度调整
    
    参数：
        reference_pose: 参考相机位姿（3x4矩阵，前3列旋转，第4列位置）
        n_frames: 轨迹总帧数
        distance: 平移总距离（单向）
        yaw_adjust: Yaw角调整量（度），绕Y轴旋转
        pitch_adjust: Pitch角调整量（度），绕X轴旋转
        roll_adjust: Roll角调整量（度），绕Z轴旋转
    返回：
        生成的轨迹位姿数组（[n_frames, 3, 4]）
    """
    # --------------------------
    # 1. 提取参考相机的原始旋转和位置
    # --------------------------
    R_original = reference_pose[:3, :3].copy()  # 原旋转矩阵（3x3）
    original_pos = reference_pose[:3, 3].copy()  # 原位置（x, y, z）
    
    # --------------------------
    # 2. 构造三个旋转的矩阵（按Yaw→Pitch→Roll顺序叠加）
    # --------------------------
    # 角度转弧度
    yaw_rad = np.deg2rad(yaw)
    pitch_rad = np.deg2rad(pitch)
    roll_rad = np.deg2rad(roll)
    
    # 1) Yaw旋转矩阵（绕Y轴：左右转向）
    R_yaw = np.array([
        [np.cos(yaw_rad), 0, -np.sin(yaw_rad)],  # sin项取负
        [0, 1, 0],
        [np.sin(yaw_rad), 0, np.cos(yaw_rad)]   # sin项取负
    ])
    
    # 2) Pitch旋转矩阵（绕X轴：上下俯仰）
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    # 3) Roll旋转矩阵（绕Z轴：左右倾斜）
    R_roll = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ])
    
    # 组合旋转矩阵：先Yaw→再Pitch→最后Roll（顺序影响最终姿态）
    R_rot = R_roll @ R_pitch @ R_yaw  # 旋转复合矩阵
    # 与原旋转矩阵结合（先原姿态，再叠加新旋转）
    R_new = R_original @ R_rot  # 最终旋转矩阵
    
    # --------------------------
    # 3. 生成平移轨迹位置（与之前一致）
    # --------------------------
    axis_idx = 0 
    if axis == 'x': 
      axis_idx = 0
    elif axis == 'y':
      axis_idx = 1
    else: axis_idx = 2
    offsets = np.linspace(0, distance, n_frames)  # 单向平移
    positions = []
    for offset in offsets:
        new_pos = original_pos.copy()
        new_pos[0]*=x_scale
        new_pos[1]*=y_scale
        new_pos[2]*=z_scale
        new_pos[axis_idx] += offset  # 只修改x或y轴
        positions.append(new_pos)
    positions = np.array(positions)  # [n_frames, 3]
    
    # --------------------------
    # 4. 提取调整后的相机朝向参数
    # --------------------------
    # 新的上向量（旋转矩阵第2列）
    up = -R_new[:, 1].copy()
    up = up / np.linalg.norm(up)  # 归一化
    # 新的观察方向（旋转矩阵第3列的反方向）
    look_dir = -R_new[:, 2].copy()
    look_dir = look_dir / np.linalg.norm(look_dir)
    
    # --------------------------
    # 5. 构造最终位姿矩阵
    # --------------------------
    trajectory_poses = np.stack([
        viewmatrix(look_dir, up, pos)  # 每个位置的位姿：新旋转+新位置
        for pos in positions
    ])
    
    return trajectory_poses

def generate_path_custom(viewpoint_cameras, traj_dir=None, n_frames=480, scale_percentile=90, shift=[0, 0], filter=False,distance=1.0,pitch_custom=0,roll=0,yaw=0,x_scale=1,y_scale=1,z_scale=1,axis='x'):
  """生成相机轨迹路径"""
  # 从所有输入相机中提取相机到世界的变换矩阵（c2w）
  c2ws = np.array([np.linalg.inv(np.asarray((cam.world_to_camera.T).cpu().numpy())) for cam in viewpoint_cameras])
  # 应用坐标系转换：将COLMAP坐标系转换为标准坐标系（翻转y和z轴）
  pose = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
  # 使用PCA变换位姿，使主成分对齐到XYZ轴，并获取变换矩阵
  pose_recenter, colmap_to_world_transform = transform_poses_pca(pose)
  #使用第一张图片的位姿作为参考位姿
  first_cam_pose = np.linalg.inv(np.asarray((viewpoint_cameras[0].world_to_camera.T).cpu().numpy()))  # 3x4矩阵（包含位置）

  # 左乘 diag([1,-1,-1,1]) 表示“将COLMAP坐标系的位姿映射到标准坐标系”
  first_cam_pose = np.diag([1, -1, 1, 1]) @ first_cam_pose  # 左乘！左乘！左乘！

  new_poses = generate_translation_path(
      reference_pose=first_cam_pose,  # 传入完整的3x4位姿
      n_frames=n_frames,
      distance=distance,
      axis=axis,
      pitch=pitch_custom,
      roll=roll,
      yaw=yaw,
      x_scale=x_scale,
      y_scale=y_scale,
      z_scale=z_scale
  )

  new_poses = pad_poses(new_poses)

  traj = []  # 初始化轨迹列表
  # 遍历生成的每个新位姿，创建相机对象
  for idx, c2w in enumerate(tqdm(new_poses, desc="Generating trajectory")):
      # 再次应用坐标系转换，转回COLMAP坐标系
      c2w = c2w @ np.diag([1, -1, -1, 1])
      # 深拷贝第一个参考相机，并移动到CUDA设备
      cam = copy.deepcopy(viewpoint_cameras[0]).to_device("cuda")
      # 确保高度和宽度是偶数（某些渲染器要求）
      cam.height = int(cam.height / 2) * 2
      cam.width = int(cam.width / 2) * 2
      # 设置新的世界到相机的变换矩阵（c2w的逆矩阵的转置）
      cam.world_to_camera = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
      # 计算完整的投影矩阵：世界到相机变换 × 内参投影矩阵
      cam.full_projection = (cam.world_to_camera.unsqueeze(0).bmm(cam.projection.unsqueeze(0))).squeeze(0)
      # 计算相机中心位置（世界到相机矩阵的逆矩阵的第4列前3个元素）
      cam.camera_center = cam.world_to_camera.inverse()[3, :3]
      traj.append(cam)  # 将相机添加到轨迹列表
      # 如果指定了轨迹目录，保存相机参数为pickle文件
      if traj_dir is not None:
          # 保存为pickle文件，文件名格式如 00000.pkl
          with open(os.path.join(traj_dir, f"{str(idx).zfill(5)}.pkl"), "wb") as f:
              pickle.dump({
                  "world_view_transform": cam.world_to_camera.cpu().numpy(),  # 世界到视图变换矩阵
                  "image_height": cam.height,  # 图像高度
                  "image_width": cam.width,  # 图像宽度
                  "FoVx": cam.fov_x.item(),  # 水平视场角
              }, f)

  # 如果需要过滤，返回轨迹、变换矩阵和重新居中的位姿；否则只返回轨迹
  if filter:
    return traj, colmap_to_world_transform, pose_recenter
  else:
    return traj

def generate_single_pose(reference_pose: np.ndarray,
                         x_offset: float = 0.0,
                         y_offset: float = 0.0,
                         z_offset: float = 0.0,
                         yaw: float = 0.0,
                         pitch: float = 0.0,
                         roll: float = 0.0) -> np.ndarray:
    """
    基于参考位姿，生成单个调整后的相机位姿（位置偏移+角度调整）
    
    参数：
        reference_pose: 参考相机位姿（3x4矩阵）
        x_offset/y_offset/z_offset: 位置偏移（相对于参考位姿）
        yaw/pitch/roll: 角度调整（度）
    返回：
        单个调整后的位姿矩阵（3x4）
    """
    # 1. 提取参考位姿的旋转和位置
    R_original = reference_pose[:3, :3].copy()  # 原旋转矩阵
    original_pos = reference_pose[:3, 3].copy()  # 原位置

    # 2. 计算角度调整的旋转矩阵（复用之前的旋转逻辑）
    yaw_rad = np.deg2rad(yaw)
    pitch_rad = np.deg2rad(pitch)
    roll_rad = np.deg2rad(roll)

    # Yaw旋转矩阵（绕Y轴，正值向右）
    R_yaw = np.array([
        [np.cos(yaw_rad), 0, -np.sin(yaw_rad)],
        [0, 1, 0],
        [np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    # Pitch旋转矩阵（绕X轴，正值抬头）
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    # Roll旋转矩阵（绕Z轴，正值右旋）
    R_roll = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ])

    # 组合旋转矩阵
    R_rot = R_roll @ R_pitch @ R_yaw
    R_new = R_original @ R_rot  # 最终旋转矩阵

    # 3. 计算新位置（原位置 + 偏移量）
    new_pos = original_pos.copy()
    new_pos[0] += x_offset  # x轴偏移
    new_pos[1] += y_offset  # y轴偏移
    new_pos[2] += z_offset  # z轴偏移

    # 4. 构造单一位姿矩阵（3x4）
    up = -R_new[:, 1].copy()
    up = up / np.linalg.norm(up)
    look_dir = -R_new[:, 2].copy()
    look_dir = look_dir / np.linalg.norm(look_dir)

    return viewmatrix(look_dir, up, new_pos)  # 单个3x4位姿矩阵

def generate_path(viewpoint_cameras, traj_dir=None, n_frames=480, scale_percentile=90, shift=[0, 0], pitch=None, filter=False):
  """生成相机轨迹路径"""
  # 从所有输入相机中提取相机到世界的变换矩阵（c2w）
  # world_to_camera的转置再求逆得到camera_to_world
  c2ws = np.array([np.linalg.inv(np.asarray((cam.world_to_camera.T).cpu().numpy())) for cam in viewpoint_cameras])
  # 应用坐标系转换：将COLMAP坐标系转换为标准坐标系（翻转y和z轴）
  pose = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
  # 使用PCA变换位姿，使主成分对齐到XYZ轴，并获取变换矩阵
  pose_recenter, colmap_to_world_transform = transform_poses_pca(pose)

  new_poses = generate_ellipse_path(poses=pose_recenter, pitch=pitch, shift=shift,
                                    n_frames=n_frames, scale_percentile=scale_percentile)
  # 将新位姿变换回原始坐标系（逆变换） PCA处理
  new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(new_poses)

  traj = []  # 初始化轨迹列表
  # 遍历生成的每个新位姿，创建相机对象
  for idx, c2w in enumerate(tqdm(new_poses, desc="Generating trajectory")):
      # 再次应用坐标系转换，转回COLMAP坐标系
      c2w = c2w @ np.diag([1, -1, -1, 1])
      # 深拷贝第一个参考相机，并移动到CUDA设备
      cam = copy.deepcopy(viewpoint_cameras[0]).to_device("cuda")
      # 确保高度和宽度是偶数（某些渲染器要求）
      cam.height = int(cam.height / 2) * 2
      cam.width = int(cam.width / 2) * 2
      # 设置新的世界到相机的变换矩阵（c2w的逆矩阵的转置）
      cam.world_to_camera = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
      # 计算完整的投影矩阵：世界到相机变换 × 内参投影矩阵
      cam.full_projection = (cam.world_to_camera.unsqueeze(0).bmm(cam.projection.unsqueeze(0))).squeeze(0)
      # 计算相机中心位置（世界到相机矩阵的逆矩阵的第4列前3个元素）
      cam.camera_center = cam.world_to_camera.inverse()[3, :3]
      traj.append(cam)  # 将相机添加到轨迹列表
      # 如果指定了轨迹目录，保存相机参数为pickle文件
      if traj_dir is not None:
          # 保存为pickle文件，文件名格式如 00000.pkl
          with open(os.path.join(traj_dir, f"{str(idx).zfill(5)}.pkl"), "wb") as f:
              pickle.dump({
                  "world_view_transform": cam.world_to_camera.cpu().numpy(),  # 世界到视图变换矩阵
                  "image_height": cam.height,  # 图像高度
                  "image_width": cam.width,  # 图像宽度
                  "FoVx": cam.fov_x.item(),  # 水平视场角
              }, f)

  # 如果需要过滤，返回轨迹、变换矩阵和重新居中的位姿；否则只返回轨迹
  if filter:
    return traj, colmap_to_world_transform, pose_recenter
  else:
    return traj

def record_path(viewpoint_cameras, traj_dir=None):
  """记录相机路径（与generate_path类似，但不生成新路径，而是记录现有路径）"""
  # 从所有输入相机中提取相机到世界的变换矩阵
  c2ws = np.array([np.linalg.inv(np.asarray((cam.world_to_camera.T).cpu().numpy())) for cam in viewpoint_cameras])
  # 应用坐标系转换
  pose = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
  # 使用PCA变换位姿
  pose_recenter, colmap_to_world_transform = transform_poses_pca(pose)

  # 将重新居中的位姿变换回原始坐标系（不生成新路径，只是变换现有路径）
  new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(pose_recenter)

  traj = []  # 初始化轨迹列表
  # 遍历每个位姿，创建相机对象
  for idx, c2w in enumerate(tqdm(new_poses, desc="Generating trajectory")):
      # 应用坐标系转换
      c2w = c2w @ np.diag([1, -1, -1, 1])
      # 深拷贝参考相机并移动到CUDA
      cam = copy.deepcopy(viewpoint_cameras[0]).to_device("cuda")
      # 确保高度和宽度是偶数
      cam.height = int(cam.height / 2) * 2
      cam.width = int(cam.width / 2) * 2
      # 设置世界到相机的变换矩阵
      cam.world_to_camera = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
      # 计算完整投影变换矩阵（注意这里用的是full_proj_transform而不是full_projection）
      cam.full_proj_transform = (cam.world_to_camera.unsqueeze(0).bmm(cam.projection.unsqueeze(0))).squeeze(0)
      # 计算相机中心
      cam.camera_center = cam.world_to_camera.inverse()[3, :3]
      traj.append(cam)  # 添加到轨迹
      # 如果指定了轨迹目录，保存相机参数
      if traj_dir is not None:
          # 保存为pickle文件，文件名格式如 00000.pkl
          with open(os.path.join(traj_dir, f"{str(idx).zfill(5)}.pkl"), "wb") as f:
              pickle.dump({
                  "world_view_transform": cam.world_to_camera.cpu().numpy(),  # 世界到视图变换
                  "image_height": cam.height,  # 图像高度
                  "image_width": cam.width,  # 图像宽度
                  "FoVx": cam.fov_x.item(),  # 水平视场角
              }, f)

  return traj  # 返回轨迹列表

def generate_static_path(viewpoint_cameras, traj_dir=None, n_frames=480, filter=False):
  """生成静态相机轨迹：所有帧都使用第一张相机的位姿
  
  Args:
    viewpoint_cameras: 输入相机列表，将使用第一个相机的位姿
    traj_dir: 轨迹保存目录，如果为None则不保存
    n_frames: 生成的帧数
    filter: 是否返回变换矩阵（用于过滤，但静态路径不需要变换，返回None）
    
  Returns:
    如果filter=False，返回相机轨迹列表
    如果filter=True，返回 (轨迹列表, None, None)
  """
  # 获取第一个相机作为参考
  first_camera = viewpoint_cameras[0]
  
  traj = []  # 初始化轨迹列表
  
  # 生成n_frames个相同的相机位姿
  for idx in tqdm(range(n_frames), desc="Generating static trajectory"):
    # 深拷贝第一个参考相机，并移动到CUDA设备
    cam = copy.deepcopy(first_camera).to_device("cuda")
    # 确保高度和宽度是偶数（某些渲染器要求）
    cam.height = int(cam.height / 2) * 2
    cam.width = int(cam.width / 2) * 2
    # 保持原始的世界到相机变换矩阵（不需要修改，因为位姿相同）
    # 重新计算完整投影矩阵（确保一致性）
    cam.full_projection = (cam.world_to_camera.unsqueeze(0).bmm(cam.projection.unsqueeze(0))).squeeze(0)
    # 重新计算相机中心位置（确保一致性）
    cam.camera_center = cam.world_to_camera.inverse()[3, :3]
    traj.append(cam)  # 将相机添加到轨迹列表
    
    # 如果指定了轨迹目录，保存相机参数为pickle文件
    if traj_dir is not None:
      # 保存为pickle文件，文件名格式如 00000.pkl
      with open(os.path.join(traj_dir, f"{str(idx).zfill(5)}.pkl"), "wb") as f:
        pickle.dump({
          "world_view_transform": cam.world_to_camera.cpu().numpy(),  # 世界到视图变换矩阵
          "image_height": cam.height,  # 图像高度
          "image_width": cam.width,  # 图像宽度
          "FoVx": cam.fov_x.item(),  # 水平视场角
        }, f)
  
  # 如果需要过滤，返回轨迹和None（静态路径不需要变换矩阵）
  if filter:
    return traj, None, None
  else:
    return traj

def load_img(pth: str) -> np.ndarray:
  """加载图像并转换为float32类型"""
  with open(pth, 'rb') as f:  # 以二进制模式打开文件
    image = np.array(Image.open(f), dtype=np.float32)  # 使用PIL打开图像并转换为numpy数组，数据类型为float32
  return image  # 返回图像数组


def create_videos(base_dir, input_dir, out_name, num_frames=480):
  """从保存到磁盘的图像创建视频"""
  # 检查点路径的最后两部分是实验名称和场景名称
  video_prefix = f'{out_name}'  # 视频文件名前缀

  # 计算所需的零填充位数（至少5位，或根据帧数确定）
  zpad = max(5, len(str(num_frames - 1)))
  idx_to_str = lambda idx: str(idx).zfill(zpad)  # 将索引转换为零填充的字符串

  os.makedirs(base_dir, exist_ok=True)  # 创建输出目录（如果不存在）
  render_dist_curve_fn = np.log  # 深度渲染的距离曲线函数（对数变换）
  
  # 加载一个示例帧以获取图像形状和深度范围
  depth_file = os.path.join(input_dir, 'vis', f'depth_{idx_to_str(0)}.tiff')
  depth_frame = load_img(depth_file)  # 加载深度图
  shape = depth_frame.shape  # 获取图像形状
  p = 3  # 百分位数阈值（用于去除异常值）
  # 计算深度的百分位数范围（去除3%的异常值）
  distance_limits = np.percentile(depth_frame.flatten(), [p, 100 - p])
  # 对距离限制应用对数变换
  lo, hi = [render_dist_curve_fn(x) for x in distance_limits]
  print(f'Video shape is {shape[:2]}')  # 打印视频形状

  # 视频编码参数
  video_kwargs = {
      'shape': shape[:2],  # 视频尺寸（高度，宽度）
      'codec': 'h264',  # 视频编解码器
      'fps': 60,  # 帧率
      'crf': 18,  # 恒定速率因子（质量参数，越小质量越高）
  }
  
  # 为深度、法线和颜色分别创建视频
  for k in ['depth', 'normal', 'color']:
    video_file = os.path.join(base_dir, f'{video_prefix}_{k}.mp4')  # 视频文件路径
    input_format = 'gray' if k == 'alpha' else 'rgb'  # 输入格式（灰度或RGB）
    

    file_ext = 'png' if k in ['color', 'normal'] else 'tiff'  # 文件扩展名
    idx = 0  # 初始化索引

    # 确定第一帧的文件路径（用于检查文件是否存在）
    if k == 'color':
      file0 = os.path.join(input_dir, 'renders', f'{idx_to_str(0)}.{file_ext}')
    else:
      file0 = os.path.join(input_dir, 'vis', f'{k}_{idx_to_str(0)}.{file_ext}')

    # 如果第一帧不存在，跳过该类型
    if not os.path.exists(file0):
      print(f'Images missing for tag {k}')
      continue
    print(f'Making video {video_file}...')  # 打印正在创建的视频
    # 创建视频写入器
    with media.VideoWriter(
        video_file, **video_kwargs, input_format=input_format) as writer:
      # 遍历所有帧
      for idx in tqdm(range(num_frames)):
        # img_file = os.path.join(input_dir, f'{k}_{idx_to_str(idx)}.{file_ext}')
        # 根据类型确定图像文件路径
        if k == 'color':
          img_file = os.path.join(input_dir, 'renders', f'{idx_to_str(idx)}.{file_ext}')
        else:
          img_file = os.path.join(input_dir, 'vis', f'{k}_{idx_to_str(idx)}.{file_ext}')

        # 检查文件是否存在
        if not os.path.exists(img_file):
          ValueError(f'Image file {img_file} does not exist.')
        img = load_img(img_file)  # 加载图像
        # 根据类型处理图像
        if k in ['color', 'normal']:
          img = img / 255.  # 将uint8图像归一化到[0,1]
        elif k.startswith('depth'):
          img = render_dist_curve_fn(img)  # 对深度应用对数变换
          # 归一化深度值到[0,1]范围
          img = np.clip((img - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1)
          # 应用turbo颜色映射（从蓝色到红色）
          img = cm.get_cmap('turbo')(img)[..., :3]

        # 将图像转换为uint8格式（0-255）
        frame = (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)
        writer.add_image(frame)  # 将帧添加到视频
        idx += 1  # 递增索引

def save_img_u8(img, pth):
  """将[0, 1]范围内的图像（可能是RGB）保存为uint8格式的PNG文件"""
  with open(pth, 'wb') as f:  # 以二进制写入模式打开文件
    # 将图像从[0,1]范围转换为uint8格式（0-255），并保存为PNG
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
            f, 'PNG')  # 处理NaN值，裁剪到[0,1]，缩放到[0,255]，转换为uint8


def save_img_f32(depthmap, pth):
  """将图像（可能是深度图）保存为float32格式的TIFF文件"""
  with open(pth, 'wb') as f:  # 以二进制写入模式打开文件
    # 处理NaN值，转换为float32格式，保存为TIFF
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, 'TIFF')