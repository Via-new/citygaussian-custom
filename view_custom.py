from gaussian_viewer import GaussianViewer
from plyfile import PlyData
import numpy as np

# 读取PLY文件
ply_path = "outputs/citygsv2_mc_aerial_custom/checkpoints/epoch=11-step=60000-xyz_rgb.ply"
ply_data = PlyData.read(ply_path)

# 提取顶点、颜色等信息（需根据PLY文件结构调整）
vertices = np.vstack([ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']]).T
colors = np.vstack([
    ply_data['vertex']['red'], 
    ply_data['vertex']['green'], 
    ply_data['vertex']['blue']
]).T / 255.0  # 转换为0-1范围

# 启动Web Viewer
viewer = GaussianViewer(port=8000)
viewer.show(vertices=vertices, colors=colors)
viewer.open_in_browser()  # 自动在浏览器中打开