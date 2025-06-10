import os
import sys
import yaml
import torch
import torch_scatter
import alphashape
import numpy as np
from torch import nn
from tqdm import tqdm
from shapely.geometry import Point
from argparse import ArgumentParser, Namespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.models.vanilla_gaussian import VanillaGaussian
from internal.renderers.vanilla_renderer import VanillaRenderer
from internal.utils.graphics_utils import fetch_ply
from internal.utils.general_utils import inverse_sigmoid

def points_in_polygon(points: torch.Tensor, polygon: torch.Tensor) -> torch.Tensor:
    """
    Determine whether each point in `points` is inside the polygon defined by `polygon`.

    Args:
        points (torch.Tensor): Tensor of shape (N, 2) representing N points (x, y).
        polygon (torch.Tensor): Tensor of shape (M, 2) representing polygon vertices (x, y) in order.
                                 The polygon does not have to be explicitly closed.

    Returns:
        torch.Tensor: A boolean tensor of shape (N,) where True indicates the point is inside the polygon.
    """
    # Ensure the polygon is closed: repeat the first vertex at the end if needed.
    if not torch.allclose(polygon[0], polygon[-1]):
        polygon = torch.cat([polygon, polygon[:1]], dim=0)
    
    # Separate points into x and y coordinates: shape (N, 1)
    x = points[:, 0:1]
    y = points[:, 1:2]

    # Separate polygon coordinates
    poly_x = polygon[:, 0]
    poly_y = polygon[:, 1]
    
    # Form segments of polygon edges; note that the polygon is already closed.
    poly_x1 = poly_x[:-1]  # starting x for each edge
    poly_y1 = poly_y[:-1]  # starting y for each edge
    poly_x2 = poly_x[1:]   # ending x for each edge
    poly_y2 = poly_y[1:]   # ending y for each edge

    # Broadcast to compare each point with each edge. 
    # Condition for y: the y-coordinate of the point is between the y's of the segment's endpoints.
    cond = ((poly_y1.unsqueeze(0) > y) != (poly_y2.unsqueeze(0) > y))

    # Compute the x coordinate where the edge intersects the horizontal line at y.
    # Handle division by zero carefully; since the condition already screens out horizontal lines 
    # (they would not contribute to an intersection) the division is valid.
    x_intersect = poly_x1.unsqueeze(0) + (y - poly_y1.unsqueeze(0)) * (poly_x2.unsqueeze(0) - poly_x1.unsqueeze(0)) / (poly_y2.unsqueeze(0) - poly_y1.unsqueeze(0))
    
    # Check if the x coordinate of the point is to the left of this intersection.
    intersections = (x < x_intersect)

    # Count intersections across all edges for each point.
    count = torch.sum(cond & intersections, dim=1)

    # A point is inside if and only if the count of intersections is odd.
    inside = (count % 2 == 1)
    return inside

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

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--output_path', type=str, help='path of config', default=None)
    parser.add_argument('--ply_path', type=str, help='path of config', default='data/GauU_Scene/LFLS/LFLS_ds.ply')
    parser.add_argument('--data_path', type=str, help='path of data', default=None)
    parser.add_argument("--train", action="store_true", help="whether to use train set as trajectories")
    parser.add_argument("--vox_grid", type=int, help="number of voxelization grid", default=25)
    parser.add_argument("--std_ratio", type=float, help="used to control filtering threshold", default=2.0)
    parser.add_argument('--transform_path', type=str, help='path of transformation matrix (txt)', default=None)
    parser.add_argument('--vis_threshold', type=int, help='emperical threhold for visibility frequency', default=60)
    args = parser.parse_args(sys.argv[1:])

    loadable_file = GaussianModelLoader.search_load_file(args.output_path)
    ckpt = torch.load(loadable_file, map_location="cpu")

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

    if args.train:
        cameras = dataparser_outputs.train_set.cameras
    else:
        cameras = dataparser_outputs.test_set.cameras

    device = torch.device("cuda")
    bkgd_color = torch.tensor(ckpt["hyper_parameters"]["background_color"], device=device)

    # load groundtruth point cloud and dataset
    model = VanillaGaussian(
        sh_degree=3,
    ).instantiate()
    pcd = fetch_ply(args.ply_path)
    model.setup_from_pcd(xyz=pcd.points, rgb=pcd.colors)
    model = model.to("cuda")
    model._opacity = nn.Parameter(inverse_sigmoid(torch.ones((model.get_xyz.shape[0], 1), 
                                                             dtype=torch.float, device="cuda") * 0.3))

    renderer = VanillaRenderer()
    renderer.setup(stage="val")
    renderer = renderer.to("cuda")

    # count visibility frequency
    dataset = dataparser_outputs.train_set
    bg_color=torch.tensor([0, 0, 0], dtype=torch.float, device="cuda")
    with torch.no_grad():
        visible_cnt = torch.zeros(model.get_xyz.shape[0], dtype=torch.long, device="cuda")
        for idx in tqdm(range(0, len(dataset.cameras))):
            camera = dataset.cameras[idx].to_device("cuda")
            output = renderer(camera, model, bg_color=bg_color)
            visible_cnt[output['visibility_filter']] += 1
        mask = visible_cnt.cpu().numpy() > args.vis_threshold
        xyz = np.float32(model.get_xyz.cpu().numpy())
    
    # load transformation matrix (align z axis to vertical direction)
    with open(args.transform_path, 'r') as f:
        transform = np.float32(np.loadtxt(f))
    xyz_homo = np.concatenate([xyz[:, :3], np.ones_like(xyz[:, :1])], axis=-1)
    xyz = (xyz_homo @ np.linalg.inv(transform).T)[:, :3]

    # generate alpha shape
    hull = alphashape.alphashape(xyz[mask][::200, :2], alpha=2.0)
    # if MultiPolygon, take the smallest convex Polygon containing all the points in the object
    hull = hull.convex_hull if hull.geom_type == 'MultiPolygon' else hull
    exterior = torch.tensor(hull.exterior.xy, device=device).permute(1, 0)

    model = GaussianModelLoader.initialize_model_from_checkpoint(
        ckpt,
        device=device,
    )
    model.freeze()
    model.pre_activate_all_properties()
    xyz_homo = torch.cat([model.get_xyz[:, :3], torch.ones_like(model.get_xyz[:, :1])], dim=-1)
    xyz = (xyz_homo @ torch.tensor(np.linalg.inv(transform).T, device=xyz_homo.device))[:, :3]
    inside = points_in_polygon(xyz[:, :2], exterior)

    print(f"{inside.sum()} of {len(inside)} gaussians are inside the region")

    mask = ~inside

    with torch.no_grad():
        x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
        y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
        voxel_size = torch.tensor([(x_max - x_min) / args.vox_grid, (y_max - y_min) / args.vox_grid], device="cuda")
        xy_range = torch.tensor([x_min, y_min, x_max, y_max], device="cuda")
        mask |= voxel_filtering_no_gt(voxel_size, xy_range, xyz, args.std_ratio).bool()
        print(f"Filtered out {mask.sum()} of {len(mask)} gaussians according to visibility and statistics")

    ckpt['state_dict']['gaussian_model.gaussians.opacities'][mask] = -13.8  # 1e-6
    torch.save(ckpt, loadable_file.replace('.ckpt', '_filtered.ckpt'))
    
