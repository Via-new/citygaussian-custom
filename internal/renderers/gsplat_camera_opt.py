from typing import Tuple, Optional, Any, List, Dict, Mapping, Literal
from dataclasses import dataclass, field
import lightning
import copy
import torch
import torch.nn.functional as F
from torch import nn

from ..cameras import Camera
from ..utils.colmap import rotmat2qvec_torch, qvec2rotmat_torch
from ..models.gaussian import GaussianModel

from .gsplat_v1_renderer import GSplatV1Renderer, GSplatV1RendererModule, spherical_harmonics, spherical_harmonics_decomposed
from .gsplat_mip_splatting_renderer_v2 import MipSplattingRendererMixin

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

@dataclass
class ModelConfig:
    
    n_cameras: int = -1
    pose_opt_type: Literal["sfm", "mlp", "7dmlp"] = "sfm"
    cam_scale: float = 1.0
    mlp_width: int = 64
    mlp_depth: int = 2


@dataclass
class OptimizationConfig:
    
    embeds_lr : float = 1e-5
    embeds_lr_final_factor: float = 1.0  # No decay by default
    embeds_weight_decay: float = 0.0
    eps: float = 1e-15
    max_steps: int = 30_000


class CameraOptModule(nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int, cam_scale=None):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: torch.Tensor, embed_ids: torch.Tensor) -> torch.Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)
    

class CameraOptModuleMLP(torch.nn.Module):
    """Camera pose optimization module using MLP."""

    def __init__(self, n: int, mlp_width: int = 64, mlp_depth: int = 2, cam_scale: float = 1.0):
        super().__init__()
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        
        # Initial embeddings for each camera
        self.embeds = torch.nn.Embedding(n, mlp_width)
        self.num_cams = n
        
        # MLP layers
        activation = torch.nn.ReLU(inplace=True)
        layers = []
        layers.append(torch.nn.Linear(mlp_width, mlp_width))
        layers.append(activation)
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(activation)
        # Output layer produces 9D adjustments (3D position + 6D rotation)
        layers.append(torch.nn.Linear(mlp_width, 9))
        self.mlp = torch.nn.Sequential(*layers)

        self.cam_scale = cam_scale
        
    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)
        #torch.nn.init.normal_(self.embeds.weight)
        # Also initialize the last layer of MLP with small weights
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)
        # Initialize the last layer of MLP with small weights
        torch.nn.init.normal_(self.mlp[-1].weight, std=std)
        torch.nn.init.normal_(self.mlp[-1].bias, std=std)

    def forward(self, camtoworlds: torch.Tensor, embed_ids: torch.Tensor) -> torch.Tensor:
        """Adjust camera pose based on MLP outputs with SGLD noise.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]
        
        # Get embeddings and process through MLP with noise
        embeddings = self.embeds(embed_ids)  # (..., mlp_width)
        pose_deltas = self.mlp(embeddings)  # (..., 9)
        
        # Split into position and rotation deltas
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )  # (..., 3, 3)
        
        # Create transformation matrix
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx * self.cam_scale
            
        return torch.matmul(camtoworlds, transform)

class CameraOptModule7dMLP(torch.nn.Module):
    """Camera pose optimization module using MLP."""

    def __init__(self, n: int, mlp_width: int = 64, mlp_depth: int = 4, cam_scale: float = 1.0):
        super().__init__()
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        
        # Initial embeddings for each camera
        self.num_cams = n
        
        # MLP layers
        activation = torch.nn.ReLU(inplace=True)
        layers = []
        layers.append(torch.nn.Linear(7, mlp_width))
        layers.append(activation)
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(activation)
        # Output layer produces 9D adjustments (3D position + 6D rotation)
        layers.append(torch.nn.Linear(mlp_width, 7))
        self.mlp = torch.nn.Sequential(*layers)

        self.cam_scale = cam_scale
        
    def zero_init(self):
        # torch.nn.init.zeros_(self.embeds.weight)
        #torch.nn.init.normal_(self.embeds.weight)
        # Also initialize the last layer of MLP with small weights
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def random_init(self, std: float):
        # torch.nn.init.normal_(self.embeds.weight, std=std)
        # Initialize the last layer of MLP with small weights
        torch.nn.init.normal_(self.mlp[-1].weight, std=std)
        torch.nn.init.normal_(self.mlp[-1].bias, std=std)

    def forward(self, camtoworlds: torch.Tensor, embed_ids: torch.Tensor) -> torch.Tensor:
        """Adjust camera pose based on MLP outputs with SGLD noise.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        if camtoworlds.ndim == 2:
            camtoworlds = camtoworlds.unsqueeze(0)
        batch_shape = camtoworlds.shape[:-2]
        
        # Get embeddings and process through MLP with noise
        quaternions = rotmat2qvec_torch(camtoworlds[..., :3, :3])  # (..., 4)
        translations = camtoworlds[..., :3, 3] / self.cam_scale  # (..., 3)
        if translations.ndim == 1:
            translations = translations  # shape: (1, 3, 3)
        mlp_input = torch.cat((quaternions, translations), dim=-1)  # (..., 7)

        pose_deltas = self.mlp(mlp_input)  # (..., 7)
        
        pose_corrected = mlp_input + pose_deltas  # (..., 7)
        rot = qvec2rotmat_torch(pose_corrected[..., :4])
        dx = pose_corrected[..., 4:]
        
        camtoworlds_corrected = torch.eye(4, device=camtoworlds.device).repeat((*batch_shape, 1, 1))
        camtoworlds_corrected[..., :3, :3] = rot
        camtoworlds_corrected[..., :3, 3] = dx * self.cam_scale
            
        return camtoworlds_corrected.squeeze()

@dataclass
class GSplatCameraOptRenderer(GSplatV1Renderer):

    model: ModelConfig = field(default_factory=lambda: ModelConfig())

    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def instantiate(self, *args, **kwargs) -> "GSplatCameraOptRendererModule":

        if getattr(self, "model_config", None) is not None:
            # checkpoint generated by previous version
            self.model = self.config.model
            self.optimization = self.config.optimization

        return GSplatCameraOptRendererModule(self)


class GSplatCameraOptRendererModule(GSplatV1RendererModule):
    """
    rgb = f(point_features, appearance_embedding, view_direction)
    """

    def setup(self, stage: str, lightning_module=None, *args: Any, **kwargs: Any) -> Any:
        if lightning_module is not None:
            if self.config.model.n_cameras <= 0:
                dataparser_outputs = lightning_module.trainer.datamodule.dataparser_outputs
                self.config.model.n_cameras = len(dataparser_outputs.appearance_group_ids)
                self.config.model.cam_scale = dataparser_outputs.camera_extent
                
            self._setup_model(lightning_module.device)
            print(self.model)

    def _setup_model(self, device=None):
        if self.config.model.pose_opt_type == "mlp":
            self.model = CameraOptModuleMLP(
                n=self.config.model.n_cameras,
                mlp_width=self.config.model.mlp_width,
                mlp_depth=self.config.model.mlp_depth,
                cam_scale=self.config.model.cam_scale
            )
        elif self.config.model.pose_opt_type == "7dmlp":
            self.model = CameraOptModule7dMLP(
                n=self.config.model.n_cameras,
                mlp_width=self.config.model.mlp_width,
                mlp_depth=self.config.model.mlp_depth,
                cam_scale=self.config.model.cam_scale
            )
        else:
            self.model = CameraOptModule(self.config.model.n_cameras)
        
        self.model.zero_init()
        if device is not None:
            self.model.to(device=device)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        self.config.model.n_cameras = state_dict["model.embeds.weight"].shape[0]
        self._setup_model(device=state_dict["model.embeds.weight"].device)
        return super().load_state_dict(state_dict, strict)

    def training_setup(self, module: lightning.LightningModule):

        embedding_optimizer, embedding_scheduler = self._create_optimizer_and_scheduler(
            self.model.parameters(),
            "pose_params",
            lr_init=self.config.optimization.embeds_lr,
            weight_decay=self.config.optimization.embeds_weight_decay,
            lr_final_factor=self.config.optimization.embeds_lr_final_factor,
            max_steps=self.config.optimization.max_steps,
            eps=self.config.optimization.eps,
        )

        return embedding_optimizer, embedding_scheduler

    def forward(self, viewpoint_camera, pc, bg_color: torch.Tensor, scaling_modifier=1.0, render_types: list = None, **kwargs):
        w2cs = torch.transpose(viewpoint_camera.world_to_camera, -2, -1)
        c2ws = torch.linalg.inv(w2cs)
        c2ws_corrected = self.model(c2ws, viewpoint_camera.appearance_id)
        
        viewpoint_camera_corrected = copy.deepcopy(viewpoint_camera)
        viewpoint_camera_corrected.world_to_camera = torch.transpose(torch.linalg.inv(c2ws_corrected), -2, -1)

        return super().forward(
            viewpoint_camera=viewpoint_camera_corrected,
            pc=pc, bg_color=bg_color,
            scaling_modifier=scaling_modifier,
            render_types=render_types,
            **kwargs
        )

    def training_forward(self, step: int, module: lightning.LightningModule, viewpoint_camera: Camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, **kwargs):
        return self.forward(viewpoint_camera, pc, bg_color, scaling_modifier, **kwargs)

    @staticmethod
    def _create_optimizer_and_scheduler(
            params,
            name,
            lr_init,
            lr_final_factor,
            weight_decay,
            max_steps,
            eps,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = torch.optim.Adam(
            params=[
                {"params": list(params), "name": name}
            ],
            lr=lr_init,
            weight_decay=weight_decay,
            eps=eps,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=lr_final_factor ** (1 / max_steps),
        )

        return optimizer, scheduler


# With MipSplatting version

@dataclass
class GSplatCameraOptMipRenderer(GSplatCameraOptRenderer):
    filter_2d_kernel_size: float = 0.1

    def instantiate(self, *args, **kwargs) -> "GSplatCameraOptMipRendererModule":
        return GSplatCameraOptMipRendererModule(self)


class GSplatCameraOptMipRendererModule(MipSplattingRendererMixin, GSplatCameraOptRendererModule):
    pass
