"""
3D Gaussian Splatting as Markov Chain Monte Carlo
https://ubc-vision.github.io/3dgs-mcmc/

Most codes are copied from https://github.com/ubc-vision/3dgs-mcmc
"""

import os
import torch
import kornia
import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass, field

from .metric import MetricImpl
from .mcmc_metrics import MCMCMetrics, MCMCMetricsImpl
from ..renderers.gsplat_camera_opt import GSplatCameraOptRendererModule
from ..configs.instantiate_config import InstantiatableConfig

@dataclass
class PoseOptMCMCMetrics(MCMCMetrics):

    corr_path: str = None

    epi_loss_max_iter: int = 5_000

    epi_loss_weight: float = 0.01

    def instantiate(self, *args, **kwargs) -> MetricImpl:
        return PoseOptMCMCMetricsImpl(self)

class PoseOptMCMCMetricsImpl(MCMCMetricsImpl):

    def __init__(self, config: InstantiatableConfig, *args, **kwargs) -> None:

        super(PoseOptMCMCMetricsImpl, self).__init__(config, *args, **kwargs)

        self.corr_points_i_normalized = None
        self.corr_points_j_normalized = None
        self.corr_points_i = None
        self.corr_points_j = None
        self.image_names_i = None
        self.image_names_j = None
        self.appearance_ids_i = None
        self.appearance_ids_j = None
        self.initialized = False

        self.c2ws_i = None
        self.c2ws_j = None
        self.Ks_i = None
        self.Ks_j = None

        if config.corr_path is not None:
            print(f"Loading tracking data from {config.corr_path}")
            correspondence = np.load(config.corr_path, allow_pickle=True)
            self.corr_points_i_normalized = correspondence.item()["corr_points_i_normalized"]
            self.corr_points_j_normalized =correspondence.item()["corr_points_j_normalized"]
            self.image_names_i = correspondence.item()["image_names_i"]
            self.image_names_j = correspondence.item()["image_names_j"]
    
    def initialize_corr(self, trainset, device):

        in_mask_i = np.isin(self.image_names_i, trainset.image_names)
        in_mask_j = np.isin(self.image_names_j, trainset.image_names)
        in_mask = in_mask_i & in_mask_j

        self.corr_points_i = torch.tensor(self.corr_points_i_normalized[in_mask], device=trainset.cameras.R.device, dtype=trainset.cameras.R.dtype)
        self.corr_points_j = torch.tensor(self.corr_points_j_normalized[in_mask], device=trainset.cameras.R.device, dtype=trainset.cameras.R.dtype)
        self.image_names_i = self.image_names_i[in_mask]
        self.image_names_j = self.image_names_j[in_mask]

        c2ws = torch.linalg.inv(torch.transpose(trainset.cameras.world_to_camera, -2, -1))
        Ks = torch.stack([camera.get_K() for camera in trainset.cameras], dim=0)
        indexes_i = [trainset.image_names.index(img_name) for img_name in self.image_names_i]
        indexes_j = [trainset.image_names.index(img_name) for img_name in self.image_names_j]
        self.c2ws_i = c2ws[indexes_i]
        self.c2ws_j = c2ws[indexes_j]
        self.Ks_i = Ks[indexes_i]
        self.Ks_j = Ks[indexes_j]

        self.corr_points_i[:, 0] *= trainset.cameras.width[indexes_i]
        self.corr_points_i[:, 1] *= trainset.cameras.height[indexes_i]
        self.corr_points_j[:, 0] *= trainset.cameras.width[indexes_j]
        self.corr_points_j[:, 1] *= trainset.cameras.height[indexes_j]

        self.appearance_ids_i = trainset.cameras.appearance_id[indexes_i].to(device)
        self.appearance_ids_j = trainset.cameras.appearance_id[indexes_j].to(device)

        self.corr_points_i = self.corr_points_i.to(device)
        self.corr_points_j = self.corr_points_j.to(device)
        self.c2ws_i = self.c2ws_i.to(device)
        self.c2ws_j = self.c2ws_j.to(device)
        self.Ks_i = self.Ks_i.to(device)
        self.Ks_j = self.Ks_j.to(device)

        self.initialized = True
        print(f"[INFO] Initialized {len(self.corr_points_i)} correspondences from {self.config.corr_path}.")
    
    def epipolar_loss(self, pl_module, basic_metrics: Tuple[Dict[str, Any], Dict[str, bool]], step: int) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        if not self.initialized and self.config.corr_path is not None:
            train_set = pl_module._trainer.datamodule.dataparser_outputs.train_set
            self.initialize_corr(train_set, device=basic_metrics[0]["loss"].device)
        
        if self.initialized and step < self.config.epi_loss_max_iter and \
              isinstance(pl_module.renderer, GSplatCameraOptRendererModule):
            c2ws_corrected_i = pl_module.renderer.model(self.c2ws_i, self.appearance_ids_i)
            c2ws_corrected_j = pl_module.renderer.model(self.c2ws_j, self.appearance_ids_j)
            w2cs_corrected_i = torch.linalg.inv(c2ws_corrected_i)
            w2cs_corrected_j = torch.linalg.inv(c2ws_corrected_j)

            P_i = self.Ks_i @ w2cs_corrected_i
            P_j = self.Ks_j @ w2cs_corrected_j
            Fm = kornia.geometry.epipolar.fundamental_from_projections(P_i[:, :3], P_j[:, :3])
            err = kornia.geometry.symmetrical_epipolar_distance(self.corr_points_i[:, None, :2], self.corr_points_j[:, None, :2], Fm, squared=False, eps=1e-08)
            lepipolar = err.mean()
        
        else:
            lepipolar = torch.tensor(0.0, device=basic_metrics[0]["loss"].device)

        basic_metrics[0]["loss"] = basic_metrics[0]["loss"] + lepipolar * self.config.epi_loss_weight
        basic_metrics[0]["err_epipolar"] = lepipolar
        basic_metrics[1]["err_epipolar"] = True

        return basic_metrics

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        basic_metrics = super().get_train_metrics(pl_module, gaussian_model, step, batch, outputs)
        return self.epipolar_loss(pl_module, basic_metrics, step)
