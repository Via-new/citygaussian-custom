import os
import sys

import random
import torch
import numpy as np
from argparse import ArgumentParser
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from internal.utils.gaussian_model_loader import GaussianModelLoader
from tools.metric_torch import evaluate_auc, umeyama


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--result_path', type=str, help='path of ckpt file', default=None)
    parser.add_argument('--ref_folder', type=str, help='path of reference folder', default=None)
    parser.add_argument('--disable_correction', action='store_true', help='disable pose correction')
    parser.add_argument('--device', type=str, help='device to use', default='cuda:0')
    args = parser.parse_args(sys.argv[1:])

    # loading model and renderer
    assert 'test' in args.result_path, "Please provide a test checkpoint path."

    load_from = GaussianModelLoader.search_load_file(args.result_path)
    ckpt = torch.load(load_from, map_location="cpu")
    scene = ckpt["datamodule_hyper_parameters"]["path"].split("/")[-1]
    ref_dataset_path = os.path.join(args.ref_folder, scene)
    dataset_path = ckpt["datamodule_hyper_parameters"]["path"]

    bkgd_color = ckpt["hyper_parameters"]["background_color"]
    model = GaussianModelLoader.initialize_model_from_checkpoint(
        ckpt,
        device=args.device,
    )
    model.freeze()
    model.pre_activate_all_properties()

    # initialize renderer
    renderer = GaussianModelLoader.initialize_renderer_from_checkpoint(
        ckpt,
        stage="validate",
        device=args.device,
    )
    print("[INFO] Gaussian count: {}".format(model.get_xyz.shape[0]))

    dataparser_config = ckpt["datamodule_hyper_parameters"]["parser"]

    dataparser_outputs = dataparser_config.instantiate(
        path=dataset_path,
        output_path=os.getcwd(),
        global_rank=0,
    ).get_outputs()

    ref_dataparser_outputs = dataparser_config.instantiate(
        path=ref_dataset_path,
        output_path=os.getcwd(),
        global_rank=0,
    ).get_outputs()

    # correct test view poses
    w2cs_train_pred = torch.transpose(dataparser_outputs.train_set.cameras.world_to_camera, -2, -1).to(args.device)
    w2cs_val_pred = torch.transpose(dataparser_outputs.val_set.cameras.world_to_camera, -2, -1).to(args.device)
    w2cs_pred = torch.concatenate([w2cs_train_pred, w2cs_val_pred], dim=0)
    appearance_ids_pred = torch.concatenate([dataparser_outputs.train_set.cameras.appearance_id, 
                                            dataparser_outputs.val_set.cameras.appearance_id], dim=0).to(args.device)
    
    if args.disable_correction:
        print("[INFO] Pose correction disabled.")
        w2cs_pred_corrected = w2cs_pred
        prefix="uncorrected_"
    else:
        c2ws_pred = torch.linalg.inv(w2cs_pred)
        c2ws_pred_corrected = renderer.model(c2ws_pred, appearance_ids_pred)
        w2cs_pred_corrected = torch.linalg.inv(c2ws_pred_corrected)
        prefix="corrected_"

    pred_se3 = torch.eye(4, device=args.device).unsqueeze(0).repeat(len(w2cs_pred_corrected), 1, 1)
    pred_se3[:, :3, :3] = torch.tensor(w2cs_pred_corrected[:, :3, :3], device=args.device)
    pred_se3[:, 3, :3] = torch.tensor(w2cs_pred_corrected[:, :3, 3], device=args.device)

    w2cs_train_ref = torch.transpose(ref_dataparser_outputs.train_set.cameras.world_to_camera, -2, -1).to(args.device)
    w2cs_val_ref = torch.transpose(ref_dataparser_outputs.val_set.cameras.world_to_camera, -2, -1).to(args.device)
    w2cs_ref = torch.concatenate([w2cs_train_ref, w2cs_val_ref], dim=0)

    gt_se3 = torch.eye(4, device=args.device).unsqueeze(0).repeat(len(w2cs_ref), 1, 1)
    gt_se3[:, :3, :3] = torch.tensor(w2cs_ref[:, :3, :3], device=args.device)
    gt_se3[:, 3, :3] = torch.tensor(w2cs_ref[:, :3, 3], device=args.device)

    auc_results = evaluate_auc(pred_se3, gt_se3, device=args.device)

    result_file = os.path.join(args.result_path, f"{prefix}pose_results.txt")
    with open(result_file, "w") as f:
        f.write(f"Image Count: {len(w2cs_ref)},\n")
        f.write(f"Relative Rotation Error (degrees): {auc_results['rel_rangle_deg']},\n")
        f.write(f"Relative Translation Error (degrees): {auc_results['rel_tangle_deg']},\n")
        f.write(f"AUC at 30 degrees: {auc_results['Auc_30']},\n")
        f.write(f"Racc_5: {auc_results['Racc_5']},\n")
        f.write(f"Racc_15: {auc_results['Racc_15']},\n")
        f.write(f"Tacc_5: {auc_results['Tacc_5']},\n")
        f.write(f"Tacc_15: {auc_results['Tacc_15']},\n")
    
    print("[INFO] Pose evaluation results saved to {}".format(result_file))