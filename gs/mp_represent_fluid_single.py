"""
Gaussian reconstruction script for single-frame fluid velocity fields.

This file is simplified from mp_represent_fluid.py by removing temporal logic.
It supports batch processing of multiple independent single-frame fluid samples.
"""

import os
import sys
import time
import h5py
from pathlib import Path
import logging
import logging.handlers
import torch.multiprocessing as multiprocessing
import torch
import torch.nn.functional as F
import numpy as np
from random import randint

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, OptimizationParams

from radar_gs.render import render, render_with_fourier_modulation
from radar_gs.gaussian_model import GaussianModel
from radar_gs.camera import cameraList_from_camInfos
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, l2_loss, SSIM, Energy

try:
    from torch.utils.tensorboard.writer import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# ========== Data loading ==========

from typing import NamedTuple

class CameraInfo(NamedTuple):
    R: np.array
    T: np.array
    view_depth: float
    gt_image: np.array
    channel: int
    width: int
    height: int


class SceneInfo:
    def __init__(self, point_cloud=None, xoy=None, supplement=None):
        self.point_cloud = point_cloud
        self.xoy = xoy
        self.supplement = supplement

    def set_xoy(self, xoy):
        self.xoy = xoy

    def set_supplement(self, supplement):
        self.supplement = supplement


def load_single_frame(hdf_path, device="cpu"):
    """
    Load a single-frame fluid field.

    Args:
        hdf_path: Path to the HDF5 file.
        device: Target device.

    Returns:
        data: Velocity tensor with shape [3, D, H, W].
    """
    with h5py.File(hdf_path, "r") as f:
        vx = f["Vx"][:]
        vy = f["Vy"][:]
        vz = f["Vz"][:]

    data = np.stack([vx, vy, vz], axis=0)  # [3, D, H, W]; adjust if the source layout differs
    data = torch.tensor(data, dtype=torch.float32, device=device)

    # Ensure the tensor layout is [3, D, H, W].
    if data.dim() == 4 and data.shape[0] == 3:
        pass  # Already in the expected layout.
    else:
        # Reorder axes to match the expected layout.
        data = data.permute(0, 3, 1, 2).contiguous()
    return data


def generate_init_points(data):
    """
    Generate an initial Gaussian point cloud.

    Args:
        data: Velocity field as a NumPy array with shape [3, D, H, W].

    Returns:
        ply_data: Point cloud in PLY format.
    """
    from plyfile import PlyData, PlyElement

    C, D, H, W = data.shape

    # Heuristic sample count: 3 * D * H * W / 5.
    num_samples = int(3 * D * H * W / 5)
    total_points = D * H * W
    num_samples = min(num_samples, total_points)

    # Compute velocity magnitude for weighted sampling.
    velocity_magnitude = np.sqrt((data ** 2).sum(axis=0))  # [D, H, W]

    probabilities = velocity_magnitude.flatten()
    probabilities = probabilities / (probabilities.sum() + 1e-8)

    # Weighted random sampling.
    indices = np.random.choice(
        total_points,
        size=num_samples,
        replace=False,
        p=probabilities
    )

    # Convert flattened indices back to 3D coordinates.
    z = indices // (H * W)
    x = (indices % (H * W)) // W
    y = indices % W

    # Gaussian centers, using the grid corner as the origin.
    pos_x = x.astype(np.float32) - H * 0.5 + np.random.randn(len(x)) * 0.5
    pos_y = y.astype(np.float32) - W * 0.5 + np.random.randn(len(y)) * 0.5
    pos_z = z.astype(np.float32) - D * 0.5 + np.random.randn(len(z)) * 0.5

    dtype_full = [
        (attribute, "f4")
        for attribute in [
            "x", "y", "z",
            "vx", "vy", "vz",
            "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3",
        ]
    ]

    position = np.stack((pos_x, pos_y, pos_z), axis=1)
    features = data[:, z, x, y].transpose(1, 0)  # [N, 3]

    scale_rot = np.zeros((len(x), 7))
    # Use a larger initial scale after removing SC.
    # Previous setting: -5.2158 -> softplus -> 0.005 -> *128 -> 0.64
    # Current setting: initialize to a more reasonable physical scale (~1.0).
    scale_rot[:, :3] = 0.0  # log(1) = 0, softplus(0) ≈ 0.69
    scale_rot[:, 3] = 1.0  # Quaternion w component.

    elements = list(map(tuple, np.concatenate((position, features, scale_rot), axis=1)))
    el = PlyElement.describe(np.array(elements, dtype=dtype_full), "gaussians")
    return PlyData([el])


def generate_cameras(np_data):
    """
    Generate slice cameras for multiple viewing directions.

    Args:
        np_data: Velocity field as a NumPy array with shape [3, D, H, W].

    Returns:
        xoy_cam_infos: List of XY slice cameras.
        supplement_cam_infos: List of YZ and XZ slice cameras.
    """
    C, D, H, W = np_data.shape
    xoy_cam_infos = []
    supplement_cam_infos = []

    # XY slices along the Z axis.
    R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    T = np.array([0, 0, D*0.5])
    for view_depth in range(D):
        gt_image = np_data[:, view_depth, :, :]  # [3, H, W]
        cam_info = CameraInfo(
            R=R, T=T,
            view_depth=view_depth,
            gt_image=gt_image,
            channel=C,
            width=W,
            height=H
        )
        xoy_cam_infos.append(cam_info)

    # YZ slices along the X axis.
    for view_depth in range(H):
        R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        T = np.array([0, 0, H*0.5])
        gt_image = np_data[:, :, view_depth, :]  # [3, D, W]
        cam_info = CameraInfo(
            R=R, T=T,
            view_depth=view_depth,
            gt_image=gt_image,
            channel=C,
            width=gt_image.shape[2],
            height=gt_image.shape[1],
        )
        supplement_cam_infos.append(cam_info)

    # XZ slices along the Y axis.
    for view_depth in range(W):
        R = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        T = np.array([0, 0, W*0.5])
        gt_image = np_data[:, :, :, view_depth]  # [3, D, H]
        cam_info = CameraInfo(
            R=R, T=T,
            view_depth=view_depth,
            gt_image=gt_image,
            channel=C,
            width=gt_image.shape[2],
            height=gt_image.shape[1],
        )
        supplement_cam_infos.append(cam_info)

    return xoy_cam_infos, supplement_cam_infos


# ========== Training ==========

def reconstruct_single_frame(hdf_file, args, opt, mission_idx, total_frames, pid2device, log_queue):
    """
    Reconstruct a single-frame fluid sample.
    """
    import torch

    pid = os.getpid()
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        num_gpus = len(visible_devices.split(","))
    else:
        num_gpus = torch.cuda.device_count()

    if pid in pid2device:
        device_idx = pid2device[pid]
    else:
        device_idx = mission_idx % num_gpus
        pid2device[pid] = device_idx
    device = torch.device(f"cuda:{device_idx}")
    torch.cuda.set_device(device)

    # Configure the worker logger.
    logger = logging.getLogger(f"worker_{pid}")
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        handler = logging.handlers.QueueHandler(log_queue)
        logger.addHandler(handler)

    hdf_path = Path(hdf_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output files.
    output_file = output_dir / f"{hdf_path.stem}_gaussians.hdf5"

    start_time = time.time()
    print(f"[{mission_idx+1}/{total_frames}] Processing {hdf_path.name} on cuda:{device_idx}")
    logger.info(f"[{mission_idx+1}/{total_frames}] Processing {hdf_path.name} on cuda:{device_idx}")

    # 1. Load data.
    data = load_single_frame(hdf_path, device)
    np_data = data.cpu().numpy()

    # 2. Build the initial point cloud and cameras.
    ply_data = generate_init_points(np_data)
    xoy_cam_infos, supplement_cam_infos = generate_cameras(np_data)

    scene_info = SceneInfo(point_cloud=ply_data, xoy=xoy_cam_infos, supplement=supplement_cam_infos)
    scene_info.set_xoy(cameraList_from_camInfos(scene_info.xoy))
    scene_info.set_supplement(cameraList_from_camInfos(scene_info.supplement))

    # 3. Initialize the Gaussian model.
    gaussians = GaussianModel(
        device,
        fourier_mod_order=args.fourier_mod_order,
        fourier_coupled_order=args.fourier_coupled_order
    )
    gaussians.init_points(scene_info.point_cloud)
    gaussians.training_setup(opt, max_iteration=opt.frame_iterations)
    # Loss functions.
    ssim = SSIM(channel=3).to(device)
    get_energy = Energy(window_size=11, sigm=torch.e, device=device)

    # 4. Training loop.
    xoy_viewpoint_stack = None
    supplement_viewpoint_stack = None

    for iteration in range(1, opt.frame_iterations + 1):
        # Sample a camera.
        if not xoy_viewpoint_stack:
            xoy_viewpoint_stack = scene_info.xoy.copy()
        if not supplement_viewpoint_stack:
            supplement_viewpoint_stack = scene_info.supplement.copy()

        # Use XY cameras most of the time (2/3), with supplemental views occasionally (1/3).
        if iteration % 3 != 1:
            # Use a supplemental camera every third iteration (YZ/XZ slices provide sz gradients).
            viewpoint_cam = supplement_viewpoint_stack.pop(randint(0, len(supplement_viewpoint_stack) - 1))
        else:
            # Otherwise use XY cameras for the main supervision signal.
            viewpoint_cam = xoy_viewpoint_stack.pop(randint(0, len(xoy_viewpoint_stack) - 1))

        # Three-stage loss schedule.
        if iteration <= opt.energy_iterations:
            # Energy stage.
            image = render(viewpoint_cam, gaussians, None, energy=True)["render"]
            gt_image = viewpoint_cam.gt_image.to(device)
            gt_image = get_energy(torch.sqrt((gt_image**2).sum(axis=0)).unsqueeze(dim=0))
            loss = l2_loss(image, gt_image)

        elif iteration <= 10000:
            # L2 stage.
            render_output = render(viewpoint_cam, gaussians, None, energy=False)
            image = render_output["render"]
            gt_image = viewpoint_cam.gt_image.to(device)
            loss = l2_loss(image, gt_image)

        else:
            # Fourier modulation stage.
            image = render_with_fourier_modulation(
                viewpoint_cam, gaussians, None,
                energy=False, iteration=iteration
            )["render"]
            gt_image = viewpoint_cam.gt_image.to(device)
            loss = l2_loss(image, gt_image)

        loss.backward()

        # Optimizer step.
        with torch.no_grad():
            g_lr = gaussians.update_learning_rate(
                iteration,
                energy=True if iteration < opt.energy_iterations else False
            )

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        # Print progress.
        if iteration % 500 == 0:
            print(f"  [{hdf_path.name}] Iter {iteration}/{opt.frame_iterations}, Loss: {loss.item():.6f}")

        if iteration % 1000 == 0:
            logger.info(f"[{hdf_path.name}] Iter {iteration}, Loss: {loss.item():.6f}")

    comp_kwargs = {"compression": "gzip", "compression_opts": 4}

    # 5. Save results.
    with h5py.File(output_file, "w") as f:
        f.attrs["source_file"] = str(hdf_path)
        f.attrs["iterations"] = opt.frame_iterations
        f.attrs["volume_shape"] = np_data.shape  # Save the volume shape as (C, D, H, W).
        f.attrs["fourier_mod_order"] = gaussians.fourier_mod_order
        f.attrs["fourier_coupled_order"] = gaussians.fourier_coupled_order

        # Gaussian tensor layout: [xyz(3), intensity(3), scale(3), rot(4), fourier_mod(K*4), fourier_coupled(K)]
        data = gaussians.get_gaussians_as_tensor()
        f.create_dataset("gaussians", data=data.detach().cpu().numpy(), **comp_kwargs)

    # 6. Render and save the reconstructed volume [3, D, H, W].
    recon_file = output_dir / f"{hdf_path.stem}_recon.hdf5"
    print(f"  [{hdf_path.name}] Rendering reconstructed volume...")
    logger.info(f"[{hdf_path.name}] Rendering reconstructed volume")

    render_fn = render_with_fourier_modulation if args.fourier_mod_order and args.fourier_mod_order > 0 else render
    render_kwargs = {"energy": False}
    if render_fn is render_with_fourier_modulation:
        # Avoid debug file writes in render_with_fourier_modulation
        render_kwargs["iteration"] = 1

    with torch.no_grad():
        recon_stack = None
        for idx, cam in enumerate(scene_info.xoy):
            render_out = render_fn(cam, gaussians, None, **render_kwargs)["render"]
            render_cpu = render_out.detach().cpu()
            if recon_stack is None:
                _, H, W = render_cpu.shape
                recon_stack = torch.empty(
                    (len(scene_info.xoy), render_cpu.shape[0], H, W),
                    dtype=render_cpu.dtype
                )
            recon_stack[idx] = render_cpu

        recon = recon_stack.permute(1, 0, 2, 3).contiguous().numpy()

    with h5py.File(recon_file, "w") as f:
        f.attrs["source_file"] = str(hdf_path)
        f.attrs["gaussians_file"] = str(output_file)
        f.attrs["iterations"] = opt.frame_iterations
        f.create_dataset("reconstruction", data=recon, **comp_kwargs)

    elapsed = time.time() - start_time
    print(
        f"[{mission_idx+1}/{total_frames}] Done {hdf_path.name}. Time: {elapsed/60:.2f}min. "
        f"Saved to {output_file} and {recon_file}"
    )
    logger.info(f"[{mission_idx+1}/{total_frames}] Done {hdf_path.name}. Time: {elapsed/60:.2f}min")

    torch.cuda.empty_cache()


def listener(log_queue, log_file):
    """Log listener process."""
    root = logging.getLogger()
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)

    while True:
        try:
            if not log_queue.empty():
                record = log_queue.get()
                if record is None:
                    break
                logger = logging.getLogger(record.name)
                logger.handle(record)
        except Exception:
            import traceback
            traceback.print_exc()


def main(args):
    """Main entry: scan input files and process them in parallel."""
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scan all HDF5 files.
    hdf_files = sorted(data_dir.glob(args.pattern))
    total_frames = len(hdf_files)

    if total_frames == 0:
        print(f"No files found matching pattern '{args.pattern}' in {data_dir}")
        return

    print(f"Found {total_frames} frames to process")
    print(f"Output directory: {output_dir}")

    # Save the run configuration.
    with open(output_dir / "cfg_args.txt", "w") as f:
        f.write(str(Namespace(**vars(args))))

    # Set up multiprocessing.
    manager = multiprocessing.Manager()
    pid2device = manager.dict()
    log_queue = manager.Queue()
    log_file = output_dir / "training.log"

    # Start the log listener.
    listener_process = multiprocessing.Process(target=listener, args=(log_queue, log_file))
    listener_process.start()

    # Build optimization parameters.
    opt = OptimizationParams(ArgumentParser()).extract(args)

    if args.debug:
        # Debug mode: single-process sequential execution.
        args.num_processes = 1
        args.frame_iterations = 1500
        for idx, hdf_file in enumerate(hdf_files[:2]):  # Process only the first two files.
            reconstruct_single_frame(
                hdf_file, args, opt, idx, total_frames, pid2device, log_queue
            )
    else:
        # Normal mode: multiprocessing.
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            for idx, hdf_file in enumerate(hdf_files):
                pool.apply_async(
                    reconstruct_single_frame,
                    (hdf_file, args, opt, idx, total_frames, pid2device, log_queue),
                    error_callback=lambda e: print(f"Error: {e}")
                )
            pool.close()
            pool.join()

    # Shut down logging.
    log_queue.put(None)
    listener_process.join()

    print(f"\nAll {total_frames} frames processed. Results saved to {output_dir}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = ArgumentParser(description="Single-frame fluid velocity field reconstruction")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)

    # Input / output.
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing HDF5 files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--pattern", type=str, default="*.hdf5",
                        help="Glob pattern for input files (default: *.hdf5)")

    # Training parameters.
    parser.add_argument("--num_processes", type=int, default=1,
                        help="Number of parallel processes")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode (single process, fewer iterations)")
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")

    # Fourier modulation parameters.
    parser.add_argument("--fourier_mod_order", type=int, default=0,
                        help="Separable Fourier basis order (0 to disable)")
    parser.add_argument("--fourier_coupled_order", type=int, default=0,
                        help="Coupled Fourier basis order (0 to disable)")

    args = parser.parse_args()

    # Initialization.
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    main(args)

    print("\nFluid reconstruction complete.")
