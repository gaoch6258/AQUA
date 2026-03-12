import os
import sys
from pathlib import Path
from typing import NamedTuple
import random
import h5py
from plyfile import PlyData, PlyElement
import numpy as np
import torch
f=h5py.File("/home/zhouhy/gaoch/Fourier/data3d/xyz.hdf5","r")
x_coordinate=f["x-coordinate"][:]
y_coordinate=f["y-coordinate"][:]
z_coordinate=f["z-coordinate"][:]

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

    def __len__(self):
        return len(self.xoy) + len(self.supplement)

    def set_xoy(self, xoy):
        self.xoy = xoy

    def set_supplement(self, supplement):
        self.supplement = supplement


class FluidDataset:
    def __init__(self, max_init_points, num_vertical_samples, seq_len) -> None:
        """
        Fluid velocity field dataset.

        Args:
            max_init_points: Maximum number of initial points.
            num_vertical_samples: Number of vertical samples used for supplementary views.
            seq_len: Sequence length.
        """
        self.max_init_points = max_init_points
        self.num_vertical_samples = num_vertical_samples
        self.seq_len = seq_len

    def preprocess(self, hdf_path, seq_name, device="cpu"):
        """
        Load and preprocess a fluid velocity field.

        Args:
            hdf_path: Path to the HDF5 file.
            seq_name: Sequence name, e.g. "fluid_t0000".
            device: Target device.

        Returns:
            data: Velocity tensor with shape [3, D, H, W].
        """
        with h5py.File(hdf_path, "r") as f:
            group = f[seq_name]

            keys = ["Vx", "Vy", "Vz"]
            # vx: velocity along x
            # vy: velocity along y
            # vz: velocity along z

            data = np.stack([group[key][:] for key in keys], axis=0)

        # Optional normalization: scale by the global maximum into [-1, 1].
        """v_max = np.abs(data).max()
        if v_max > 0:
            data = data / v_max"""

        data = torch.tensor(data, dtype=torch.float32, device=device)
        data = data.permute(0, 3, 1, 2).contiguous()
        #data = torch.clamp(data, -1, 1).float()

        return data

    def generateFluidSceneInfo(self, hdf_path, seq_name, device="cpu", init_ply=False):
        """
        Generate scene information for a fluid sample.

        Args:
            hdf_path: Path to the HDF5 file.
            seq_name: Sequence name.
            device: Target device.
            init_ply: Whether to initialize the point cloud.

        Returns:
            scene_info: Scene information object.
            frame: Frame data used for flow estimation.
        """
        data = self.preprocess(hdf_path, seq_name, device)

        if init_ply:
            ply_data = self.generateInitPoints(data.cpu().numpy())
        else:
            ply_data = None

        xoy_cam_infos, supplementary_cam_infos = self.generateFluidCameras(data.cpu().numpy())

        scene_info = SceneInfo(point_cloud=ply_data, xoy=xoy_cam_infos, supplement=supplementary_cam_infos)
        # Return the vx channel as the frame for flow estimation if needed.
        return scene_info, data[0] * 255

    def generateInitPoints(self, data):
        """
        Generate an initial Gaussian point cloud for a fluid velocity field.

        Args:
            data: Velocity field as a NumPy array with shape [3, D, H, W].

        Returns:
            ply_data: Point cloud data in PLY format.
        """
        C, D, H, W = data.shape  # [3, 128, 128, 128]

        # Compute velocity magnitude as a scalar field.
        velocity_magnitude = np.sqrt((data ** 2).sum(axis=0))  # [D, H, W]

        # Weighted sampling based on velocity magnitude, with denser sampling in high-speed regions.
        total_points = D * H * W

        # Build sampling probabilities.
        probabilities = velocity_magnitude.flatten()
        probabilities = probabilities / (probabilities.sum() + 1e-8)  # Avoid division by zero.

        # Heuristic sample count: 3 * D * H * W / 20.
        num_samples = int(3 * D * H * W / 20)
        num_samples = min(num_samples, total_points)  # Do not exceed the total number of points.
        indices = np.random.choice(
            total_points,
            size=num_samples,
            replace=False,
            p=probabilities
        )

        # Convert flat indices back to 3D coordinates.
        z = indices // (H * W)
        x = (indices % (H * W)) // W
        y = indices % W

        """# Add noise to avoid poor local minima.
        pos_x = x.astype(np.float32) + np.random.randn(len(x)) * 0.5
        pos_y = y.astype(np.float32) + np.random.randn(len(y)) * 0.5
        pos_z = z.astype(np.float32) + np.random.randn(len(z)) * 0.5"""
        pos_x = x_coordinate[x] + np.random.randn(len(x)) * 0.5 / 128
        pos_y = y_coordinate[y] + np.random.randn(len(y)) * 0.5 / 128
        pos_z = z_coordinate[z] + np.random.randn(len(z)) * 0.5 / 128

        # Define the PLY schema. Three velocity channels replace the original six radar channels.
        dtype_full = [
            (attribute, "f4")
            for attribute in [
                "x",
                "y",
                "z",
                "vx",        # Velocity x component
                "vy",        # Velocity y component
                "vz",        # Velocity z component
                "scale_0",
                "scale_1",
                "scale_2",
                "rot_0",
                "rot_1",
                "rot_2",
                "rot_3",
            ]
        ]

        # Coordinate transform: map [0, D/H/W] indices into a centered coordinate system.
        #position = np.stack((pos_x - H * 0.5 + 0.5, pos_y - W * 0.5 + 0.5, pos_z - D *0.5 + 0.5), axis=1)
        position = np.stack((pos_x,pos_y,pos_z),axis=1)
        # Extract velocity values at the sampled positions.
        features = data[:, z, x, y].transpose(1, 0)  # [N, 3]

        # Initialize scaling and rotation (quaternion).
        scale_rot = np.zeros((len(x), 7))
        scale_rot[:, :3] = -5.2158
        scale_rot[:, 3] = 1.0  # Initialize quaternion w to 1.

        # Concatenate all attributes.
        elements = list(map(tuple, np.concatenate((position, features, scale_rot), axis=1)))
        el = PlyElement.describe(np.array(elements, dtype=dtype_full), "gaussians")
        return PlyData([el])

    def generateFluidSceneInfoWithInverseGaussian(self, hdf_path, seq_name, gaussians_tensor, energy_img, device="cpu"):
        """
        Regenerate scene information with inverse Gaussians during reconstruction.

        Args:
            hdf_path: Path to the HDF5 file.
            seq_name: Sequence name.
            gaussians_tensor: Gaussian tensor with shape [N, 13].
            energy_img: Energy image with shape [D, H, W].
            device: Target device.

        Returns:
            scene_info: Scene information.
            frame: Frame data.
            indices: Resampled indices.
        """
        data = self.preprocess(hdf_path, seq_name, device)

        C, D, H, W = data.shape
        N = gaussians_tensor.shape[0]

        # Compute velocity magnitude.
        velocity_magnitude = torch.sqrt((data ** 2).sum(dim=0))  # [D, H, W]

        # Separate covered and uncovered regions.
        cover_mask = (velocity_magnitude > 0.01) & (energy_img > 0.01)
        uncover_mask = (velocity_magnitude > 0.01) & (energy_img < 0.01)

        # Resample uncovered regions.
        ratio = uncover_mask.sum() / (cover_mask.sum() + uncover_mask.sum() + 1e-8)
        z, x, y = uncover_mask.cpu().numpy().nonzero()
        num_uncover = int(N * ratio) if int(N * ratio) < len(x) else len(x)
        indices_uncover = random.sample(range(len(x)), num_uncover)
        x = x[indices_uncover]
        y = y[indices_uncover]
        z = z[indices_uncover]

        # Add noise.
        pos_x = x.astype(np.float32) + np.random.randn(len(x)) * 0.5
        pos_y = y.astype(np.float32) + np.random.randn(len(y)) * 0.5
        pos_z = z.astype(np.float32) + np.random.randn(len(z)) * 0.5

        dtype_full = [
            (attribute, "f4")
            for attribute in [
                "x", "y", "z",
                "vx", "vy", "vz",
                "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3",
            ]
        ]

        position = np.stack((pos_x - H * 0.5 + 0.5, pos_y - W * 0.5 + 0.5, pos_z - D *0.5 + 0.5), axis=1)
        features = data[:, z, x, y].transpose(1, 0).cpu().numpy()
        scale_rot = np.zeros((len(x), 7))
        scale_rot[:, 3] = 1.0

        uncover_gaussian = np.concatenate((position, features, scale_rot), axis=1)

        # Resample covered regions.
        indices = random.sample(range(N), N - len(x))
        gaussians_tensor = gaussians_tensor[indices]
        cover_gaussian = gaussians_tensor.numpy()

        elements = list(map(tuple, np.concatenate([cover_gaussian, uncover_gaussian], axis=0)))
        el = PlyElement.describe(np.array(elements, dtype=dtype_full), "gaussians")
        ply_data = PlyData([el])

        xoy_cam_infos, supplementary_cam_infos = self.generateFluidCameras(data.cpu().numpy())

        scene_info = SceneInfo(point_cloud=ply_data, xoy=xoy_cam_infos, supplement=supplementary_cam_infos)
        return scene_info, data[0] * 255, indices

    def generateFluidCameras(self, np_data):
        """
        Generate multi-view slice cameras for the fluid field (XY, YZ, and XZ).

        Args:
            np_data: Velocity field as a NumPy array with shape [3, D, H, W].

        Returns:
            xoy_cam_infos: List of XY slice cameras.
            supplementary_cam_infos: List of YZ and XZ slice cameras.
        """
        C, D, H, W = np_data.shape  # [3, 128, 128, 128]
        xoy_cam_infos = []
        supplementary_cam_infos = []

        # ========== 1. XY slices along Z (main views) ==========
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        T = np.array([0, 0, D * 0.5])
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
                # get yoz (vertical slice, fixed x)
        # Select slices with sufficient data along H dimension
        for view_depth in range(H):
            # Camera looks at y-z plane, with x=view_depth as the cutting plane
            # R maps: world_y -> cam_x, world_z -> cam_y, world_x -> cam_z (depth)
            R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
            # T_z makes z_cam = x_index, T_y centers the vertical axis
            T = np.array([0, 0, H * 0.5 ])
            gt_image = np_data[:, :, view_depth, :]  # shape: (C, D, W)
            cam_info = CameraInfo(
                R=R,
                T=T,
                view_depth=view_depth,
                gt_image=gt_image,
                channel=C,
                width=gt_image.shape[2],  # W
                height=gt_image.shape[1],  # D
            )
            supplementary_cam_infos.append(cam_info)

        # get xoz (vertical slice, fixed y)
        # Select slices with sufficient data along W dimension
        for view_depth in range(W):
            # Camera looks at x-z plane, with y=view_depth as the cutting plane
            # R maps: world_x -> cam_x, world_z -> cam_y, world_y -> cam_z (depth)
            R = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
            # T_z makes z_cam = y_index, T_y centers the vertical axis
            T = np.array([0, 0, W * 0.5])
            gt_image = np_data[:, :, :, view_depth]  # shape: (C, D, H)
            cam_info = CameraInfo(
                R=R,
                T=T,
                view_depth=view_depth,
                gt_image=gt_image,
                channel=C,
                width=gt_image.shape[2],  # H
                height=gt_image.shape[1],  # D
            )
            supplementary_cam_infos.append(cam_info)


        """# ========== 2. YZ slices along X (supplementary views) ==========

        for h in range(H):
            gt_image = np_data[:, :, h, :]  # [3, D, W]
            # Rotation matrix for the YZ plane.
            R = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
            T = np.array([h - H * 0.5 + 0.5, 0, 0])
            cam_info = CameraInfo(
                R=R, T=T,
                view_depth=0,  # YZ slices do not use view_depth.
                gt_image=gt_image,
                channel=C,
                width=W,   # Width of the YZ slice equals the original W.
                height=D   # Height of the YZ slice equals the original D.
            )
            supplementary_cam_infos.append(cam_info)

        # ========== 3. XZ slices along Y (supplementary views) ==========

        for w in range(W):
            gt_image = np_data[:, :, :, w]  # [3, D, H]
            # Rotation matrix for the XZ plane.
            R = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
            T = np.array([0, w - W * 0.5 + 0.5, 0])
            cam_info = CameraInfo(
                R=R, T=T,
                view_depth=0,
                gt_image=gt_image,
                channel=C,
                width=H,   # Width of the XZ slice equals the original H.
                height=D   # Height of the XZ slice equals the original D.
            )
            #supplementary_cam_infos.append(cam_info)"""

        return xoy_cam_infos, supplementary_cam_infos


if __name__ == "__main__":
    # Simple smoke test.
    dataset = FluidDataset(max_init_points=2**14 * 3, num_vertical_samples=25, seq_len=25)

    # Assume you already have an HDF5 file with fluid data.
    # scene_info, frame = dataset.generateFluidSceneInfo("your_fluid_data.hdf5", "fluid_t0000")
    # print(f"Point cloud: {scene_info.point_cloud}")
    # print(f"XY cameras: {len(scene_info.xoy)}")
    # print(f"Supplementary cameras: {len(scene_info.supplement)}")
    print("FluidDataset created successfully!")
