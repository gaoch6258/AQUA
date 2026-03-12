import torch
import numpy as np
from utils.general_utils import get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.general_utils import strip_symmetric, build_scaling_rotation


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.intensity_activation = torch.relu

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.rotation_activation = torch.nn.functional.normalize

        self.covariance_activation = build_covariance_from_scaling_rotation

    def __init__(self, device, fourier_mod_order=0, fourier_coupled_order=0):
        """
        Args:
            device: Target device.
            fourier_mod_order: Order of the separable Fourier basis. Use 0 to disable.
            fourier_coupled_order: Order of the coupled Fourier basis. Use 0 to disable.
        """
        self._xyz = torch.empty(0)
        self._intensity = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.freeze_xyz = True
        self.setup_functions()
        self._device = device

        # Separable Fourier basis: [N, K, 4] -> [cos_kx, sin_kx, cos_ky, sin_ky]
        self.fourier_mod_order = fourier_mod_order
        self._fourier_mod_coeffs = torch.empty(0)

        # Coupled Fourier basis: [N, K_coupled, 1] -> cos(kx) * cos(ky)
        self.fourier_coupled_order = max(0, fourier_coupled_order)
        self._fourier_coupled_coeffs = torch.empty(0)

    def capture(self):
        return (
            self._xyz,
            self._intensity,
            self._scaling,
            self._rotation,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
        )

    def restore(self, model_args, training_args):
        (
            self._xyz,
            self._intensity,
            self._scaling,
            self._rotation,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def device(self):
        return self._device

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_xy(self):
        return self._xyz[..., :2]

    @property
    def get_intensity(self):
        return self._intensity

    @property
    def get_fourier_mod_coeffs(self):
        """Return separable Fourier coefficients with shape [N, K, 4]."""
        return self._fourier_mod_coeffs

    @property
    def get_fourier_coupled_coeffs(self):
        """Return coupled Fourier coefficients with shape [N, K_coupled, 1]."""
        return self._fourier_coupled_coeffs

    @torch.no_grad()
    def updata_gaussians(self, delta_gaussians):
        self._xyz = self._xyz + delta_gaussians.get_dxyz
        self._intensity = self._intensity + delta_gaussians.get_dintensity
        self._scaling = self._scaling + delta_gaussians.get_dscaling
        self._rotation = self._rotation + delta_gaussians.get_drotation
        if self.fourier_mod_order > 0 and hasattr(delta_gaussians, '_dfourier_mod_coeffs'):
            self._fourier_mod_coeffs = self._fourier_mod_coeffs + delta_gaussians.get_dfourier_mod_coeffs
            self._fourier_coupled_coeffs = self._fourier_coupled_coeffs + delta_gaussians.get_dfourier_coupled_coeffs

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def init_points(self, plydata):
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        # Detect the data type automatically: radar data vs fluid velocity fields.
        property_names = [p.name for p in plydata.elements[0].properties]

        if "vx" in property_names:
            # Fluid velocity field data (3 channels).
            intensity = np.stack(
                [
                    np.asarray(plydata.elements[0]["vx"]),
                    np.asarray(plydata.elements[0]["vy"]),
                    np.asarray(plydata.elements[0]["vz"]),
                ],
                axis=1,
            )
        else:
            # Radar data (6 channels).
            intensity = np.stack(
                [
                    np.asarray(plydata.elements[0]["Z_H"]),
                    np.asarray(plydata.elements[0]["SW"]),
                    np.asarray(plydata.elements[0]["AzShr"]),
                    np.asarray(plydata.elements[0]["Div"]),
                    np.asarray(plydata.elements[0]["Z_DR"]),
                    np.asarray(plydata.elements[0]["K_DP"]),
                ],
                axis=1,
            )

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=self._device).requires_grad_(True))
        self._intensity = nn.Parameter(
            torch.tensor(intensity, dtype=torch.float, device=self._device).requires_grad_(True)
        )
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=self._device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=self._device).requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self._device)

        # Initialize per-Gaussian Fourier modulation coefficients.
        if self.fourier_mod_order > 0:
            N = xyz.shape[0]
            K = self.fourier_mod_order
            K_coupled = self.fourier_coupled_order

            # Separable basis coefficients: [N, K, 4]
            # Use small random values to avoid perfect symmetry.
            self._fourier_mod_coeffs = nn.Parameter(
                torch.randn(N, K, 4, dtype=torch.float, device=self._device) * 0.001
            )

            # Coupled basis coefficients: [N, K_coupled, 1]
            self._fourier_coupled_coeffs = nn.Parameter(
                torch.randn(N, K_coupled, 1, dtype=torch.float, device=self._device) * 0.001
            )

    def training_setup(self, training_args, max_iteration):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self._device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self._device)

        l = [
            {"params": [self._xyz], "name": "xyz"},
            {"params": [self._intensity], "name": "intensity"},
            {"params": [self._scaling], "name": "scaling"},
            {"params": [self._rotation], "name": "rotation"},
        ]

        # Add per-Gaussian Fourier modulation coefficients to the optimizer.
        if self.fourier_mod_order > 0:
            l.append({"params": [self._fourier_mod_coeffs], "name": "fourier_mod_coeffs"})
            l.append({"params": [self._fourier_coupled_coeffs], "name": "fourier_coupled_coeffs"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.scheduler_args = get_expon_lr_func(
            lr_init=training_args.lr_init,
            lr_final=training_args.lr_final,
            lr_delay_mult=training_args.lr_delay_mult,
            lr_delay_steps=training_args.lr_delay_steps,
            max_steps=15000,
        )
        self.scheduler_args1 = get_expon_lr_func(
            lr_init=training_args.lr_final/2,
            lr_final=training_args.lr_final/10,
            lr_delay_mult=training_args.lr_delay_mult,
            lr_delay_steps=training_args.lr_delay_steps,
            max_steps=max_iteration,
        )

    def update_learning_rate(self, iteration, energy=False):
        """Learning rate scheduling per step with progressive Fourier introduction"""
        
        lr = self.scheduler_args(iteration) if iteration < 15000 else self.scheduler_args1(iteration)
        if energy:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "xyz":
                    param_group["lr"] = lr * 5
                else:
                    param_group["lr"] = 0
        else:
            # Progressive three-stage learning-rate schedule.
            if iteration <= 5000:
                # Stage 1: optimize only base parameters to build a stable representation.
                for param_group in self.optimizer.param_groups:
                    if param_group["name"] in ["scaling", "rotation"]:
                        param_group["lr"] = lr * 2.5
                    elif param_group["name"] == "intensity":
                        param_group["lr"] = lr * 2.5
                    elif param_group["name"] in ["fourier_mod_coeffs", "fourier_coupled_coeffs"]:
                        param_group["lr"] = 0  # Temporarily freeze Fourier coefficients.
                    else:
                        param_group["lr"] = lr
            elif iteration <= 10000:
                # Stage 2: introduce Fourier modulation gradually while reducing base learning rates.
                alpha = (iteration - 5000) / 5000  # Smoothly transition from 0 to 1.
                for param_group in self.optimizer.param_groups:
                    if param_group["name"] in ["scaling", "rotation"]:
                        param_group["lr"] = lr * 2.5 * (1 - 0.5 * alpha)
                    elif param_group["name"] == "intensity":
                        param_group["lr"] = lr * 2.5 * (1 - 0.5 * alpha)
                    elif param_group["name"] == "fourier_mod_coeffs":
                        # Use a moderate learning rate for separable basis terms.
                        param_group["lr"] = lr * 0.5 * alpha
                    elif param_group["name"] == "fourier_coupled_coeffs":
                        # Use a smaller learning rate for coupled basis terms, which are more sensitive.
                        param_group["lr"] = lr * 0.25 * alpha
                    else:
                        param_group["lr"] = lr
            else:
                # Stage 3: jointly fine-tune all parameters.
                for param_group in self.optimizer.param_groups:
                    if param_group["name"] in ["scaling", "rotation"]:
                        param_group["lr"] = lr * 1.25
                    elif param_group["name"] == "intensity":
                        param_group["lr"] = lr * 1.25
                    elif param_group["name"] == "fourier_mod_coeffs":
                        param_group["lr"] = lr * 5
                    elif param_group["name"] == "fourier_coupled_coeffs":
                        param_group["lr"] = lr * 2.5
                    else:
                        param_group["lr"] = lr
        return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "intensity"]
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def get_gaussians_as_tensor(self):
        xyz = self.get_xyz              # [N, 3]
        intensity = self.get_intensity  # [N, 3]
        scale = self.get_scaling        # [N, 3]
        rot = self.get_rotation         # [N, 4]

        parts = [xyz, intensity, scale, rot]

        # Flatten Fourier coefficients before concatenation.
        if self.fourier_mod_order > 0:
            # [N, K, 4] -> [N, K*4]
            fourier_mod = self.get_fourier_mod_coeffs.flatten(1)
            parts.append(fourier_mod)
        if self.fourier_coupled_order > 0:
            # [N, K, 1] -> [N, K]
            fourier_coupled = self.get_fourier_coupled_coeffs.flatten(1)
            parts.append(fourier_coupled)

        return torch.cat(parts, dim=-1)


class DeltaGaussianModel:
    def __init__(self, gaussian_model, inverse_d_gaussian=None, indices=None, max_points=None):
        self.fourier_mod_order = gaussian_model.fourier_mod_order
        self.fourier_coupled_order = gaussian_model.fourier_coupled_order

        if inverse_d_gaussian is None:
            self._dxyz = nn.Parameter(torch.zeros_like(gaussian_model.get_xyz).requires_grad_(True))
        else:
            N = len(indices)
            dxyz = torch.cat(
                [
                    -inverse_d_gaussian.get_dxyz[indices],
                    torch.zeros(
                        (max_points - N, 3), dtype=gaussian_model.get_xyz.dtype, device=gaussian_model.get_xyz.device
                    ),
                ],
                dim=0,
            )
            self._dxyz = nn.Parameter(dxyz.requires_grad_(True))

        self._dintensity = nn.Parameter(torch.zeros_like(gaussian_model.get_intensity).requires_grad_(True))
        self._dscaling = nn.Parameter(torch.zeros_like(gaussian_model.get_scaling).requires_grad_(True))
        self._drotation = nn.Parameter(torch.zeros_like(gaussian_model.get_rotation).requires_grad_(True))

        # Delta terms for per-Gaussian Fourier modulation coefficients.
        if self.fourier_mod_order > 0:
            self._dfourier_mod_coeffs = nn.Parameter(
                torch.zeros_like(gaussian_model.get_fourier_mod_coeffs).requires_grad_(True)
            )
            self._dfourier_coupled_coeffs = nn.Parameter(
                torch.zeros_like(gaussian_model.get_fourier_coupled_coeffs).requires_grad_(True)
            )

        self.scaling_activation = torch.exp
        self.optimizer = None
        self._device = gaussian_model.device

    def reset(self, inverse_d_gaussian=None, indices=None, max_points=None):
        if inverse_d_gaussian is not None:
            N = len(indices)
            dxyz = torch.cat(
                [
                    -inverse_d_gaussian.get_dxyz[indices],
                    torch.zeros((max_points - N, 3), dtype=self._dxyz.dtype, device=self._dxyz.device),
                ],
                dim=0,
            )
            self._dxyz.data.copy_(dxyz)
        else:
            self._dxyz.data.zero_()

        self._dintensity.data.zero_()
        self._dscaling.data.zero_()
        self._drotation.data.zero_()

        if self.fourier_mod_order > 0:
            self._dfourier_mod_coeffs.data.zero_()
            self._dfourier_coupled_coeffs.data.zero_()

    @property
    def device(self):
        return self._device

    @property
    def get_dscaling(self):
        return self._dscaling

    @property
    def get_drotation(self):
        return self._drotation

    @property
    def get_dxyz(self):
        return self._dxyz

    @property
    def get_dxy(self):
        return self._dxyz[..., :2]

    @property
    def get_dintensity(self):
        return self._dintensity

    @property
    def get_dfourier_mod_coeffs(self):
        """Return delta values for separable Fourier coefficients with shape [N, K, 4]."""
        return self._dfourier_mod_coeffs

    @property
    def get_dfourier_coupled_coeffs(self):
        """Return delta values for coupled Fourier coefficients with shape [N, K_coupled, 1]."""
        return self._dfourier_coupled_coeffs

    def get_delta_as_tensor(self):
        dxyz = self.get_dxyz
        dintensity = self.get_dintensity
        dscale = self.get_dscaling
        drot = self.get_drotation
        return torch.cat([dxyz, dintensity, dscale, drot], dim=-1).clone().detach()

    def training_setup(self, training_args, max_iteration):

        if self.optimizer is None:
            l = [
                {"params": [self._dxyz], "name": "dxyz"},
                {"params": [self._dintensity], "name": "dintensity"},
                {"params": [self._dscaling], "name": "dscaling"},
                {"params": [self._drotation], "name": "drotation"},
            ]

            # Add delta parameters for per-Gaussian Fourier modulation coefficients.
            if self.fourier_mod_order > 0:
                l.append({"params": [self._dfourier_mod_coeffs], "name": "dfourier_mod_coeffs"})
                l.append({"params": [self._dfourier_coupled_coeffs], "name": "dfourier_coupled_coeffs"})

            self.optimizer = torch.optim.Adam(l, lr=0, eps=1e-15)

        self.scheduler_args = get_expon_lr_func(
            lr_init=training_args.lr_init,
            lr_final=training_args.lr_final,
            lr_delay_mult=training_args.lr_delay_mult,
            lr_delay_steps=training_args.lr_delay_steps,
            max_steps=max_iteration,
        )

    def update_learning_rate(self, iteration, energy=True):
        """Learning rate scheduling per step"""
        lr = self.scheduler_args(iteration)
        if energy:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "dxyz":
                    param_group["lr"] = lr * 5
                else:
                    param_group["lr"] = 0
        else:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "dxyz":
                    param_group["lr"] = lr * 5
                elif param_group["name"] in ["dscaling", "drotation"]:
                    param_group["lr"] = lr * 2.5
                elif param_group["name"] == "dintensity":
                    param_group["lr"] = lr
                elif param_group["name"] == "dfourier_mod_coeffs":
                    # Use a moderate learning rate for separable basis deltas.
                    param_group["lr"] = lr * 0.5
                elif param_group["name"] == "dfourier_coupled_coeffs":
                    # Use a smaller learning rate for coupled basis deltas.
                    param_group["lr"] = lr * 0.25
                else:
                    param_group["lr"] = lr
        return lr
