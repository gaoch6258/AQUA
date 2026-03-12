import torch
import numpy as np
import torch.nn.functional as F
from diff_gaussian_rasterization_radar import GaussianRasterizationSettings, GaussianRasterizer
from radar_gs.gaussian_model import GaussianModel

# SC = 128  # Removed: physical coordinates are used directly, so no scale factor is needed.

def render(viewpoint_camera, pc: GaussianModel, d_pc=None, energy=False, scaling_modifier=1.0):
    """
    Render the specific cross-section of the 3D radar.
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    if energy:
        # For energy calculation, we render the current spacial distribution of Gaussians in a single channel
        raster_settings = GaussianRasterizationSettings(
            image_channel=1,
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.to(pc.device),
            viewdepth=viewpoint_camera.view_depth,
            prefiltered=False,
            debug=False
        )
    else:
        raster_settings = GaussianRasterizationSettings(
            image_channel=(viewpoint_camera.image_channel),
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.to(pc.device),
            viewdepth=viewpoint_camera.view_depth,
            prefiltered=False,
            debug=False
        )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Get the means, intensity, scales and rotations of the Gaussians.
    if d_pc is None:
        means3D = pc.get_xyz  # Use physical coordinates directly without additional scaling.
        intensity = pc.get_intensity
        means2D = screenspace_points
        scales = F.softplus(pc.get_scaling)  # No SC scaling.
        rotations = pc.get_rotation
    else:
        # Add the deformations
        new_xyz = pc.get_xyz.detach() + d_pc.get_dxyz
        means3D = new_xyz  # Use physical coordinates directly.
        intensity = pc.get_intensity.detach() + d_pc.get_dintensity
        means2D = screenspace_points
        scales = F.softplus(pc.get_scaling.detach() + d_pc.get_dscaling)  # No SC scaling.
        rotations = F.normalize(pc.get_rotation.detach() + d_pc.get_drotation)

    if energy:
        # Assume each Gaussian is of the same shape and intensity
        intensity = torch.ones((intensity.shape[0], 1), device=means3D.device, dtype=means3D.dtype) * 0.2
        scales = torch.ones_like(scales)
        rotations = torch.zeros_like(rotations)
        rotations[..., 0] += 1

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_radar, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        intensity=intensity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_radar,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}


def render_with_fourier_modulation(viewpoint_camera, pc: GaussianModel, d_pc=None, energy=False,
                                    scaling_modifier=1.0, fourier_order=None,iteration=0):
    """
    Per-Gaussian Fourier-modulated rendering for XY slices.

    Core identity:
    pixel(px,py) = Σᵢ intensity_i × α_i(px,py) × F_i(px,py)

    where F_i(px,py) = 1 + Σ_k [a_ik·cos(kω·px) + b_ik·sin(kω·px) + c_ik·cos(kω·py) + d_ik·sin(kω·py)]

    Expanded form:
    = Σᵢ intensity_i × α_i                            # Term 0: base rendering
    + Σᵢ (intensity_i × a_i1) × α_i × cos(ω·px)      # Term 1: per-Gaussian modulation
    + Σᵢ (intensity_i × b_i1) × α_i × sin(ω·px)      # Term 2
    + ...

    Each term of the form "Σᵢ (intensity_i × coeff_i) × α_i" is a standard Gaussian render.
    The only change is to replace intensity with intensity_i × coeff_i, then multiply by the pixel-wise Fourier basis.
    """
    # Fall back to standard rendering for energy mode or when Fourier modulation is disabled.
    if energy or pc.fourier_mod_order == 0:
        return render(viewpoint_camera, pc, d_pc, energy, scaling_modifier)

    # Determine the Fourier order.
    K = fourier_order if fourier_order is not None else pc.fourier_mod_order
    K_coupled = pc.fourier_coupled_order

    # ========== Step 0: Prepare base parameters ==========
    device = pc.device

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Configure rasterization.
    raster_settings = GaussianRasterizationSettings(
        image_channel=viewpoint_camera.image_channel,
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.to(device),
        viewdepth=viewpoint_camera.view_depth,
        prefiltered=False,
        debug=False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Load Gaussian parameters in physical coordinates.
    if d_pc is None:
        means3D = pc.get_xyz  # No SC scaling.
        base_intensity = pc.get_intensity  # [N, 3]
        scales = F.softplus(pc.get_scaling)  # No SC scaling.
        rotations = pc.get_rotation
        # Fourier modulation coefficients.
        fourier_coeffs = pc.get_fourier_mod_coeffs  # [N, K, 4]
        fourier_coupled_coeffs = pc.get_fourier_coupled_coeffs  # [N, K_coupled, 1]
    else:
        new_xyz = pc.get_xyz.detach() + d_pc.get_dxyz
        means3D = new_xyz  # No SC scaling.
        base_intensity = pc.get_intensity.detach() + d_pc.get_dintensity  # [N, 3]
        scales = F.softplus(pc.get_scaling.detach() + d_pc.get_dscaling)  # No SC scaling.
        rotations = F.normalize(pc.get_rotation.detach() + d_pc.get_drotation)
        # Fourier modulation coefficients (base + delta).
        if d_pc.fourier_mod_order > 0:
            fourier_coeffs = pc.get_fourier_mod_coeffs + d_pc.get_dfourier_mod_coeffs
            fourier_coupled_coeffs = pc.get_fourier_coupled_coeffs + d_pc.get_dfourier_coupled_coeffs
        else:
            fourier_coeffs = pc.get_fourier_mod_coeffs
            fourier_coupled_coeffs = pc.get_fourier_coupled_coeffs

    # ========== Step 1: Build the pixel coordinate grid for the XY slice ==========
    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)

    # Use normalized grid coordinates.
    x_coords = torch.arange(W, device=device, dtype=torch.float32)/W  # [W]
    y_coords = torch.arange(H, device=device, dtype=torch.float32)/H  # [H]

    # Create a grid with shape [H, W].
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [H, W]

    # ========== Step 2: Base render (Term 0: the F=1 component) ==========
    base_render, radii = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        intensity=base_intensity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )  # [C, H, W]

    result = base_render.clone()

    # ========== Step 3: Render separable Fourier modulation terms ==========
    for k in range(1, K + 1):
        omega = 2 * torch.pi * k*100 # Angular frequency for the k-th order.

        # Extract k-th order coefficients [N, 4] -> [cos_x, sin_x, cos_y, sin_y].
        coeffs_k = fourier_coeffs[:, k-1, :]  # [N, 4]
        coeff_cos_x = coeffs_k[:, 0:1]  # [N, 1````````]
        coeff_sin_x = coeffs_k[:, 1:2]  # [N, 1]
        coeff_cos_y = coeffs_k[:, 2:3]  # [N, 1]
        coeff_sin_y = coeffs_k[:, 3:4]  # [N, 1]

        # Render the four separable components.
        # base_intensity has shape [N, 3], coeff has shape [N, 1], so broadcasting yields [N, 3].

        render_cos_x, _ = rasterizer(
            means3D=means3D,
            means2D=screenspace_points,
            intensity=base_intensity * coeff_cos_x,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )

        render_sin_x, _ = rasterizer(
            means3D=means3D,
            means2D=screenspace_points,
            intensity=base_intensity * coeff_sin_x,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )

        render_cos_y, _ = rasterizer(
            means3D=means3D,
            means2D=screenspace_points,
            intensity=base_intensity * coeff_cos_y,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )

        render_sin_y, _ = rasterizer(
            means3D=means3D,
            means2D=screenspace_points,
            intensity=base_intensity * coeff_sin_y,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )

        # Evaluate the pixel-wise Fourier basis functions.
        # Expand grid_x and grid_y from [H, W] to [1, H, W] for broadcasting with [C, H, W].
        cos_kx = torch.cos(omega * grid_x).unsqueeze(0)  # [1, H, W]
        sin_kx = torch.sin(omega * grid_x).unsqueeze(0)  # [1, H, W]
        cos_ky = torch.cos(omega * grid_y).unsqueeze(0)  # [1, H, W]
        sin_ky = torch.sin(omega * grid_y).unsqueeze(0)  # [1, H, W]

        # Accumulate Fourier modulation terms.
        result = result + render_cos_x * cos_kx
        result = result + render_sin_x * sin_kx
        result = result + render_cos_y * cos_ky
        result = result + render_sin_y * sin_ky

    # ========== Step 4: Render coupled Fourier terms ==========
    for k in range(1, K_coupled + 1):
        omega = 2 * torch.pi * k*100

        # Extract the k-th coupled coefficient [N, 1].
        coeff_xy_k = fourier_coupled_coeffs[:, k-1, 0:1]  # [N, 1]

        # Render the coupled term.
        render_xy, _ = rasterizer(
            means3D=means3D,
            means2D=screenspace_points,
            intensity=base_intensity * coeff_xy_k,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )

        # Evaluate the coupled basis cos(kω·px) × cos(kω·py).
        cos_kx = torch.cos(omega * grid_x)  # [H, W]
        cos_ky = torch.cos(omega * grid_y)  # [H, W]
        basis_xy = (cos_kx * cos_ky).unsqueeze(0)  # [1, H, W]

        # Accumulate the coupled contribution.
        result = result + render_xy * basis_xy

    return {
        "render": result,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii
    }
