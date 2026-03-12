

from __future__ import annotations  

import json
import sys
from pathlib import Path
from typing import NamedTuple, Optional, Union
import numpy as np
import torch
from PIL import Image


FOURIER_PROJECT_PATH = str(Path(__file__).resolve().parent / "gs")
if FOURIER_PROJECT_PATH not in sys.path:
    sys.path.insert(0, FOURIER_PROJECT_PATH)


_3DGS_AVAILABLE = False

try:
    import h5py
    from radar_gs.gaussian_model import GaussianModel
    from radar_gs.render import render as gs_render
    from radar_gs.render import render_with_fourier_modulation
    from radar_gs.camera import Camera
    _3DGS_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入3DGS渲染模块: {e}")
    print(f"请确保 {FOURIER_PROJECT_PATH} 路径正确且依赖已安装")

    GaussianModel = None
    Camera = None
    gs_render = None
    render_with_fourier_modulation = None


class CameraInfo(NamedTuple):

    R: np.ndarray
    T: np.ndarray
    view_depth: float
    gt_image: np.ndarray
    channel: int
    width: int
    height: int


class RealFluidRenderer:


    def __init__(
        self,
        gaussians_hdf5_path: str,
        qa_json_path: Optional[str] = None,
        device: str = "cuda",
        use_fourier: bool = True,
        fourier_mod_order: int = 2,
        fourier_coupled_order: int = 1,
        verbose: bool = True,
    ):


        self.device = torch.device(device)
        self.use_fourier = use_fourier
        self.verbose = bool(verbose)


        if not _3DGS_AVAILABLE:
            raise RuntimeError(
                "3DGS渲染模块不可用。请检查:\n"
                f"1. Fourier项目路径是否正确: {FOURIER_PROJECT_PATH}\n"
                "2. 是否安装了所需依赖 (h5py, diff_gaussian_rasterization_radar等)"
            )


        if self.verbose:
            print(f"加载Gaussian模型: {gaussians_hdf5_path}")
        (
            self.gaussians_data,
            self.sequence_shape,
            self.fourier_mod_order,
            self.fourier_coupled_order,
        ) = self._load_gaussians(
            gaussians_hdf5_path,
            fourier_mod_order,
            fourier_coupled_order
        )
        self.gaussians = None
        self._current_time = None


        self._parse_sequence_shape()


        self.qa_data = {}
        if qa_json_path and Path(qa_json_path).exists():
            if self.verbose:
                print(f"加载QA数据: {qa_json_path}")
            with open(qa_json_path, "r", encoding="utf-8") as f:
                qa_obj = json.load(f)
                qa_list = qa_obj.get("qas") if isinstance(qa_obj, dict) else qa_obj
                qa_list = qa_list or []

                for item in qa_list:
                    question = item.get("question")
                    if not question:
                        continue
                    answer = item.get("answer")
                    if answer is None:
                        answer = item.get("answer_text")
                    self.qa_data[question] = answer
            if self.verbose:
                print(f"  加载了 {len(self.qa_data)} 条QA数据")


        self._setup_cameras()

    def _load_gaussians(
        self,
        hdf5_path: str,
        fourier_mod_order: int,
        fourier_coupled_order: int
    ) -> tuple:


        with h5py.File(hdf5_path, "r") as f:
            data = f["gaussians"][:]


            sequence_shape = None
            if "sequence_shape" in f.attrs:
                sequence_shape = tuple(f.attrs["sequence_shape"])
            elif "volume_shape" in f.attrs:
                sequence_shape = tuple(f.attrs["volume_shape"])

            if sequence_shape is None:
                raise ValueError(
                    f"Gaussian文件中没有 sequence_shape/volume_shape 属性: {hdf5_path}\n"
                    "请使用新版训练代码重新训练，或手动添加该属性"
                )
            if self.verbose:
                print(f"  序列尺寸: {sequence_shape}")


            file_fourier_mod_order = int(f.attrs.get("fourier_mod_order", 0))
            file_fourier_coupled_order = int(f.attrs.get("fourier_coupled_order", 0))
            if self.verbose:
                print(f"  傅里叶阶数: mod={file_fourier_mod_order}, coupled={file_fourier_coupled_order}")


        fourier_mod_order = file_fourier_mod_order
        fourier_coupled_order = file_fourier_coupled_order

        if data.ndim == 2:
            data = data[None, ...]

        if self.verbose:
            print(f"  加载了 {data.shape[0]} 帧 Gaussian，单帧 {data.shape[1]} 个点")
        return data, sequence_shape, fourier_mod_order, fourier_coupled_order

    def _setup_cameras(self):


        self.R_xy = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        self.T_xy = np.array([0, 0, self.num_z * 0.5], dtype=np.float32)


        self.R_yz = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.T_yz = np.array([0, 0, self.num_x * 0.5], dtype=np.float32)


        self.R_xz = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)
        self.T_xz = np.array([0, 0, self.num_y * 0.5], dtype=np.float32)

    def _parse_sequence_shape(self) -> None:
        seq = tuple(self.sequence_shape)
        if len(seq) == 5:
            T, D, H, W, C = seq
        elif len(seq) == 4:

            C, T, H, W = seq
            D = 1
            if self.verbose:
                print("警告: sequence_shape 为旧格式，按2D场 + 时间降级处理")
        else:
            raise ValueError(f"不支持的 sequence_shape 维度: {seq}")

        self.num_timesteps = int(T)
        self.num_z = int(D)
        self.num_x = int(H)
        self.num_y = int(W)
        self.num_channels = int(C)

    def _normalize_plane(self, plane: str) -> str:
        if not plane:
            return "xy"
        plane = plane.lower()
        return plane if plane in {"xy", "yz", "xz"} else "xy"

    def _coord_to_index(self, coord: float, size: int, mode: str) -> int:
        if mode == "slice":
            idx = int(round(coord * size + 0.5))
        else:
            idx = int(round(coord * size - 0.5))
        return max(0, min(size - 1, idx))

    def _resolve_slice_index(
        self,
        plane: str,
        slice_index: int | None,
        slice_coord: float | None,
    ) -> int:
        plane = self._normalize_plane(plane)
        if plane == "xy":
            size = self.num_z
        elif plane == "yz":
            size = self.num_x
        else:
            size = self.num_y

        if slice_index is not None:
            return max(0, min(size - 1, int(slice_index)))
        if slice_coord is not None:
            return self._coord_to_index(float(slice_coord), size, mode="slice")
        return size // 2

    def _set_gaussians(self, time: int) -> None:
        time = max(0, min(self.num_timesteps - 1, int(time)))
        if self._current_time == time and self.gaussians is not None:
            return
        frame = self.gaussians_data[time]

        if self.gaussians is None:
            self.gaussians = GaussianModel(
                self.device,
                fourier_mod_order=self.fourier_mod_order,
                fourier_coupled_order=self.fourier_coupled_order,
            )
            self._init_gaussians_from_frame(frame)
        else:
            self._update_gaussians_from_frame(frame)

        self._current_time = time

    def _split_gaussian_frame(self, frame: np.ndarray) -> tuple[np.ndarray, ...]:
        idx = 0
        xyz = frame[:, idx:idx + 3]; idx += 3
        intensity = frame[:, idx:idx + 3]; idx += 3
        scales = frame[:, idx:idx + 3]; idx += 3
        rots = frame[:, idx:idx + 4]; idx += 4
        fourier_mod = None
        fourier_coupled = None
        if self.fourier_mod_order > 0:
            k = self.fourier_mod_order * 4
            fourier_mod = frame[:, idx:idx + k]
            idx += k
        if self.fourier_coupled_order > 0:
            k = self.fourier_coupled_order
            fourier_coupled = frame[:, idx:idx + k]
            idx += k
        return xyz, intensity, scales, rots, fourier_mod, fourier_coupled

    def _init_gaussians_from_frame(self, frame: np.ndarray) -> None:
        xyz, intensity, scales, rots, fourier_mod, fourier_coupled = self._split_gaussian_frame(frame)
        self.gaussians._xyz = torch.nn.Parameter(
            torch.tensor(xyz, dtype=torch.float32, device=self.device)
        )
        self.gaussians._intensity = torch.nn.Parameter(
            torch.tensor(intensity, dtype=torch.float32, device=self.device)
        )
        self.gaussians._scaling = torch.nn.Parameter(
            torch.tensor(scales, dtype=torch.float32, device=self.device)
        )
        self.gaussians._rotation = torch.nn.Parameter(
            torch.tensor(rots, dtype=torch.float32, device=self.device)
        )
        self.gaussians.max_radii2D = torch.zeros(
            (xyz.shape[0],), device=self.device
        )
        if self.fourier_mod_order > 0 and fourier_mod is not None:
            fm = fourier_mod.reshape(xyz.shape[0], self.fourier_mod_order, 4)
            self.gaussians._fourier_mod_coeffs = torch.nn.Parameter(
                torch.tensor(fm, dtype=torch.float32, device=self.device)
            )
        if self.fourier_coupled_order > 0 and fourier_coupled is not None:
            fc = fourier_coupled.reshape(xyz.shape[0], self.fourier_coupled_order, 1)
            self.gaussians._fourier_coupled_coeffs = torch.nn.Parameter(
                torch.tensor(fc, dtype=torch.float32, device=self.device)
            )

    def _update_gaussians_from_frame(self, frame: np.ndarray) -> None:
        xyz, intensity, scales, rots, fourier_mod, fourier_coupled = self._split_gaussian_frame(frame)
        if self.gaussians._xyz.shape[0] != xyz.shape[0]:

            self._init_gaussians_from_frame(frame)
            return
        self.gaussians._xyz.data.copy_(torch.tensor(xyz, dtype=torch.float32, device=self.device))
        self.gaussians._intensity.data.copy_(torch.tensor(intensity, dtype=torch.float32, device=self.device))
        self.gaussians._scaling.data.copy_(torch.tensor(scales, dtype=torch.float32, device=self.device))
        self.gaussians._rotation.data.copy_(torch.tensor(rots, dtype=torch.float32, device=self.device))
        if self.fourier_mod_order > 0 and fourier_mod is not None:
            fm = fourier_mod.reshape(xyz.shape[0], self.fourier_mod_order, 4)
            self.gaussians._fourier_mod_coeffs.data.copy_(
                torch.tensor(fm, dtype=torch.float32, device=self.device)
            )
        if self.fourier_coupled_order > 0 and fourier_coupled is not None:
            fc = fourier_coupled.reshape(xyz.shape[0], self.fourier_coupled_order, 1)
            self.gaussians._fourier_coupled_coeffs.data.copy_(
                torch.tensor(fc, dtype=torch.float32, device=self.device)
            )

    def _render_slice_tensor(self, time: int, plane: str, slice_index: int) -> torch.Tensor:
        plane = self._normalize_plane(plane)
        time = max(0, min(self.num_timesteps - 1, int(time)))
        slice_index = max(0, min(self._get_max_index(plane), int(slice_index)))

        self._set_gaussians(time)
        camera = self._create_camera(slice_index, plane)
        camera = camera.to(self.device)

        with torch.no_grad():
            if self.use_fourier and self.gaussians.fourier_mod_order > 0:
                render_output = render_with_fourier_modulation(
                    camera, self.gaussians, None,
                    energy=False, iteration=1
                )
            else:
                render_output = gs_render(
                    camera, self.gaussians, None, energy=False
                )
            rendered = render_output["render"]  # [C, H, W]
        return rendered

    def _compute_vorticity_slice(self, rendered: torch.Tensor, plane: str) -> np.ndarray:
        plane = self._normalize_plane(plane)
        if rendered.shape[0] < 2:
            return np.zeros((rendered.shape[1], rendered.shape[2]), dtype=np.float32)
        arr = rendered.detach().cpu().numpy()
        vx = arr[0]
        vy = arr[1] if arr.shape[0] > 1 else np.zeros_like(vx)
        vz = arr[2] if arr.shape[0] > 2 else np.zeros_like(vx)
        if plane == "xy":
            dvdx = np.gradient(vy, axis=0)
            dudy = np.gradient(vx, axis=1)
            vorticity = dvdx - dudy
        elif plane == "yz":
            dvzdy = np.gradient(vz, axis=1)
            dvydz = np.gradient(vy, axis=0)
            vorticity = dvzdy - dvydz
        else:  # xz
            dvxdz = np.gradient(vx, axis=0)
            dvzdx = np.gradient(vz, axis=1)
            vorticity = dvxdz - dvzdx
        return vorticity

    def _create_camera(self, index: int, plane: str = "xy") -> Camera:


        C = self.num_channels

        if plane == "xy":
            R, T_vec = self.R_xy, self.T_xy
            width, height = self.num_y, self.num_x  
        elif plane == "yz":
            R, T_vec = self.R_yz, self.T_yz
            width, height = self.num_y, self.num_z  
        else:  # xz
            R, T_vec = self.R_xz, self.T_xz
            width, height = self.num_x, self.num_z  


        gt_image = np.zeros((C, height, width), dtype=np.float32)

        cam_info = CameraInfo(
            R=R,
            T=T_vec,
            view_depth=index,
            gt_image=gt_image,
            channel=C,
            width=width,
            height=height
        )

        return Camera(cam_info)

    def _get_max_index(self, plane: str) -> int:

        if plane == "xy":
            return self.num_z - 1
        if plane == "yz":
            return self.num_x - 1
        return self.num_y - 1

    def render(
        self,
        time: int,
        plane: str = "xy",
        slice_index: int | None = None,
        slice_coord: float | None = None,
        quantity: str = "velocity",
    ) -> Image.Image:


        plane = self._normalize_plane(plane)
        time = max(0, min(self.num_timesteps - 1, int(time)))
        index = self._resolve_slice_index(plane, slice_index, slice_coord)

        rendered = self._render_slice_tensor(time, plane, index)


        return self._tensor_to_pil(rendered, time, plane, index, quantity=quantity)

    def _tensor_to_pil(
        self,
        tensor: torch.Tensor,
        time: int,
        plane: str,
        slice_index: int,
        quantity: str = "velocity",
    ) -> Image.Image:


        plane = self._normalize_plane(plane)

        if quantity == "vorticity":
            speed = torch.from_numpy(
                self._compute_vorticity_slice(tensor, plane)
            ).to(tensor.device)
        else:

            if tensor.shape[0] == 3:
                speed = torch.sqrt((tensor ** 2).sum(dim=0))  # [H, W]
            else:
                speed = tensor[0]


        speed_np = speed.cpu().numpy()
        speed_min, speed_max = speed_np.min(), speed_np.max()
        if speed_max > speed_min:
            speed_norm = (speed_np - speed_min) / (speed_max - speed_min)
        else:
            speed_norm = np.zeros_like(speed_np)


        H, W = speed_norm.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)


        rgb[:, :, 0] = (speed_norm * 255).astype(np.uint8)  # R
        rgb[:, :, 1] = ((1 - np.abs(speed_norm - 0.5) * 2) * 255).astype(np.uint8)  # G
        rgb[:, :, 2] = ((1 - speed_norm) * 255).astype(np.uint8)  # B


        img = Image.fromarray(rgb, mode='RGB')


        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20
            )
        except:
            font = ImageFont.load_default()


        max_index = self._get_max_index(plane)
        if plane == "xy":
            info_text = f"t={time}, z={slice_index}/{max_index} (x-y plane)"
        elif plane == "yz":
            info_text = f"t={time}, x={slice_index}/{max_index} (y-z plane)"
        else:
            info_text = f"t={time}, y={slice_index}/{max_index} (x-z plane)"
        draw.text((10, 10), info_text, fill='white', font=font)


        speed_text = f"Speed: {speed_min:.4f} - {speed_max:.4f} m/s"
        draw.text((10, 35), speed_text, fill='white', font=font)

        return img

    def get_reference_answer(self, question: str) -> Union[str, float]:


        if question in self.qa_data:
            return self.qa_data[question]


        for q, a in self.qa_data.items():
            if question in q or q in question:
                return a


        return "未找到答案"

    def _get_velocity_field(self, time: int, plane: str, slice_index: int) -> np.ndarray:

        rendered = self._render_slice_tensor(time, plane, slice_index)
        if rendered.shape[0] == 3:
            speed = torch.sqrt((rendered ** 2).sum(dim=0))
        else:
            speed = rendered[0]
        return speed.cpu().numpy()

    def _get_vorticity_field(self, time: int, plane: str, slice_index: int) -> np.ndarray:

        rendered = self._render_slice_tensor(time, plane, slice_index)
        return self._compute_vorticity_slice(rendered, plane)
