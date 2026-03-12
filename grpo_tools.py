import math
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from real_fluid_renderer import RealFluidRenderer


def _normalize_plane(plane: str | None) -> str:
    if not plane:
        return "xy"
    plane = str(plane).lower()
    if plane in {"xy", "yz", "xz"}:
        return plane
    if plane in {"xyz", "xyplane"}:
        return "xy"
    if plane in {"yzplane"}:
        return "yz"
    if plane in {"xzplane"}:
        return "xz"
    return "xy"


def _normalize_quantity(quantity: str | None) -> str:
    if quantity is None:
        return "velocity"
    q = str(quantity).strip().lower()
    if q in {"velocity", "vel", "v"}:
        return "velocity"
    if q in {"velocity_magnitude", "speed", "speed_magnitude", "energy", "energy_density", "kinetic_energy"}:
        return "velocity"
    if q in {"vorticity", "omega", "curl", "vort"}:
        return "vorticity"
    if q in {"vorticity_magnitude", "omega_mag", "vort_mag"}:
        return "vorticity"
    return "velocity"


_NUMERIC_RE = re.compile(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _extract_numeric(value, preferred_keys: list[str] | None = None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            match = _NUMERIC_RE.search(text)
            if match:
                try:
                    return float(match.group(0))
                except ValueError:
                    return None
            return None
    if isinstance(value, (list, tuple)) and value:
        return _extract_numeric(value[0], preferred_keys=None)
    if isinstance(value, dict):
        if preferred_keys:
            for key in preferred_keys:
                if key in value:
                    return _extract_numeric(value[key], preferred_keys=None)
        for item in value.values():
            num = _extract_numeric(item, preferred_keys=None)
            if num is not None:
                return num
    return None


def _coord_to_index(coord: float, size: int) -> int:
    idx = int(round(coord * size - 0.5))
    return max(0, min(size - 1, idx))


def _index_to_coord(index: int, size: int) -> float:
    if size <= 0:
        return 0.0
    coord = (float(index) + 0.5) / float(size)
    return max(0.0, min(1.0, coord))


def _coord_to_index_low(coord: float, size: int) -> int:
    idx = int(math.ceil(coord * size - 0.5))
    return max(0, min(size - 1, idx))


def _coord_to_index_high(coord: float, size: int) -> int:
    idx = int(math.floor(coord * size - 0.5))
    return max(0, min(size - 1, idx))


def _is_int_like(value: float, tol: float = 1e-6) -> bool:
    return abs(value - round(value)) <= tol


def _normalize_index_value(value: float | None, size: int) -> int | None:
    if value is None:
        return None
    num = _extract_numeric(value)
    if num is None:
        return None
    idx = int(round(num)) - 1
    return max(0, min(size - 1, idx))


def _normalize_time_index(renderer: RealFluidRenderer, time_value) -> int:
    num = _extract_numeric(time_value)
    if num is None:
        return 0
    idx = int(round(num)) - 1
    return max(0, min(renderer.num_timesteps - 1, idx))


def _normalize_time_indices(renderer: RealFluidRenderer, values) -> list[int]:
    if not values:
        return []
    if not isinstance(values, (list, tuple)):
        values = [values]
    result = []
    for item in values:
        result.append(_normalize_time_index(renderer, item))
    return result


def _normalize_radius_index(renderer: RealFluidRenderer, radius_value) -> int:
    num = _extract_numeric(radius_value)
    if num is None:
        return 1
    if 0.0 <= num <= 1.0 and not _is_int_like(num):
        size = min(renderer.num_x, renderer.num_y, renderer.num_z)
        idx = int(round(num * size))
        return max(0, min(size - 1, idx))
    return max(0, int(round(num)))


def _parse_bound_pair(value) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    if isinstance(value, (list, tuple)):
        if len(value) >= 2:
            return _extract_numeric(value[0]), _extract_numeric(value[1])
        if len(value) == 1:
            num = _extract_numeric(value[0])
            return num, num
    if isinstance(value, dict):
        low = _extract_numeric(value.get("low"))
        if low is None:
            low = _extract_numeric(value.get("min"))
        if low is None:
            low = _extract_numeric(value.get("start"))
        if low is None:
            low = _extract_numeric(value.get("from"))
        high = _extract_numeric(value.get("high"))
        if high is None:
            high = _extract_numeric(value.get("max"))
        if high is None:
            high = _extract_numeric(value.get("end"))
        if high is None:
            high = _extract_numeric(value.get("to"))
        return low, high
    num = _extract_numeric(value)
    return num, num


def _bound_to_index_low(value: float | None, size: int) -> int | None:
    if value is None:
        return None
    num = _extract_numeric(value)
    if num is None:
        return None
    if 0.0 <= num <= 1.0:
        return _coord_to_index_low(float(num), size)
    return _normalize_index_value(num, size)


def _bound_to_index_high(value: float | None, size: int) -> int | None:
    if value is None:
        return None
    num = _extract_numeric(value)
    if num is None:
        return None
    if 0.0 <= num <= 1.0:
        return _coord_to_index_high(float(num), size)
    return _normalize_index_value(num, size)


def _normalize_roi_indices_2d(
    plane: str,
    shape: tuple[int, int],
    roi,
) -> tuple[int, int, int, int] | None:
    if roi is None:
        return None
    h, w = int(shape[0]), int(shape[1])

    axis0, axis1 = {"xy": ("x", "y"), "yz": ("y", "z"), "xz": ("x", "z")}[plane]
    range0 = None
    range1 = None

    points = None
    if isinstance(roi, dict):
        pts = roi.get("points")
        if isinstance(pts, (list, tuple)) and len(pts) >= 2:
            points = pts
    if points:
        axis_map = {"xy": (0, 1), "yz": (1, 2), "xz": (0, 2)}[plane]
        vals0 = []
        vals1 = []
        for p in points:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            if len(p) >= 3:
                v0 = _extract_numeric(p[axis_map[0]])
                v1 = _extract_numeric(p[axis_map[1]])
            else:
                v0 = _extract_numeric(p[0])
                v1 = _extract_numeric(p[1])
            if v0 is not None:
                vals0.append(v0)
            if v1 is not None:
                vals1.append(v1)
        if vals0 and vals1:
            range0 = [min(vals0), max(vals0)]
            range1 = [min(vals1), max(vals1)]
            roi = None
    if isinstance(roi, dict):
        range0 = roi.get(axis0)
        if range0 is None:
            range0 = roi.get("axis0") or roi.get("u")
        range1 = roi.get(axis1)
        if range1 is None:
            range1 = roi.get("axis1") or roi.get("v")
    elif isinstance(roi, (list, tuple)):
        if len(roi) == 2 and any(isinstance(item, (list, tuple, dict)) for item in roi):
            range0, range1 = roi[0], roi[1]
        elif len(roi) >= 4:
            range0, range1 = roi[0:2], roi[2:4]
        elif len(roi) == 2:
            range0, range1 = roi, roi
    low0, high0 = _parse_bound_pair(range0)
    low1, high1 = _parse_bound_pair(range1)
    if low0 is None and high0 is None:
        low0, high0 = 0.0, 1.0
    if low1 is None and high1 is None:
        low1, high1 = 0.0, 1.0
    idx0 = _bound_to_index_low(low0, h)
    idx1 = _bound_to_index_high(high0, h)
    idy0 = _bound_to_index_low(low1, w)
    idy1 = _bound_to_index_high(high1, w)
    if idx0 is None:
        idx0 = 0
    if idx1 is None:
        idx1 = h - 1
    if idy0 is None:
        idy0 = 0
    if idy1 is None:
        idy1 = w - 1
    return max(0, idx0), min(h - 1, idx1), max(0, idy0), min(w - 1, idy1)


def _normalize_roi_indices_3d(renderer: RealFluidRenderer, roi) -> tuple[int, int, int, int, int, int] | None:
    if roi is None:
        return None
    nx, ny, nz = renderer.num_x, renderer.num_y, renderer.num_z
    range_x = None
    range_y = None
    range_z = None
    if isinstance(roi, dict):
        range_x = roi.get("x")
        range_y = roi.get("y")
        range_z = roi.get("z")
    elif isinstance(roi, (list, tuple)):
        if len(roi) == 3 and any(isinstance(item, (list, tuple, dict)) for item in roi):
            range_x, range_y, range_z = roi[0], roi[1], roi[2]
        elif len(roi) >= 6:
            range_x, range_y, range_z = roi[0:2], roi[2:4], roi[4:6]
        elif len(roi) == 3:
            range_x = [roi[0], roi[0]]
            range_y = [roi[1], roi[1]]
            range_z = [roi[2], roi[2]]
    low_x, high_x = _parse_bound_pair(range_x)
    low_y, high_y = _parse_bound_pair(range_y)
    low_z, high_z = _parse_bound_pair(range_z)
    if low_x is None and high_x is None:
        low_x, high_x = 0.0, 1.0
    if low_y is None and high_y is None:
        low_y, high_y = 0.0, 1.0
    if low_z is None and high_z is None:
        low_z, high_z = 0.0, 1.0
    ix0 = _bound_to_index_low(low_x, nx)
    ix1 = _bound_to_index_high(high_x, nx)
    iy0 = _bound_to_index_low(low_y, ny)
    iy1 = _bound_to_index_high(high_y, ny)
    iz0 = _bound_to_index_low(low_z, nz)
    iz1 = _bound_to_index_high(high_z, nz)
    if ix0 is None:
        ix0 = 0
    if ix1 is None:
        ix1 = nx - 1
    if iy0 is None:
        iy0 = 0
    if iy1 is None:
        iy1 = ny - 1
    if iz0 is None:
        iz0 = 0
    if iz1 is None:
        iz1 = nz - 1
    return (
        max(0, ix0),
        min(nx - 1, ix1),
        max(0, iy0),
        min(ny - 1, iy1),
        max(0, iz0),
        min(nz - 1, iz1),
    )


def _data_eps(data: np.ndarray) -> float:
    if data.size == 0:
        return float(np.finfo(np.float32).eps)
    scale = float(np.max(np.abs(data)))
    return float(np.finfo(np.float32).eps * max(scale, 1.0))


def _central_diff_periodic(field: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    forward = np.roll(field, -1, axis=axis)
    backward = np.roll(field, 1, axis=axis)
    return (forward - backward) / (2.0 * spacing)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    eps = max(_data_eps(a), _data_eps(b))
    if float(np.std(a)) <= eps or float(np.std(b)) <= eps:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _compute_point_gradient_stats(field: np.ndarray, axis0: int, axis1: int) -> dict:
    h, w = field.shape
    left = (axis0 - 1) % h
    right = (axis0 + 1) % h
    down = (axis1 - 1) % w
    up = (axis1 + 1) % w

    val_left = float(field[left, axis1])
    val_right = float(field[right, axis1])
    val_down = float(field[axis0, down])
    val_up = float(field[axis0, up])

    dx = 1.0 / float(h)
    dy = 1.0 / float(w)
    grad_x = abs(val_right - val_left) / (2.0 * dx)
    grad_y = abs(val_up - val_down) / (2.0 * dy)

    grad_x_field = _central_diff_periodic(field, axis=0, spacing=dx)
    grad_y_field = _central_diff_periodic(field, axis=1, spacing=dy)
    grad_mag = np.sqrt(grad_x_field ** 2 + grad_y_field ** 2)
    grad_median = float(np.median(grad_mag))
    grad_std = float(np.std(grad_mag))

    return {
        "grad_x": grad_x,
        "grad_y": grad_y,
        "grad_median": grad_median,
        "grad_std": grad_std,
    }


def _weighted_mean_std(coords: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return float(np.mean(coords)), float(np.std(coords))
    mean = float(np.sum(coords * weights) / weight_sum)
    var = float(np.sum(weights * (coords - mean) ** 2) / weight_sum)
    return mean, float(np.sqrt(var))


def _weighted_bounds(coords: np.ndarray, weights: np.ndarray) -> tuple[float, float, float, float]:
    mean, std = _weighted_mean_std(coords, weights)
    coord_min = float(np.min(coords))
    coord_max = float(np.max(coords))
    lower = max(coord_min, mean - std)
    upper = min(coord_max, mean + std)
    return lower, upper, mean, std


def _weighted_centroid(coords_x: np.ndarray, coords_y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return np.array([float(np.mean(coords_x)), float(np.mean(coords_y))])
    cx = float(np.sum(coords_x * weights) / weight_sum)
    cy = float(np.sum(coords_y * weights) / weight_sum)
    return np.array([cx, cy])


def _coords_for_shape(n0: int, n1: int) -> tuple[np.ndarray, np.ndarray]:
    coords0 = (np.arange(n0) + 0.5) / float(n0)
    coords1 = (np.arange(n1) + 0.5) / float(n1)
    return np.meshgrid(coords0, coords1, indexing="ij")


def _coords_for_roi(
    full_shape: tuple[int, int],
    low0: int,
    low1: int,
    roi_h: int,
    roi_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    full_h = int(full_shape[0])
    full_w = int(full_shape[1])
    coords0 = (np.arange(low0, low0 + roi_h) + 0.5) / float(full_h)
    coords1 = (np.arange(low1, low1 + roi_w) + 0.5) / float(full_w)
    return np.meshgrid(coords0, coords1, indexing="ij")


def _shrink_cache(cache: dict, max_items: int) -> None:
    if len(cache) <= max_items:
        return
    for key in list(cache.keys())[:-max_items]:
        cache.pop(key, None)


def _get_tool_cache(renderer: RealFluidRenderer) -> dict:
    cache = getattr(renderer, "_tool_cache", None)
    if cache is None:
        cache = {"velocity": {}, "vorticity": {}}
        setattr(renderer, "_tool_cache", cache)
    return cache


def _get_velocity_volume(renderer: RealFluidRenderer, time: int) -> np.ndarray:
    cache = _get_tool_cache(renderer)["velocity"]
    if time in cache:
        return cache[time]
    nx, ny, nz = renderer.num_x, renderer.num_y, renderer.num_z
    volume = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    for z in range(nz):
        rendered = renderer._render_slice_tensor(time, "xy", z)
        arr = rendered.detach().cpu().numpy()
        if arr.shape[0] >= 3:
            volume[:, :, z, :] = np.transpose(arr[:3, :, :], (1, 2, 0))
        else:
            volume[:, :, z, 0] = arr[0]
    cache[time] = volume
    _shrink_cache(cache, max_items=2)
    return volume


def _render_velocity_slice(renderer: RealFluidRenderer, time: int, plane: str, slice_index: int) -> np.ndarray:
    plane = _normalize_plane(plane)
    rendered = renderer._render_slice_tensor(time, plane, slice_index)
    arr = rendered.detach().cpu().numpy()
    if arr.shape[0] >= 3:
        if plane == "xy":
            return np.transpose(arr[:3, :, :], (1, 2, 0))
        return np.transpose(arr[:3, :, :], (2, 1, 0))
    height, width = arr.shape[1], arr.shape[2]
    slice_vec = np.zeros((height, width, 3), dtype=np.float32)
    slice_vec[..., 0] = arr[0]
    return slice_vec


def _render_speed_slice(renderer: RealFluidRenderer, time: int, plane: str, slice_index: int) -> np.ndarray:
    slice_vec = _render_velocity_slice(renderer, time, plane, slice_index)
    return np.linalg.norm(slice_vec, axis=-1)


_COLORMAP_CACHE: dict[str, np.ndarray] = {}


def _build_linear_colormap(stops: list[tuple[float, tuple[int, int, int]]], size: int = 256) -> np.ndarray:
    if size <= 1:
        return np.array([[0, 0, 0]], dtype=np.uint8)
    stops = sorted(stops, key=lambda x: x[0])
    xs = [s[0] for s in stops]
    cols = [s[1] for s in stops]
    lut = np.zeros((size, 3), dtype=np.uint8)
    for i in range(size):
        t = i / float(size - 1)
        if t <= xs[0]:
            lut[i] = cols[0]
            continue
        if t >= xs[-1]:
            lut[i] = cols[-1]
            continue
        for j in range(len(xs) - 1):
            if xs[j] <= t <= xs[j + 1]:
                t0, t1 = xs[j], xs[j + 1]
                c0 = np.array(cols[j], dtype=np.float32)
                c1 = np.array(cols[j + 1], dtype=np.float32)
                alpha = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
                lut[i] = np.clip(c0 + alpha * (c1 - c0), 0, 255).astype(np.uint8)
                break
    return lut


def _get_colormap_lut(name: str | None) -> np.ndarray:
    key = str(name or "viridis").strip().lower()
    if key in _COLORMAP_CACHE:
        return _COLORMAP_CACHE[key]
    if key in {"gray", "grayscale"}:
        lut = np.stack([np.arange(256), np.arange(256), np.arange(256)], axis=1).astype(np.uint8)
        _COLORMAP_CACHE[key] = lut
        return lut

    stops = [
        (0.0, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.5, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.0, (253, 231, 37)),
    ]
    lut = _build_linear_colormap(stops, size=256)
    _COLORMAP_CACHE[key] = lut
    return lut


def _compute_vmin_vmax(field: np.ndarray, vmin: float | None, vmax: float | None) -> tuple[float, float]:
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return 0.0, 1.0
    if vmin is None:
        vmin = float(np.nanpercentile(finite, 1.0))
    else:
        vmin = float(vmin)
    if vmax is None:
        vmax = float(np.nanpercentile(finite, 99.0))
    else:
        vmax = float(vmax)
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(finite))
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        eps = max(_data_eps(finite), 1e-6)
        vmax = vmin + eps
    return vmin, vmax


def _format_value(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.3g}"


def _field_to_rgb(field: np.ndarray, vmin: float, vmax: float, cmap: str | None) -> np.ndarray:
    lut = _get_colormap_lut(cmap)
    scale = vmax - vmin
    norm = (field - vmin) / scale
    norm = np.clip(norm, 0.0, 1.0)
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    idx = (norm * 255.0).round().astype(np.uint8)
    return lut[idx]


def _build_colorbar(height: int, cmap: str | None, width: int) -> Image.Image:
    bar_width = max(2, int(width))
    gradient_idx = np.linspace(255, 0, height).round().astype(np.uint8)
    lut = _get_colormap_lut(cmap)
    bar = lut[gradient_idx]
    bar = np.repeat(bar[:, None, :], bar_width, axis=1)
    return Image.fromarray(bar, mode="RGB")


def _compose_with_colorbar(
    main_img: Image.Image,
    bar_img: Image.Image,
    vmin: float,
    vmax: float,
    title: str,
    gap: int = 6,
) -> Image.Image:
    main_w, main_h = main_img.size
    bar_w, bar_h = bar_img.size
    if bar_h != main_h:
        bar_img = bar_img.resize((bar_w, main_h), resample=Image.BILINEAR)
    font = ImageFont.load_default()
    draw_tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    label_width = 64
    label_gap = 4
    left_pad = 8
    right_pad = 8
    top_pad = 4
    bottom_pad = 6
    title_pad = 4
    tick_count = 3

    def _measure(text: str) -> tuple[int, int]:
        try:
            bbox = draw_tmp.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            return draw_tmp.textsize(text, font=font)

    title_w, title_h = _measure(title)
    title_height = title_h + 2
    y_main = top_pad + title_height + title_pad
    canvas_w = left_pad + main_w + gap + bar_img.size[0] + label_gap + label_width + right_pad
    canvas_h = y_main + main_h + bottom_pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    canvas.paste(main_img, (left_pad, y_main))
    canvas.paste(bar_img, (left_pad + main_w + gap, y_main))
    draw = ImageDraw.Draw(canvas)
    title_x = left_pad + max(0, (main_w - title_w) // 2)
    draw.text((title_x, top_pad), title, fill=(0, 0, 0), font=font)

    if tick_count < 2:
        tick_values = [vmin]
    else:
        step = (vmax - vmin) / (tick_count - 1)
        tick_values = [vmin + step * idx for idx in range(tick_count)]
    tick_labels = [_format_value(val) for val in tick_values]
    label_h = max(_measure(label)[1] for label in tick_labels) if tick_labels else 0
    tick_x0 = left_pad + main_w + gap + bar_img.size[0]
    for val, label in zip(tick_values, tick_labels, strict=False):
        if vmax == vmin:
            norm = 0.0
        else:
            norm = (val - vmin) / (vmax - vmin)
        norm = max(0.0, min(1.0, norm))
        y = y_main + int(round((1.0 - norm) * (main_h - 1)))
        draw.line((tick_x0, y, tick_x0 + 4, y), fill=(0, 0, 0))
        label_y = max(y_main, min(y - label_h // 2, y_main + main_h - label_h))
        draw.text((tick_x0 + label_gap, label_y), label, fill=(0, 0, 0), font=font)
    return canvas


def _pad_to_multiple(img: Image.Image, multiple: int, fill: tuple[int, int, int]) -> Image.Image:
    width, height = img.size
    pad_w = (multiple - (width % multiple)) % multiple
    pad_h = (multiple - (height % multiple)) % multiple
    if pad_w == 0 and pad_h == 0:
        return img
    canvas = Image.new("RGB", (width + pad_w, height + pad_h), color=fill)
    canvas.paste(img, (0, 0))
    return canvas


def _compute_vorticity_components(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = volume.shape[0]
    dx = 1.0 / float(n)
    u = volume[..., 0]
    v = volume[..., 1]
    w = volume[..., 2]

    dudy = _central_diff_periodic(u, axis=1, spacing=dx)
    dudz = _central_diff_periodic(u, axis=2, spacing=dx)

    dvdx = _central_diff_periodic(v, axis=0, spacing=dx)
    dvdz = _central_diff_periodic(v, axis=2, spacing=dx)

    dwdx = _central_diff_periodic(w, axis=0, spacing=dx)
    dwdy = _central_diff_periodic(w, axis=1, spacing=dx)

    omega_x = dwdy - dvdz
    omega_y = dudz - dwdx
    omega_z = dvdx - dudy
    omega_mag = np.sqrt(omega_x ** 2 + omega_y ** 2 + omega_z ** 2)
    return omega_x, omega_y, omega_z, omega_mag


def _get_vorticity_components(renderer: RealFluidRenderer, time: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache = _get_tool_cache(renderer)["vorticity"]
    if time in cache:
        return cache[time]
    volume = _get_velocity_volume(renderer, time)
    comps = _compute_vorticity_components(volume)
    cache[time] = comps
    _shrink_cache(cache, max_items=2)
    return comps


def _get_vorticity_slice(omega_mag: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "xy":
        return omega_mag[:, :, index]
    if plane == "yz":
        return omega_mag[index, :, :]
    return omega_mag[:, index, :]


def _compute_vortex_displacement(
    renderer: RealFluidRenderer,
    time: int,
    xy_slice_index: int | None,
    xz_slice_index: int | None = None,
    yz_slice_index: int | None = None,
) -> dict:
    if xy_slice_index is None:
        return {}
    _, _, _, omega_mag = _get_vorticity_components(renderer, time)
    slice_xy = _get_vorticity_slice(omega_mag, "xy", xy_slice_index)
    if slice_xy.size == 0:
        return {}

    loc_xy = np.unravel_index(int(np.argmax(slice_xy)), slice_xy.shape)
    x_from_xy = (int(loc_xy[0]) + 0.5) / float(renderer.num_x)
    y_from_xy = (int(loc_xy[1]) + 0.5) / float(renderer.num_y)

    center_x = float(np.mean((np.arange(renderer.num_x) + 0.5) / float(renderer.num_x)))
    center_y = float(np.mean((np.arange(renderer.num_y) + 0.5) / float(renderer.num_y)))
    center_z = float(np.mean((np.arange(renderer.num_z) + 0.5) / float(renderer.num_z)))

    if xz_slice_index is not None:
        slice_xz = _get_vorticity_slice(omega_mag, "xz", xz_slice_index)
        if slice_xz.size == 0:
            return {}
        loc_xz = np.unravel_index(int(np.argmax(slice_xz)), slice_xz.shape)
        x_from_xz = (int(loc_xz[0]) + 0.5) / float(renderer.num_x)
        z_coord = (int(loc_xz[1]) + 0.5) / float(renderer.num_z)
        x_coord = (x_from_xy + x_from_xz) / 2.0
        y_coord = y_from_xy
        dx = x_coord - center_x
        dy = y_coord - center_y
        dz = z_coord - center_z
        disp_mag = float(abs(dx) + abs(dy) + abs(dz))

        disp_mags = []
        for t in range(renderer.num_timesteps):
            _, _, _, omega_t = _get_vorticity_components(renderer, t)
            slice_xy_t = _get_vorticity_slice(omega_t, "xy", xy_slice_index)
            slice_xz_t = _get_vorticity_slice(omega_t, "xz", xz_slice_index)
            loc_xy_t = np.unravel_index(int(np.argmax(slice_xy_t)), slice_xy_t.shape)
            loc_xz_t = np.unravel_index(int(np.argmax(slice_xz_t)), slice_xz_t.shape)
            x_xy = (int(loc_xy_t[0]) + 0.5) / float(renderer.num_x)
            x_xz = (int(loc_xz_t[0]) + 0.5) / float(renderer.num_x)
            x_avg_t = (x_xy + x_xz) / 2.0
            y_t = (int(loc_xy_t[1]) + 0.5) / float(renderer.num_y)
            z_t = (int(loc_xz_t[1]) + 0.5) / float(renderer.num_z)
            disp_mags.append(
                float(
                    abs(x_avg_t - center_x)
                    + abs(y_t - center_y)
                    + abs(z_t - center_z)
                )
            )
    else:

        x_coord = x_from_xy
        y_coord = y_from_xy
        dx = x_coord - center_x
        dy = y_coord - center_y
        dz = 0.0
        disp_mag = float(abs(dx) + abs(dy))
        disp_mags = []
        for t in range(renderer.num_timesteps):
            _, _, _, omega_t = _get_vorticity_components(renderer, t)
            slice_xy_t = _get_vorticity_slice(omega_t, "xy", xy_slice_index)
            loc_xy_t = np.unravel_index(int(np.argmax(slice_xy_t)), slice_xy_t.shape)
            x_t = (int(loc_xy_t[0]) + 0.5) / float(renderer.num_x)
            y_t = (int(loc_xy_t[1]) + 0.5) / float(renderer.num_y)
            disp_mags.append(float(abs(x_t - center_x) + abs(y_t - center_y)))

    disp_threshold = float(np.median(np.array(disp_mags))) if disp_mags else 0.0
    return {
        "displacement": [float(dx), float(dy), float(dz)],
        "disp_mag": disp_mag,
        "disp_threshold": disp_threshold,
    }


def _slice_size(renderer: RealFluidRenderer, plane: str) -> int:
    if plane == "xy":
        return renderer.num_z
    if plane == "yz":
        return renderer.num_x
    return renderer.num_y


def _plane_shape(renderer: RealFluidRenderer, plane: str) -> tuple[int, int]:
    if plane == "xy":
        return renderer.num_x, renderer.num_y
    if plane == "yz":
        return renderer.num_y, renderer.num_z
    return renderer.num_x, renderer.num_z


def _normalize_slice_indices(
    renderer: RealFluidRenderer,
    plane: str,
    slice_index,
    slice_coord,
) -> tuple[str, int | None]:
    plane = _normalize_plane(plane)
    coord_key = {"xy": "z", "yz": "x", "xz": "y"}[plane]

    coord = _extract_numeric(slice_coord, preferred_keys=[coord_key])
    idx_val = _extract_numeric(slice_index)

    if coord is None and idx_val is not None:
        if 0.0 <= idx_val <= 1.0 and not _is_int_like(idx_val):
            coord = idx_val
            idx_val = None

    if coord is not None:
        size = _slice_size(renderer, plane)
        idx = _coord_to_index(coord, size)
    elif idx_val is not None:
        size = _slice_size(renderer, plane)
        idx = _normalize_index_value(idx_val, size)
    else:
        idx = None

    return plane, idx


def _normalize_point_indices(
    renderer: RealFluidRenderer,
    plane: str,
    point_index,
    point_coord,
) -> list[int] | None:
    plane = _normalize_plane(plane)
    if point_coord is not None:
        if isinstance(point_coord, dict):
            if plane == "xy":
                x = _extract_numeric(point_coord, preferred_keys=["x"])
                y = _extract_numeric(point_coord, preferred_keys=["y"])
            elif plane == "yz":
                x = _extract_numeric(point_coord, preferred_keys=["y"])
                y = _extract_numeric(point_coord, preferred_keys=["z"])
            else:
                x = _extract_numeric(point_coord, preferred_keys=["x"])
                y = _extract_numeric(point_coord, preferred_keys=["z"])
            if x is not None and y is not None:
                sizes = {
                    "xy": (renderer.num_x, renderer.num_y),
                    "yz": (renderer.num_y, renderer.num_z),
                    "xz": (renderer.num_x, renderer.num_z),
                }[plane]
                return [
                    _coord_to_index(x, sizes[0]),
                    _coord_to_index(y, sizes[1]),
                ]
        elif isinstance(point_coord, (list, tuple)) and len(point_coord) >= 2:
            sizes = {
                "xy": (renderer.num_x, renderer.num_y),
                "yz": (renderer.num_y, renderer.num_z),
                "xz": (renderer.num_x, renderer.num_z),
            }[plane]
            return [
                _coord_to_index(_extract_numeric(point_coord[0]) or 0.0, sizes[0]),
                _coord_to_index(_extract_numeric(point_coord[1]) or 0.0, sizes[1]),
            ]

    if isinstance(point_index, (list, tuple)) and len(point_index) >= 2:
        sizes = {
            "xy": (renderer.num_x, renderer.num_y),
            "yz": (renderer.num_y, renderer.num_z),
            "xz": (renderer.num_x, renderer.num_z),
        }[plane]
        return [
            _normalize_index_value(point_index[0], sizes[0]) or 0,
            _normalize_index_value(point_index[1], sizes[1]) or 0,
        ]
    if point_index is not None:
        num = _extract_numeric(point_index)
        if num is None:
            return None
        sizes = {
            "xy": (renderer.num_x, renderer.num_y),
            "yz": (renderer.num_y, renderer.num_z),
            "xz": (renderer.num_x, renderer.num_z),
        }[plane]
        if 0.0 <= num <= 1.0 and not _is_int_like(num):
            return [
                _coord_to_index(num, sizes[0]),
                _coord_to_index(num, sizes[1]),
            ]
        return [
            _normalize_index_value(num, sizes[0]) or 0,
            _normalize_index_value(num, sizes[1]) or 0,
        ]
    return None


def _normalize_center_indices(
    renderer: RealFluidRenderer,
    center_index,
    center_coord,
) -> list[int] | None:
    if center_coord is not None:
        if isinstance(center_coord, dict):
            x = _extract_numeric(center_coord, preferred_keys=["x"])
            y = _extract_numeric(center_coord, preferred_keys=["y"])
            z = _extract_numeric(center_coord, preferred_keys=["z"])
            if x is not None and y is not None and z is not None:
                return [
                    _coord_to_index(x, renderer.num_x),
                    _coord_to_index(y, renderer.num_y),
                    _coord_to_index(z, renderer.num_z),
                ]
        elif isinstance(center_coord, (list, tuple)) and len(center_coord) >= 3:
            return [
                _coord_to_index(_extract_numeric(center_coord[0]) or 0.0, renderer.num_x),
                _coord_to_index(_extract_numeric(center_coord[1]) or 0.0, renderer.num_y),
                _coord_to_index(_extract_numeric(center_coord[2]) or 0.0, renderer.num_z),
            ]

    if isinstance(center_index, (list, tuple)) and len(center_index) >= 3:
        return [
            _normalize_index_value(center_index[0], renderer.num_x) or 0,
            _normalize_index_value(center_index[1], renderer.num_y) or 0,
            _normalize_index_value(center_index[2], renderer.num_z) or 0,
        ]
    return None


def _normalize_plane_slices(
    renderer: RealFluidRenderer,
    slice_indices,
    slice_coords,
) -> dict | None:
    planes = ["xy", "yz", "xz"]
    indices: dict[str, int] = {}

    if isinstance(slice_coords, list):
        slice_coords = {plane: val for plane, val in zip(planes, slice_coords)}
    if isinstance(slice_indices, list):
        slice_indices = {plane: val for plane, val in zip(planes, slice_indices)}

    if slice_coords and isinstance(slice_coords, dict):
        for plane, val in slice_coords.items():
            plane = _normalize_plane(plane)
            coord = _extract_numeric(val, preferred_keys=[{"xy": "z", "yz": "x", "xz": "y"}[plane]])
            if coord is None:
                continue
            size = _slice_size(renderer, plane)
            indices[plane] = _coord_to_index(coord, size)

    if slice_indices and isinstance(slice_indices, dict):
        for plane, val in slice_indices.items():
            plane = _normalize_plane(plane)
            num = _extract_numeric(val)
            if num is None:
                continue
            size = _slice_size(renderer, plane)
            if 0.0 <= num <= 1.0 and not _is_int_like(num):
                indices[plane] = _coord_to_index(num, size)
            else:
                indices[plane] = _normalize_index_value(num, size)

    return indices or None


FLUID_TOOLS = [


    {
        "type": "function",
        "function": {
            "name": "slice_stats",
            "description": "返回切片统计，可选返回点值与梯度信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {"anyOf": [{"type": "integer"}, {"type": "number"}], "description": "时间步索引"},
                    "plane": {"type": "string", "enum": ["xy", "yz", "xz"], "description": "切片平面"},
                    "slice_index": {"anyOf": [{"type": "integer"}, {"type": "number"}], "description": "切片索引"},
                    "slice_coord": {
                        "anyOf": [
                            {"type": "number"},
                            {"type": "array", "items": {"type": "number"}},
                            {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                    "z": {"type": "number"},
                                },
                                "additionalProperties": False,
                            },
                        ],
                        "description": "切片归一化坐标(0-1)，支持数值/数组/字典",
                    },
                    "point_index": {"type": "array", "items": {"type": "number"}, "description": "点索引[axis0, axis1]"},
                    "point_coord": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "number"}},
                            {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                    "z": {"type": "number"},
                                },
                                "additionalProperties": False,
                            },
                        ],
                        "description": "点坐标（支持数组或{x,y,z}字典）",
                    },
                    "roi": {
                        "anyOf": [{"type": "object"}, {"type": "array"}],
                        "description": "任意矩形ROI范围(0-1坐标或索引)，如 {\"x\":[0.2,0.6],\"y\":[0.4,0.9]} 或 [[x0,x1],[y0,y1]]",
                    },
                    "quantity": {
                        "type": "string",
                        "enum": [
                            "velocity",
                            "vorticity",
                            "velocity_magnitude",
                            "speed",
                            "energy",
                            "kinetic_energy",
                            "vorticity_magnitude",
                            "omega",
                            "curl",
                        ],
                        "description": "物理量（支持别名，会自动归一化）",
                    },
                },
                "required": ["time", "plane"],
            },
        },
    },


    {
        "type": "function",
        "function": {
            "name": "slice_compare",
            "description": "比较两个时间步同一切片的相关系数与均值变化；可选time_indices用于三时刻最大值比较。",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_a": {"anyOf": [{"type": "integer"}, {"type": "number"}], "description": "时间步A"},
                    "time_b": {"anyOf": [{"type": "integer"}, {"type": "number"}], "description": "时间步B"},
                    "time_indices": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "三时刻索引列表（用于最大值比较）",
                    },
                    "plane": {"type": "string", "enum": ["xy", "yz", "xz"], "description": "切片平面"},
                    "slice_index": {"anyOf": [{"type": "integer"}, {"type": "number"}], "description": "切片索引"},
                    "slice_coord": {
                        "anyOf": [
                            {"type": "number"},
                            {"type": "array", "items": {"type": "number"}},
                            {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                    "z": {"type": "number"},
                                },
                                "additionalProperties": False,
                            },
                        ],
                        "description": "切片归一化坐标(0-1)，支持数值/数组/字典",
                    },
                    "roi": {
                        "anyOf": [{"type": "object"}, {"type": "array"}],
                        "description": "任意矩形ROI范围(0-1坐标或索引)，如 {\"x\":[0.2,0.6],\"y\":[0.4,0.9]} 或 [[x0,x1],[y0,y1]]",
                    },
                    "quantity": {
                        "type": "string",
                        "enum": [
                            "velocity",
                            "vorticity",
                            "velocity_magnitude",
                            "speed",
                            "energy",
                            "kinetic_energy",
                            "vorticity_magnitude",
                            "omega",
                            "curl",
                        ],
                        "description": "物理量（支持别名，会自动归一化）",
                    },
                },
                "required": ["time_a", "time_b", "plane"],
            },
        },
    },


    {
        "type": "function",
        "function": {
            "name": "cube_components",
            "description": "计算小立方体区域内速度分量的平均幅值（radius=0 表示单点；半径支持归一化0-1或网格索引单位）；可选time_indices用于点时序最大值比较。",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {"anyOf": [{"type": "integer"}, {"type": "number"}], "description": "时间步索引"},
                    "time_indices": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "三时刻索引列表（用于点时序比较）",
                    },
                    "center_coord": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "number"}},
                            {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                    "z": {"type": "number"},
                                },
                                "additionalProperties": False,
                            },
                        ],
                        "description": "中心坐标[x,y,z]（归一化0-1，支持数组或字典）",
                    },
                    "roi": {
                        "anyOf": [{"type": "object"}, {"type": "array"}],
                        "description": "任意3D ROI范围(0-1坐标或索引)，如 {\"x\":[0.1,0.4],\"y\":[0.2,0.8],\"z\":[0.3,0.9]}",
                    },
                    "radius": {
                        "anyOf": [{"type": "integer"}, {"type": "number"}],
                        "description": "立方体半径（索引单位或归一化0-1；非整数且<=1时视为归一化）",
                    },
                },
                "required": ["time"],
            },
        },
    },


    {
        "type": "function",
        "function": {
            "name": "plane_uniformity",
            "description": "比较XY/YZ/XZ三种平面的cv，返回cv。",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {"anyOf": [{"type": "integer"}, {"type": "number"}], "description": "时间步索引"},
                    "slice_indices": {
                        "anyOf": [{"type": "object"}, {"type": "array", "items": {"type": "number"}}],
                        "description": "各平面切片索引，如 {\"xy\":5,\"yz\":7,\"xz\":5} 或 [xy,yz,xz]",
                    },
                    "slice_coords": {
                        "anyOf": [{"type": "object"}, {"type": "array", "items": {"type": "number"}}],
                        "description": "各平面切片坐标(0-1)，支持对象或数组[xy,yz,xz]",
                    },
                    "roi": {
                        "anyOf": [{"type": "object"}, {"type": "array"}],
                        "description": "通用ROI范围(0-1坐标或索引)，按平面自动取对应轴",
                    },
                    "plane_rois": {
                        "type": "object",
                        "description": "分平面ROI，如 {\"xy\":{\"x\":[...],\"y\":[...]},\"yz\":{\"y\":[...],\"z\":[...]}}",
                    },
                    "quantity": {
                        "type": "string",
                        "enum": [
                            "velocity",
                            "vorticity",
                            "velocity_magnitude",
                            "speed",
                            "energy",
                            "kinetic_energy",
                            "vorticity_magnitude",
                            "omega",
                            "curl",
                        ],
                        "description": "物理量（支持别名，会自动归一化）",
                    },
                },
                "required": ["time"],
            },
        },
    },


    {
        "type": "function",
        "function": {
            "name": "vorticity_orientation",
            "description": "返回高涡度区域的主轴统计与涡核位移信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {"anyOf": [{"type": "integer"}, {"type": "number"}], "description": "时间步索引"},
                    "xy_slice_index": {"anyOf": [{"type": "integer"}, {"type": "number"}], "description": "XY切片索引(z)"},
                    "yz_slice_index": {"anyOf": [{"type": "integer"}, {"type": "number"}], "description": "YZ切片索引(x)"},
                    "xy_slice_coord": {
                        "anyOf": [
                            {"type": "number"},
                            {"type": "array", "items": {"type": "number"}},
                            {
                                "type": "object",
                                "properties": {"z": {"type": "number"}},
                                "additionalProperties": True,
                            },
                        ],
                        "description": "XY切片坐标",
                    },
                    "yz_slice_coord": {
                        "anyOf": [
                            {"type": "number"},
                            {"type": "array", "items": {"type": "number"}},
                            {
                                "type": "object",
                                "properties": {"x": {"type": "number"}},
                                "additionalProperties": True,
                            },
                        ],
                        "description": "YZ切片坐标",
                    },
                    "xz_slice_index": {"anyOf": [{"type": "integer"}, {"type": "number"}], "description": "XZ切片索引(y)"},
                    "xz_slice_coord": {
                        "anyOf": [
                            {"type": "number"},
                            {"type": "array", "items": {"type": "number"}},
                            {
                                "type": "object",
                                "properties": {"y": {"type": "number"}},
                                "additionalProperties": True,
                            },
                        ],
                        "description": "XZ切片坐标",
                    },
                    "roi": {
                        "anyOf": [{"type": "object"}, {"type": "array"}],
                        "description": "任意3D ROI范围(0-1坐标或索引)，如 {\"x\":[0.1,0.4],\"y\":[0.2,0.8],\"z\":[0.3,0.9]}",
                    },
                },
                "required": ["time"],
            },
        },
    },


    {
        "type": "function",
        "function": {
            "name": "slice_view_colorbar",
            "description": "返回切片可视化图像（默认带colorbar）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {"anyOf": [{"type": "integer"}, {"type": "number"}], "description": "时间步索引"},
                    "plane": {"type": "string", "enum": ["xy", "yz", "xz"], "description": "切片平面"},
                    "slice_index": {"anyOf": [{"type": "integer"}, {"type": "number"}], "description": "切片索引"},
                    "slice_coord": {
                        "anyOf": [
                            {"type": "number"},
                            {"type": "array", "items": {"type": "number"}},
                            {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                    "z": {"type": "number"},
                                },
                                "additionalProperties": False,
                            },
                        ],
                        "description": "切片坐标(0-1)",
                    },
                    "quantity": {
                        "type": "string",
                        "enum": [
                            "velocity",
                            "vorticity",
                            "velocity_magnitude",
                            "speed",
                            "energy",
                            "kinetic_energy",
                            "vorticity_magnitude",
                            "omega",
                            "curl",
                        ],
                        "description": "物理量（支持别名，会自动归一化）",
                    },
                },
                "required": ["time"],
            },
        },
    },
]


def _tool_slice_stats(
    renderer: RealFluidRenderer,
    time: int,
    plane: str,
    slice_index: int | None,
    quantity: str,
    point_index: list[int] | None,
    point_coord: list[float] | None,
    roi,
) -> dict:
    plane = _normalize_plane(plane)
    if slice_index is None:
        slice_index = _slice_size(renderer, plane) // 2

    slice_vec = None
    omega_z = None
    if quantity == "vorticity":
        omega_x, omega_y, omega_z, omega_mag = _get_vorticity_components(renderer, time)
        field = _get_vorticity_slice(omega_mag, plane, slice_index)
    else:
        slice_vec = _render_velocity_slice(renderer, time, plane, slice_index)
        field = np.linalg.norm(slice_vec, axis=-1)

    roi_bounds = _normalize_roi_indices_2d(plane, field.shape, roi)
    if roi_bounds:
        low0, high0, low1, high1 = roi_bounds
        field_work = field[low0:high0 + 1, low1:high1 + 1]
        if field_work.size == 0:
            return {"error": "roi范围为空"}
    else:
        low0 = low1 = 0
        field_work = field

    mean_val = float(np.mean(field_work))
    std_val = float(np.std(field_work))
    eps = _data_eps(field_work)
    cv_val = float(std_val / mean_val) if mean_val > eps else 0.0
    max_idx_local = np.unravel_index(int(np.argmax(field_work)), field_work.shape)
    min_idx_local = np.unravel_index(int(np.argmin(field_work)), field_work.shape)
    max_idx = (int(max_idx_local[0] + low0), int(max_idx_local[1] + low1))
    min_idx = (int(min_idx_local[0] + low0), int(min_idx_local[1] + low1))
    coords_axis0_full, coords_axis1_full = _coords_for_shape(field.shape[0], field.shape[1])
    coords_axis0_roi, coords_axis1_roi = _coords_for_roi(
        field.shape,
        low0,
        low1,
        field_work.shape[0],
        field_work.shape[1],
    )
    axis0_label, axis1_label = {"xy": ("x", "y"), "yz": ("y", "z"), "xz": ("x", "z")}[plane]
    max_coord = {
        axis0_label: float(coords_axis0_roi[max_idx_local[0], max_idx_local[1]]),
        axis1_label: float(coords_axis1_roi[max_idx_local[0], max_idx_local[1]]),
    }
    min_coord = {
        axis0_label: float(coords_axis0_roi[min_idx_local[0], min_idx_local[1]]),
        axis1_label: float(coords_axis1_roi[min_idx_local[0], min_idx_local[1]]),
    }

    point_idx = _normalize_point_indices(renderer, plane, point_index, point_coord)
    point_info = {}
    if point_idx is not None:
        axis0 = max(0, min(field.shape[0] - 1, int(point_idx[0])))
        axis1 = max(0, min(field.shape[1] - 1, int(point_idx[1])))
        point_info = {
            "point_index": [axis0, axis1],
            "point_coord": {
                axis0_label: float(coords_axis0_full[axis0, axis1]),
                axis1_label: float(coords_axis1_full[axis0, axis1]),
            },
            "point_value": float(field[axis0, axis1]),
        }
        point_info.update(_compute_point_gradient_stats(field, axis0, axis1))


    result = {
        "time": time,
        "plane": plane,
        "slice_index": int(slice_index),
        "mean": mean_val,
        "std": std_val,
        "cv": cv_val,
        "max_value": float(field[max_idx]),
        "min_value": float(field[min_idx]),
        "max_index": [int(max_idx[0]), int(max_idx[1])],
        "min_index": [int(min_idx[0]), int(min_idx[1])],
        "max_coord": max_coord,
        "min_coord": min_coord,
    }
    if point_info:
        result.update(point_info)

    if quantity == "velocity":
        # Q14: triangle comparison (slice-only)
        mask_ul = coords_axis1_roi > coords_axis0_roi
        mask_lr = coords_axis1_roi < coords_axis0_roi
        mean_ul = float(np.mean(field_work[mask_ul])) if np.any(mask_ul) else 0.0
        mean_lr = float(np.mean(field_work[mask_lr])) if np.any(mask_lr) else 0.0
        diff = abs(mean_ul - mean_lr)
        result.update({
            "triangle_means": {"upper_left": mean_ul, "lower_right": mean_lr},
            "triangle_diff": diff,
        })

    if quantity == "velocity" and plane == "xy":
        if slice_vec is None:
            slice_vec = _render_velocity_slice(renderer, time, plane, slice_index)
        energy_full = 0.5 * np.sum(slice_vec ** 2, axis=-1)
        full_h, full_w = energy_full.shape
        if roi_bounds:
            roi_h = high0 - low0 + 1
            roi_w = high1 - low1 + 1
            coords0 = (np.arange(low0, low0 + roi_h) + 0.5) / float(full_h)
            coords1 = (np.arange(low1, low1 + roi_w) + 0.5) / float(full_w)
        else:
            coords0 = (np.arange(full_h) + 0.5) / float(full_h)
            coords1 = (np.arange(full_w) + 0.5) / float(full_w)

        # Q12: energy range in central region (slice-only)
        if roi_bounds:
            energy = energy_full[low0:high0 + 1, low1:high1 + 1]
            stats = {
                "mean": float(np.mean(energy)) if energy.size else 0.0,
                "std": float(np.std(energy)) if energy.size else 0.0,
                "min": float(np.min(energy)) if energy.size else 0.0,
                "max": float(np.max(energy)) if energy.size else 0.0,
            }
            x_low = (low0 + 0.5) / float(full_h)
            x_high = (high0 + 0.5) / float(full_h)
            y_low = (low1 + 0.5) / float(full_w)
            y_high = (high1 + 0.5) / float(full_w)
        else:
            energy = energy_full
            coords0 = (np.arange(full_h) + 0.5) / float(full_h)
            coords1 = (np.arange(full_w) + 0.5) / float(full_w)
            coords_x = coords0[:, None]
            coords_y = coords1[None, :]
            x_low, x_high, _, _ = _weighted_bounds(coords_x, energy)
            y_low, y_high, _, _ = _weighted_bounds(coords_y, energy)
            x_low_idx = _coord_to_index_low(x_low, full_h)
            x_high_idx = _coord_to_index_high(x_high, full_h)
            y_low_idx = _coord_to_index_low(y_low, full_w)
            y_high_idx = _coord_to_index_high(y_high, full_w)
            if x_high_idx < x_low_idx or y_high_idx < y_low_idx:
                stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            else:
                center_energy = energy[x_low_idx:x_high_idx + 1, y_low_idx:y_high_idx + 1]
                stats = {
                    "mean": float(np.mean(center_energy)) if center_energy.size else 0.0,
                    "std": float(np.std(center_energy)) if center_energy.size else 0.0,
                    "min": float(np.min(center_energy)) if center_energy.size else 0.0,
                    "max": float(np.max(center_energy)) if center_energy.size else 0.0,
                }
        eps_energy = _data_eps(energy)
        if stats["mean"] <= eps_energy:
            range_ratio = 0.0
        else:
            range_ratio = (stats["max"] - stats["min"]) / stats["mean"]
        result.update({
            "energy_center_bounds": {"x": [x_low, x_high], "y": [y_low, y_high]},
            "energy_range_ratio": float(range_ratio),
        })

    if quantity == "vorticity" and plane == "xy" and omega_z is not None:
        omega_slice = _get_vorticity_slice(omega_z, plane, slice_index)
        if roi_bounds:
            omega_slice = omega_slice[low0:high0 + 1, low1:high1 + 1]
        result.update({
            "omega_z_stats": {
                "mean": float(np.mean(omega_slice)),
                "std": float(np.std(omega_slice)),
            }
        })

    return result


def _tool_slice_compare(
    renderer: RealFluidRenderer,
    time_a: int,
    time_b: int,
    time_indices: list[int] | None,
    plane: str,
    slice_index: int | None,
    quantity: str,
    roi,
) -> dict:
    plane = _normalize_plane(plane)
    if slice_index is None:
        slice_index = _slice_size(renderer, plane) // 2
    roi_bounds = _normalize_roi_indices_2d(plane, _plane_shape(renderer, plane), roi)

    def _slice_field(time_idx: int, apply_roi: bool = True) -> np.ndarray:
        if quantity == "vorticity":
            _, _, _, omega_mag = _get_vorticity_components(renderer, time_idx)
            field = _get_vorticity_slice(omega_mag, plane, slice_index)
        else:
            field = _render_speed_slice(renderer, time_idx, plane, slice_index)
        if apply_roi and roi_bounds:
            low0, high0, low1, high1 = roi_bounds
            field = field[low0:high0 + 1, low1:high1 + 1]
        return field

    field_a = _slice_field(time_a)
    field_b = _slice_field(time_b)
    if field_a.size == 0 or field_b.size == 0:
        return {"error": "roi范围为空"}
    corr = _safe_corr(field_a.flatten(), field_b.flatten())

    mean_a = float(np.mean(field_a))
    mean_b = float(np.mean(field_b))
    result = {
        "plane": plane,
        "slice_index": int(slice_index),
        "time_a": time_a,
        "time_b": time_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "correlation": corr,
    }

    if quantity == "velocity":
        slice_vec_a = _render_velocity_slice(renderer, time_a, plane, slice_index)
        slice_vec_b = _render_velocity_slice(renderer, time_b, plane, slice_index)

        if plane == "xy":
            change = float(mean_b - mean_a)
            result.update({
                "mean_change": change,
            })

            # Q20: energy concentration trend (slice-only)
            energy_a = 0.5 * np.sum(slice_vec_a ** 2, axis=-1)
            energy_b = 0.5 * np.sum(slice_vec_b ** 2, axis=-1)
            if roi_bounds:
                low0, high0, low1, high1 = roi_bounds
                energy_a = energy_a[low0:high0 + 1, low1:high1 + 1]
                energy_b = energy_b[low0:high0 + 1, low1:high1 + 1]

            def _concentration(data: np.ndarray) -> float:
                mean_val = float(np.mean(data))
                if mean_val <= _data_eps(data):
                    return 0.0
                return float(np.std(data) / mean_val)

            cv_a = _concentration(energy_a)
            cv_b = _concentration(energy_b)
            change_cv = cv_b - cv_a
            result.update({
                "energy_cv_change": float(change_cv),
            })

        # Q9: energy centroid shift (slice-only, all planes)
        full_shape = _plane_shape(renderer, plane)
        coords_x, coords_y = _coords_for_shape(full_shape[0], full_shape[1])
        energy_a_c = np.sum(slice_vec_a ** 2, axis=-1)
        energy_b_c = np.sum(slice_vec_b ** 2, axis=-1)
        if roi_bounds:
            low0, high0, low1, high1 = roi_bounds
            mask = np.zeros_like(energy_a_c, dtype=bool)
            mask[low0:high0 + 1, low1:high1 + 1] = True
            energy_a_c = np.where(mask, energy_a_c, 0.0)
            energy_b_c = np.where(mask, energy_b_c, 0.0)
        thresh_a = float(np.mean(energy_a_c) + np.std(energy_a_c))
        thresh_b = float(np.mean(energy_b_c) + np.std(energy_b_c))
        weights_a = np.where(energy_a_c >= thresh_a, energy_a_c, 0.0)
        weights_b = np.where(energy_b_c >= thresh_b, energy_b_c, 0.0)
        center_a = _weighted_centroid(coords_x, coords_y, weights_a)
        center_b = _weighted_centroid(coords_x, coords_y, weights_b)
        dx = float(center_b[0] - center_a[0])
        dy = float(center_b[1] - center_a[1])
        result.update({
            "energy_center_shift": {"dx": dx, "dy": dy},
        })

    if time_indices and len(time_indices) >= 3:
        t_list = time_indices[:3]
        max_vals = []
        for t_idx in t_list:
            field_t = _slice_field(t_idx)
            max_vals.append(float(np.max(field_t)))
        result.update({
            "max_values": max_vals,
        })

    if quantity == "vorticity" and plane == "xy":
        # Q8: vorticity max shift (slice-only)
        full_shape = _plane_shape(renderer, plane)
        coords_x, coords_y = _coords_for_shape(full_shape[0], full_shape[1])
        field_a_full = _slice_field(time_a, apply_roi=False)
        field_b_full = _slice_field(time_b, apply_roi=False)
        mask = None
        if roi_bounds:
            low0, high0, low1, high1 = roi_bounds
            mask = np.zeros_like(field_a_full, dtype=bool)
            mask[low0:high0 + 1, low1:high1 + 1] = True
            field_a_full = np.where(mask, field_a_full, -np.inf)
            field_b_full = np.where(mask, field_b_full, -np.inf)
        loc_a = np.unravel_index(int(np.argmax(field_a_full)), field_a_full.shape)
        loc_b = np.unravel_index(int(np.argmax(field_b_full)), field_b_full.shape)
        coord_a = np.array([coords_x[loc_a], coords_y[loc_a]])
        coord_b = np.array([coords_x[loc_b], coords_y[loc_b]])
        distance = float(np.linalg.norm(coord_b - coord_a))
        result.update({
            "vort_max_distance": distance,
        })

    return result


def _tool_cube_components(
    renderer: RealFluidRenderer,
    time: int,
    center_index: list[int] | None,
    time_indices: list[int] | None,
    radius: int,
    roi,
) -> dict:
    volume = _get_velocity_volume(renderer, time)
    nx, ny, nz = renderer.num_x, renderer.num_y, renderer.num_z
    roi_bounds = _normalize_roi_indices_3d(renderer, roi)
    if roi_bounds:
        x0, x1, y0, y1, z0, z1 = roi_bounds
        xs = slice(x0, x1 + 1)
        ys = slice(y0, y1 + 1)
        zs = slice(z0, z1 + 1)
        x_idx = (x0 + x1) // 2
        y_idx = (y0 + y1) // 2
        z_idx = (z0 + z1) // 2
    else:
        if center_index is None:
            x_idx = nx // 2
            y_idx = ny // 2
            z_idx = nz // 2
        else:
            x_idx = max(0, min(nx - 1, int(center_index[0])))
            y_idx = max(0, min(ny - 1, int(center_index[1])))
            z_idx = max(0, min(nz - 1, int(center_index[2])))

        xs = slice(max(0, x_idx - radius), min(nx, x_idx + radius + 1))
        ys = slice(max(0, y_idx - radius), min(ny, y_idx + radius + 1))
        zs = slice(max(0, z_idx - radius), min(nz, z_idx + radius + 1))

    cube = volume[xs, ys, zs, :]
    mean_abs = np.mean(np.abs(cube), axis=(0, 1, 2))
    mean_speed = float(np.mean(np.linalg.norm(cube, axis=-1)))
    global_mean_abs = np.mean(np.abs(volume), axis=(0, 1, 2))
    global_mean = float(np.mean(global_mean_abs))
    global_eps = _data_eps(global_mean_abs)
    global_spread = float(np.std(global_mean_abs) / global_mean) if global_mean > global_eps else 0.0

    local_mean = float(np.mean(mean_abs))
    local_eps = _data_eps(mean_abs)
    local_spread = float(np.std(mean_abs) / local_mean) if local_mean > local_eps else 0.0


    center_coord = [
        float(_index_to_coord(int(x_idx), renderer.num_x)),
        float(_index_to_coord(int(y_idx), renderer.num_y)),
        float(_index_to_coord(int(z_idx), renderer.num_z)),
    ]
    result = {
        "time": time,
        "center_coord": center_coord,
        "mean_abs_components": mean_abs.tolist(),
        "mean_speed": mean_speed,
        "local_spread": local_spread,
        "global_spread": global_spread,
        "global_mean_abs_components": global_mean_abs.tolist(),
    }

    if time_indices and len(time_indices) >= 3:
        t_list = time_indices[:3]
        point_values = []
        for t_idx in t_list:
            volume_t = _get_velocity_volume(renderer, t_idx)
            vec = volume_t[x_idx, y_idx, z_idx]
            point_values.append(float(np.linalg.norm(vec)))

        series = []
        for t in range(renderer.num_timesteps):
            volume_t = _get_velocity_volume(renderer, t)
            vec = volume_t[x_idx, y_idx, z_idx]
            series.append(float(np.linalg.norm(vec)))
        series_arr = np.array(series)
        std_series = float(np.std(series_arr))

        result.update({
            "point_values": point_values,
            "point_series_std": std_series,
        })

    return result


def _tool_plane_uniformity(
    renderer: RealFluidRenderer,
    time: int,
    slice_indices: dict | None,
    quantity: str,
    roi,
    plane_rois: dict | None,
) -> dict:
    cv = {}
    planes = ["xy", "yz", "xz"]
    for plane in planes:
        index = None
        if slice_indices and plane in slice_indices:
            index = slice_indices[plane]
        if index is None:
            index = _slice_size(renderer, plane) // 2
        if quantity == "vorticity":
            _, _, _, omega_mag = _get_vorticity_components(renderer, time)
            speed = _get_vorticity_slice(omega_mag, plane, index)
        else:
            speed = _render_speed_slice(renderer, time, plane, index)
        plane_roi = plane_rois.get(plane) if plane_rois else roi
        roi_bounds = _normalize_roi_indices_2d(plane, speed.shape, plane_roi)
        if roi_bounds:
            low0, high0, low1, high1 = roi_bounds
            speed = speed[low0:high0 + 1, low1:high1 + 1]
        mean_val = float(np.mean(speed))
        eps = _data_eps(speed)
        cv_val = float(np.std(speed) / mean_val) if mean_val > eps else float("inf")
        cv[plane] = cv_val

    return {
        "time": time,
        "cv": cv,
    }


def _tool_slice_view_colorbar(
    renderer: RealFluidRenderer,
    time: int,
    plane: str,
    slice_index: int | None,
    quantity: str,
) -> Image.Image:
    plane = _normalize_plane(plane)
    if slice_index is None:
        slice_index = _slice_size(renderer, plane) // 2
    if quantity == "vorticity":
        _, _, _, omega_mag = _get_vorticity_components(renderer, time)
        field = _get_vorticity_slice(omega_mag, plane, slice_index)
    else:
        field = _render_speed_slice(renderer, time, plane, slice_index)

    vmin_val, vmax_val = _compute_vmin_vmax(field, None, None)
    rgb = _field_to_rgb(field, vmin_val, vmax_val, "viridis")
    main_img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    display_size = 128
    main_img = main_img.resize((display_size, display_size), resample=Image.NEAREST)
    bar_width = int(round(main_img.width * 0.06))
    bar_width = max(8, min(16, bar_width))
    if bar_width <= 0:
        return main_img
    bar_img = _build_colorbar(main_img.height, "viridis", bar_width)
    gap = int(round(main_img.width * 0.04))
    gap = max(6, min(12, gap))
    composed = _compose_with_colorbar(main_img, bar_img, vmin_val, vmax_val, title=quantity, gap=gap)
    composed = _pad_to_multiple(composed, 28, fill=(255, 255, 255))
    composed = composed.resize((composed.width * 2, composed.height * 2), resample=Image.NEAREST)
    return composed


def _tool_vorticity_orientation(
    renderer: RealFluidRenderer,
    time: int,
    xy_slice_index: int | None,
    yz_slice_index: int | None,
    xz_slice_index: int | None,
    roi,
) -> dict:
    _, _, _, omega_mag = _get_vorticity_components(renderer, time)
    nx, ny, nz = omega_mag.shape
    xs = (np.arange(nx) + 0.5) / float(nx)
    ys = (np.arange(ny) + 0.5) / float(ny)
    zs = (np.arange(nz) + 0.5) / float(nz)
    xs, ys, zs = np.meshgrid(xs, ys, zs, indexing="ij")
    coords_all = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    weights_all = omega_mag.ravel()
    roi_bounds = _normalize_roi_indices_3d(renderer, roi)
    if roi_bounds:
        x0, x1, y0, y1, z0, z1 = roi_bounds
        roi_mask = np.zeros_like(omega_mag, dtype=bool)
        roi_mask[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1] = True
    else:
        roi_mask = None

    slice_mask = None
    if xy_slice_index is not None:
        mask = np.zeros_like(omega_mag, dtype=bool)
        mask[:, :, int(xy_slice_index)] = True
        slice_mask = mask if slice_mask is None else (slice_mask | mask)
    if yz_slice_index is not None:
        mask = np.zeros_like(omega_mag, dtype=bool)
        mask[int(yz_slice_index), :, :] = True
        slice_mask = mask if slice_mask is None else (slice_mask | mask)
    if xz_slice_index is not None:
        mask = np.zeros_like(omega_mag, dtype=bool)
        mask[:, int(xz_slice_index), :] = True
        slice_mask = mask if slice_mask is None else (slice_mask | mask)

    if roi_mask is not None and slice_mask is not None:
        mask_all = roi_mask & slice_mask
    elif roi_mask is not None:
        mask_all = roi_mask
    else:
        mask_all = slice_mask

    if mask_all is not None:
        mask_flat = mask_all.ravel()
        if not np.any(mask_flat):
            return {"error": "roi范围为空"}
        coords_all = coords_all[mask_flat]
        weights_all = weights_all[mask_flat]

    def weighted_variances(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            return np.zeros(3, dtype=np.float64)
        mean = np.sum(points * weights[:, None], axis=0) / weight_sum
        var = np.sum(weights[:, None] * (points - mean) ** 2, axis=0) / weight_sum
        return var

    baseline_vars = weighted_variances(coords_all, weights_all)
    baseline_sorted = np.sort(baseline_vars)
    baseline_eps = _data_eps(baseline_vars)
    baseline_ratio = (
        float(baseline_sorted[-1] / baseline_sorted[-2])
        if baseline_sorted[-2] > baseline_eps
        else 0.0
    )

    if mask_all is None:
        threshold = float(np.mean(omega_mag) + np.std(omega_mag))
        mask_flat = (omega_mag >= threshold).ravel()
        points = coords_all[mask_flat]
        weights = weights_all[mask_flat]
    else:
        region_vals = omega_mag[mask_all]
        threshold = float(np.mean(region_vals) + np.std(region_vals))
        mask_flat = (omega_mag >= threshold).ravel()[mask_all.ravel()]
        points = coords_all[mask_flat]
        weights = weights_all[mask_flat]

    if points.shape[0] == 0:
        variances = [0.0, 0.0, 0.0]
        anisotropy_ratio = 0.0
    else:
        variances = weighted_variances(points, weights)
        sorted_vars = np.sort(variances)
        anisotropy_ratio = (
            float(sorted_vars[-1] / sorted_vars[-2])
            if sorted_vars[-2] > _data_eps(sorted_vars)
            else 0.0
        )

    result = {
        "time": time,
        "xy_slice_index": xy_slice_index,
        "yz_slice_index": yz_slice_index,
        "variances": [float(v) for v in variances],
        "anisotropy_ratio": float(anisotropy_ratio),
        "baseline_ratio": float(baseline_ratio),
        "threshold": threshold,
    }

    if xy_slice_index is not None and (xz_slice_index is not None or yz_slice_index is not None):
        result.update(_compute_vortex_displacement(
            renderer=renderer,
            time=time,
            xy_slice_index=xy_slice_index,
            xz_slice_index=xz_slice_index,
            yz_slice_index=yz_slice_index,
        ))

    return result


_SAFE_TOOL_KEYS = {
    "slice_stats": {
        "time",
        "plane",
        "slice_index",
        "mean",
        "std",
        "cv",
        "max_value",
        "min_value",
        "max_index",
        "min_index",
        "max_coord",
        "min_coord",
        "energy_range_ratio",
        "triangle_means",
        "triangle_diff",
        "omega_z_stats",
        "point_index",
        "point_coord",
        "point_value",
        "grad_x",
        "grad_y",
        "grad_median",
        "grad_std",
    },
    "slice_compare": {
        "plane",
        "slice_index",
        "time_a",
        "time_b",
        "mean_a",
        "mean_b",
        "correlation",
        "mean_change",
        "vort_max_distance",
        "energy_center_shift",
        "energy_cv_change",
        "max_values",
    },
    "cube_components": {
        "time",
        "center_coord",
        "mean_abs_components",
        "mean_speed",
        "local_spread",
        "global_spread",
        "point_values",
        "point_series_std",
    },
    "plane_uniformity": {
        "time",
        "cv",
    },
    "vorticity_orientation": {
        "variances",
        "anisotropy_ratio",
        "baseline_ratio",
        "displacement",
        "disp_mag",
        "disp_threshold",
    },
}


def _sanitize_tool_result(tool_name: str, result: dict) -> dict:
    safe_keys = _SAFE_TOOL_KEYS.get(tool_name)
    if safe_keys is None:
        return dict(result)
    return {k: v for k, v in result.items() if k in safe_keys}


def _compact_tool_result(tool_name: str, result: dict) -> dict:
    if not isinstance(result, dict) or "error" in result:
        return result

    if tool_name == "slice_stats":
        return _sanitize_tool_result(tool_name, result)

    if tool_name == "slice_compare":
        keep = {
            "plane",
            "slice_index",
            "time_a",
            "time_b",
            "mean_a",
            "mean_b",
            "correlation",
            "mean_change",
            "vort_max_distance",
            "energy_center_shift",
            "energy_cv_change",
            "max_values",
        }
        compact = {k: result.get(k) for k in keep if k in result}
        return compact

    if tool_name == "cube_components":
        keep = {
            "time",
            "center_coord",
            "mean_abs_components",
            "mean_speed",
            "local_spread",
            "global_spread",
            "point_values",
            "point_series_std",
        }
        compact = {k: result.get(k) for k in keep if k in result}
        return compact

    if tool_name == "plane_uniformity":
        keep = {"time", "cv"}
        compact = {k: result.get(k) for k in keep if k in result}
        return compact

    if tool_name == "vorticity_orientation":
        keep = {
            "variances",
            "anisotropy_ratio",
            "baseline_ratio",
            "displacement",
            "disp_mag",
            "disp_threshold",
        }
        compact = {k: result.get(k) for k in keep if k in result}
        return compact

    return _sanitize_tool_result(tool_name, result)


def execute_tool(
    renderer: RealFluidRenderer,
    tool_name: str,
    args: dict,
    sanitize: bool = True,
):


    try:
        if not isinstance(args, dict):
            return {"error": "工具参数必须是JSON对象"}

        def _finalize(result):
            if not sanitize:
                return result
            if not isinstance(result, dict):
                return result
            if "error" in result:
                return result
            compact = _compact_tool_result(tool_name, result)
            if not isinstance(compact, dict):
                return compact
            def _shift_time_value(value):
                if isinstance(value, (int, np.integer)):
                    return int(value) + 1
                if isinstance(value, float):
                    if abs(value - round(value)) <= 1e-6:
                        return int(round(value)) + 1
                    return value + 1.0
                return value
            def _shift_index_value(value):
                if isinstance(value, (int, np.integer)):
                    return int(value) + 1
                if isinstance(value, float):
                    if abs(value - round(value)) <= 1e-6:
                        return int(round(value)) + 1
                    return value
                if isinstance(value, (list, tuple)):
                    return [_shift_index_value(item) for item in value]
                return value
            for key in ("time", "time_a", "time_b"):
                if key in compact:
                    compact[key] = _shift_time_value(compact[key])
            if "time_indices" in compact and isinstance(compact["time_indices"], (list, tuple)):
                compact["time_indices"] = [
                    _shift_time_value(v) for v in compact["time_indices"]
                ]
            for key in (
                "slice_index",
                "point_index",
                "max_index",
                "min_index",
                "center_index",
                "xy_slice_index",
                "yz_slice_index",
                "xz_slice_index",
            ):
                if key in compact:
                    compact[key] = _shift_index_value(compact[key])
            return compact


        if tool_name == "slice_stats":
            if args.get("region") is not None or args.get("quadrant") is not None:
                return {"error": "region/quadrant 已禁用，请使用 roi 指定区域"}
            plane, slice_index = _normalize_slice_indices(
                renderer,
                args.get("plane", "xy"),
                args.get("slice_index"),
                args.get("slice_coord"),
            )
            quantity = _normalize_quantity(args.get("quantity"))
            return _finalize(_tool_slice_stats(
                renderer=renderer,
                time=_normalize_time_index(renderer, args.get("time", 0)),
                plane=plane,
                slice_index=slice_index,
                quantity=quantity,
                point_index=args.get("point_index"),
                point_coord=args.get("point_coord"),
                roi=args.get("roi"),
            ))


        if tool_name == "slice_compare":
            if args.get("region") is not None or args.get("quadrant") is not None:
                return {"error": "region/quadrant 已禁用，请使用 roi 指定区域"}
            plane, slice_index = _normalize_slice_indices(
                renderer,
                args.get("plane", "xy"),
                args.get("slice_index"),
                args.get("slice_coord"),
            )
            time_indices = _normalize_time_indices(renderer, args.get("time_indices"))
            quantity = _normalize_quantity(args.get("quantity"))
            return _finalize(_tool_slice_compare(
                renderer=renderer,
                time_a=_normalize_time_index(renderer, args.get("time_a", 0)),
                time_b=_normalize_time_index(renderer, args.get("time_b", 0)),
                time_indices=time_indices,
                plane=plane,
                slice_index=slice_index,
                quantity=quantity,
                roi=args.get("roi"),
            ))


        if tool_name == "cube_components":
            center_index = _normalize_center_indices(
                renderer,
                args.get("center_index"),
                args.get("center_coord"),
            )
            time_indices = _normalize_time_indices(renderer, args.get("time_indices"))
            return _finalize(_tool_cube_components(
                renderer=renderer,
                time=_normalize_time_index(renderer, args.get("time", 0)),
                center_index=center_index,
                time_indices=time_indices,
                radius=_normalize_radius_index(renderer, args.get("radius", 1)),
                roi=args.get("roi"),
            ))


        if tool_name == "plane_uniformity":
            raw_slice_indices = args.get("slice_indices")
            raw_slice_coords = args.get("slice_coords")
            raw_plane_rois = args.get("plane_rois")
            if isinstance(raw_plane_rois, list):
                plane_roi_dict = {}
                extra_slice_indices = {}
                extra_slice_coords = {}
                for item in raw_plane_rois:
                    if not isinstance(item, dict):
                        continue
                    plane = _normalize_plane(item.get("plane"))
                    if item.get("roi") is not None:
                        plane_roi_dict[plane] = item.get("roi")
                    if item.get("slice_index") is not None:
                        extra_slice_indices[plane] = item.get("slice_index")
                    if item.get("slice_coord") is not None:
                        extra_slice_coords[plane] = item.get("slice_coord")
                if extra_slice_indices:
                    if not isinstance(raw_slice_indices, dict):
                        raw_slice_indices = raw_slice_indices or {}
                    if isinstance(raw_slice_indices, dict):
                        raw_slice_indices = {**raw_slice_indices, **extra_slice_indices}
                if extra_slice_coords:
                    if not isinstance(raw_slice_coords, dict):
                        raw_slice_coords = raw_slice_coords or {}
                    if isinstance(raw_slice_coords, dict):
                        raw_slice_coords = {**raw_slice_coords, **extra_slice_coords}
                raw_plane_rois = plane_roi_dict or None

            slice_indices = _normalize_plane_slices(
                renderer,
                raw_slice_indices,
                raw_slice_coords,
            )
            quantity = _normalize_quantity(args.get("quantity"))
            return _finalize(_tool_plane_uniformity(
                renderer=renderer,
                time=_normalize_time_index(renderer, args.get("time", 0)),
                slice_indices=slice_indices,
                quantity=quantity,
                roi=args.get("roi"),
                plane_rois=raw_plane_rois,
            ))


        if tool_name == "slice_view_colorbar":
            plane, slice_index = _normalize_slice_indices(
                renderer,
                args.get("plane", "xy"),
                args.get("slice_index"),
                args.get("slice_coord"),
            )
            quantity = _normalize_quantity(args.get("quantity"))
            return _finalize(_tool_slice_view_colorbar(
                renderer=renderer,
                time=_normalize_time_index(renderer, args.get("time", 0)),
                plane=plane,
                slice_index=slice_index,
                quantity=quantity,
            ))


        if tool_name == "vorticity_orientation":
            xy_coord = _extract_numeric(args.get("xy_slice_coord"), preferred_keys=["z"])
            yz_coord = _extract_numeric(args.get("yz_slice_coord"), preferred_keys=["x"])
            xz_coord = _extract_numeric(args.get("xz_slice_coord"), preferred_keys=["y"])
            xy_idx = _extract_numeric(args.get("xy_slice_index"))
            yz_idx = _extract_numeric(args.get("yz_slice_index"))
            xz_idx = _extract_numeric(args.get("xz_slice_index"))
            if xy_idx is None and xy_coord is not None:
                xy_idx = _coord_to_index(xy_coord, renderer.num_z)
            if yz_idx is None and yz_coord is not None:
                yz_idx = _coord_to_index(yz_coord, renderer.num_x)
            if xz_idx is None and xz_coord is not None:
                xz_idx = _coord_to_index(xz_coord, renderer.num_y)
            xy_norm = _normalize_index_value(xy_idx, renderer.num_z) if xy_idx is not None else None
            yz_norm = _normalize_index_value(yz_idx, renderer.num_x) if yz_idx is not None else None
            xz_norm = _normalize_index_value(xz_idx, renderer.num_y) if xz_idx is not None else None
            return _finalize(_tool_vorticity_orientation(
                renderer=renderer,
                time=_normalize_time_index(renderer, args.get("time", 0)),
                xy_slice_index=xy_norm,
                yz_slice_index=yz_norm,
                xz_slice_index=xz_norm,
                roi=args.get("roi"),
            ))


        return {"error": f"未知工具: {tool_name}"}

    except Exception as e:
        return {"error": str(e)}
