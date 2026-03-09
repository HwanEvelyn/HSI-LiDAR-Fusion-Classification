from __future__ import annotations

import logging
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from matplotlib.path import Path as MplPath
from skimage.draw import polygon as draw_polygon

from .patch_dataset import IndexItem


class _SuppressGdalNoDataFilter(logging.Filter):
    """
    过滤掉 tifffile 模块的 GDAL_NODATA 错误信息
    """
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "parsing GDAL_NODATA tag raised ValueError" not in message


_TIFFFILE_LOGGER = logging.getLogger("tifffile")
_TIFFFILE_LOGGER.addFilter(_SuppressGdalNoDataFilter())


@dataclass
class MatData:
    hsi: np.ndarray
    lidar: np.ndarray
    gt: np.ndarray
    train_gt: np.ndarray
    test_gt: np.ndarray


def _parse_roi_txt(txt_path: Path, shape: tuple[int, int]) -> np.ndarray:
    label_map = np.zeros(shape, dtype=np.int64)
    class_id = 0

    for raw_line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("; ROI name:"):
            class_id += 1
            continue
        if line.startswith(";") or class_id == 0:
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        x = int(parts[1])
        y = int(parts[2])
        if 1 <= x <= shape[1] and 1 <= y <= shape[0]:
            label_map[y - 1, x - 1] = class_id

    return label_map


def _scan_envi_roi_headers(blob: bytes) -> list[int]:
    headers = []
    for offset in range(8, len(blob) - 220, 4):
        name_len_a = struct.unpack(">I", blob[offset : offset + 4])[0]
        name_len_b = struct.unpack(">I", blob[offset + 4 : offset + 8])[0]
        if name_len_a != name_len_b or not (1 <= name_len_a <= 40):
            continue

        name = blob[offset + 8 : offset + 8 + name_len_a]
        if len(name) != name_len_a:
            continue
        if all((48 <= ch <= 57) or (65 <= ch <= 90) or (97 <= ch <= 122) or ch == 95 for ch in name):
            headers.append(offset)
    return headers


def _parse_envi_roi(blob: bytes) -> list[dict[str, object]]:
    headers = _scan_envi_roi_headers(blob)
    if not headers:
        raise ValueError("No ENVI ROI headers found")

    rois: list[dict[str, object]] = []
    boundaries = headers + [len(blob)]

    for start, next_start in zip(boundaries[:-1], boundaries[1:]):
        name_len = struct.unpack(">I", blob[start : start + 4])[0]
        name = blob[start + 8 : start + 8 + name_len].decode("ascii")
        fields_offset = start + ((8 + name_len + 3) // 4) * 4
        npts = struct.unpack(">I", blob[fields_offset + 12 : fields_offset + 16])[0]
        first_poly = fields_offset + 180

        polygons: list[tuple[np.ndarray, np.ndarray]] = []
        pos = first_poly
        while pos < next_start:
            n_vertices, roi_type = struct.unpack(">II", blob[pos : pos + 8])
            if roi_type != 4 or not (3 <= n_vertices <= 1024):
                raise ValueError(f"Unexpected ROI block in {name} at offset {pos}")

            pos += 8
            xs = np.asarray(struct.unpack(f">{n_vertices}f", blob[pos : pos + 4 * n_vertices]), dtype=np.float32)
            pos += 4 * n_vertices
            ys = np.asarray(struct.unpack(f">{n_vertices}f", blob[pos : pos + 4 * n_vertices]), dtype=np.float32)
            pos += 4 * n_vertices
            polygons.append((xs, ys))

        rois.append({"name": name, "npts": int(npts), "polygons": polygons})

    return rois


def _polygon_pixels(
    xs: np.ndarray,
    ys: np.ndarray,
    shape: tuple[int, int],
    mode: str,
) -> set[tuple[int, int]]:
    height, width = shape
    xmin = max(1, int(np.floor(xs.min())) + 1)
    xmax = min(width, int(np.ceil(xs.max())))
    ymin = max(1, int(np.floor(ys.min())) + 1)
    ymax = min(height, int(np.ceil(ys.max())))
    if xmin > xmax or ymin > ymax:
        return set()

    if mode == "skimage":
        rr, cc = draw_polygon(ys - 1e-3, xs - 1e-3, shape=shape)
        return set(zip(rr.tolist(), cc.tolist()))

    xpix = np.arange(xmin, xmax + 1, dtype=np.float32)
    ypix = np.arange(ymin, ymax + 1, dtype=np.float32)
    xx, yy = np.meshgrid(xpix, ypix)
    if mode.startswith("half"):
        points = np.column_stack([xx.ravel() - 0.5, yy.ravel() - 0.5])
    else:
        points = np.column_stack([xx.ravel(), yy.ravel()])

    inside = MplPath(np.column_stack([xs, ys])).contains_points(points)
    if mode.endswith("boundary"):
        boundary = np.zeros(len(points), dtype=bool)
        verts = np.column_stack([xs, ys])
        for idx in range(len(verts) - 1):
            a = verts[idx]
            b = verts[idx + 1]
            ab = b - a
            ap = points - a
            cross = np.abs(ab[0] * ap[:, 1] - ab[1] * ap[:, 0])
            dot = ap[:, 0] * ab[0] + ap[:, 1] * ab[1]
            sq = ab[0] * ab[0] + ab[1] * ab[1]
            boundary |= (cross < 1e-6) & (dot >= -1e-6) & (dot <= sq + 1e-6)
        inside |= boundary

    rows = yy.ravel()[inside].astype(np.int64) - 1
    cols = xx.ravel()[inside].astype(np.int64) - 1
    return set(zip(rows.tolist(), cols.tolist()))


def _rasterize_polygons(
    polygons: list[tuple[np.ndarray, np.ndarray]],
    shape: tuple[int, int],
    expected_npts: int,
) -> tuple[np.ndarray, int]:
    candidate_modes = ["center", "half", "half_boundary", "skimage"]
    best_mask = None
    best_count = None
    best_diff = None

    for mode in candidate_modes:
        coords: set[tuple[int, int]] = set()
        for xs, ys in polygons:
            coords |= _polygon_pixels(xs, ys, shape, mode)

        mask = np.zeros(shape, dtype=bool)
        if coords:
            rr, cc = zip(*coords)
            mask[np.asarray(rr), np.asarray(cc)] = True

        count = int(mask.sum())
        diff = abs(count - expected_npts)
        if best_diff is None or diff < best_diff:
            best_mask = mask
            best_count = count
            best_diff = diff

    assert best_mask is not None and best_count is not None and best_diff is not None
    tolerance = max(2, int(round(0.02 * expected_npts)))
    if best_diff > tolerance:
        raise ValueError(
            f"ROI rasterization mismatch exceeds tolerance: expected {expected_npts}, got {best_count}"
        )
    return best_mask, best_count


def _parse_roi_bytes(blob: bytes, shape: tuple[int, int]) -> np.ndarray:
    label_map = np.zeros(shape, dtype=np.int64)
    rois = _parse_envi_roi(blob)

    for class_id, roi in enumerate(rois, start=1):
        mask, observed = _rasterize_polygons(roi["polygons"], shape, expected_npts=int(roi["npts"]))
        label_map[mask] = class_id

    return label_map


def _parse_roi_file(roi_path: Path, shape: tuple[int, int]) -> np.ndarray:
    return _parse_roi_bytes(roi_path.read_bytes(), shape)


def _parse_roi_from_zip(zip_path: Path, shape: tuple[int, int]) -> np.ndarray:
    with zipfile.ZipFile(zip_path, "r") as archive:
        roi_names = [name for name in archive.namelist() if name.lower().endswith(".roi")]
        if len(roi_names) != 1:
            raise ValueError(f"Expected exactly one ROI file in {zip_path}, found {roi_names}")
        blob = archive.read(roi_names[0])
    return _parse_roi_bytes(blob, shape)


def load_houston_hl(data_root: str | Path) -> MatData:
    root = Path(data_root)
    casi_path = root / "2013_IEEE_GRSS_DF_Contest_CASI.tif"
    lidar_path = root / "2013_IEEE_GRSS_DF_Contest_LiDAR.tif"
    tr_txt_path = root / "2013_IEEE_GRSS_DF_Contest_Samples_TR.txt"
    tr_roi_path = root / "2013_IEEE_GRSS_DF_Contest_Samples_TR.roi"
    va_zip_path = root / "2013_IEEE_GRSS_DF_Contest_Samples_VA.zip"

    for path in [casi_path, lidar_path, tr_txt_path, tr_roi_path, va_zip_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required Houston 2013 file not found: {path}")

    hsi = tifffile.imread(casi_path).astype(np.float32)
    lidar = tifffile.imread(lidar_path).astype(np.float32)

    if hsi.ndim != 3:
        raise ValueError(f"Expected HSI cube with shape (H, W, C), got {hsi.shape}")
    if lidar.ndim != 2:
        raise ValueError(f"Expected LiDAR image with shape (H, W), got {lidar.shape}")
    if hsi.shape[:2] != lidar.shape:
        raise ValueError(f"HSI and LiDAR spatial shapes do not match: {hsi.shape[:2]} vs {lidar.shape}")

    shape = hsi.shape[:2]
    train_gt = _parse_roi_txt(tr_txt_path, shape)

    # Use the ROI file as a consistency check for the training labels before trusting VA.roi.
    train_gt_from_roi = _parse_roi_file(tr_roi_path, shape)
    for class_id in np.unique(train_gt)[1:]:
        txt_count = int((train_gt == class_id).sum())
        roi_count = int((train_gt_from_roi == class_id).sum())
        if abs(txt_count - roi_count) > max(2, int(round(0.02 * txt_count))):
            raise ValueError(
                f"Training TXT/ROI mismatch for class {class_id}: txt={txt_count}, roi={roi_count}"
            )

    test_gt = _parse_roi_from_zip(va_zip_path, shape)
    gt = np.where(train_gt > 0, train_gt, test_gt)

    return MatData(
        hsi=hsi,
        lidar=lidar,
        gt=gt,
        train_gt=train_gt,
        test_gt=test_gt,
    )


def build_official_houston_split(mat_data: MatData) -> tuple[list[IndexItem], list[IndexItem], int]:
    train_coords = np.argwhere(mat_data.train_gt > 0)
    test_coords = np.argwhere(mat_data.test_gt > 0)

    train_items = [
        IndexItem(int(r), int(c), int(mat_data.train_gt[r, c]) - 1)
        for r, c in train_coords
    ]
    test_items = [
        IndexItem(int(r), int(c), int(mat_data.test_gt[r, c]) - 1)
        for r, c in test_coords
    ]

    num_classes = int(max(mat_data.train_gt.max(), mat_data.test_gt.max()))
    return train_items, test_items, num_classes
