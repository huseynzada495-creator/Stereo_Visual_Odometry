from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml
import cv2


@dataclass
class StereoCalibration:
    cam0: dict
    cam1: dict
    K0_raw: np.ndarray
    K1_raw: np.ndarray
    D0: np.ndarray
    D1: np.ndarray
    T_cam1_cam0: np.ndarray
    R_cam1_cam0: np.ndarray
    t_cam1_cam0: np.ndarray
    baseline_raw: float


@dataclass
class FisheyeRectifier:
    image_size: tuple  # (width, height)
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    map1x: np.ndarray
    map1y: np.ndarray
    map2x: np.ndarray
    map2y: np.ndarray
    K_rect: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float


def _as_contiguous_float64(arr, shape=None):
    arr = np.array(arr, dtype=np.float64, copy=True)
    if shape is not None:
        arr = arr.reshape(shape)
    return np.ascontiguousarray(arr)


def _camera_matrix(intrinsics):
    if len(intrinsics) != 4:
        raise ValueError(f"Expected 4 intrinsics [fx, fy, cx, cy], got {intrinsics}")

    fx, fy, cx, cy = [float(v) for v in intrinsics]
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return np.ascontiguousarray(K)


def load_camchain(camchain_file: Path):
    camchain_file = Path(camchain_file)
    with open(camchain_file, "r", encoding="utf-8") as f:
        calib = yaml.safe_load(f)
    return calib


def get_camera_imu_transform(calib, cam_key="cam0") -> np.ndarray:
    """
    Return T_cam_imu as a clean 4x4 float64 matrix.
    """
    if cam_key not in calib:
        raise KeyError(f"{cam_key} not found in calibration")

    cam = calib[cam_key]
    if "T_cam_imu" not in cam:
        raise KeyError(f"{cam_key} does not contain T_cam_imu")

    T_cam_imu = _as_contiguous_float64(cam["T_cam_imu"], shape=(4, 4))
    return T_cam_imu


def load_kalibr_camchain(camchain_file: Path) -> StereoCalibration:
    """
    Load Kalibr camchain.yaml for a stereo fisheye rig.
    Assumes cam1['T_cn_cnm1'] is the transform from cam0 to cam1 convention
    used by Kalibr exports for chained cameras.
    """
    calib = load_camchain(camchain_file)

    if "cam0" not in calib or "cam1" not in calib:
        raise KeyError("camchain file must contain 'cam0' and 'cam1'")

    cam0 = calib["cam0"]
    cam1 = calib["cam1"]

    for key in ["intrinsics", "distortion_coeffs"]:
        if key not in cam0:
            raise KeyError(f"cam0 missing key: {key}")
        if key not in cam1:
            raise KeyError(f"cam1 missing key: {key}")

    if "T_cn_cnm1" not in cam1:
        raise KeyError("cam1 missing key: T_cn_cnm1")

    # Validate distortion model if present
    if "distortion_model" in cam0 and cam0["distortion_model"] != "equidistant":
        print(f"Warning: cam0 distortion_model is {cam0['distortion_model']}, expected 'equidistant'")
    if "distortion_model" in cam1 and cam1["distortion_model"] != "equidistant":
        print(f"Warning: cam1 distortion_model is {cam1['distortion_model']}, expected 'equidistant'")

    K0_raw = _camera_matrix(cam0["intrinsics"])
    K1_raw = _camera_matrix(cam1["intrinsics"])

    D0 = _as_contiguous_float64(cam0["distortion_coeffs"], shape=(4, 1))
    D1 = _as_contiguous_float64(cam1["distortion_coeffs"], shape=(4, 1))

    T_cam1_cam0 = _as_contiguous_float64(cam1["T_cn_cnm1"], shape=(4, 4))
    R_cam1_cam0 = _as_contiguous_float64(T_cam1_cam0[:3, :3], shape=(3, 3))
    t_cam1_cam0 = _as_contiguous_float64(T_cam1_cam0[:3, 3], shape=(3, 1))

    baseline_raw = float(np.linalg.norm(t_cam1_cam0))

    return StereoCalibration(
        cam0=cam0,
        cam1=cam1,
        K0_raw=K0_raw,
        K1_raw=K1_raw,
        D0=D0,
        D1=D1,
        T_cam1_cam0=T_cam1_cam0,
        R_cam1_cam0=R_cam1_cam0,
        t_cam1_cam0=t_cam1_cam0,
        baseline_raw=baseline_raw,
    )


def build_fisheye_rectifier(calib: StereoCalibration, image_size: tuple) -> FisheyeRectifier:
    """
    Build stereo fisheye rectification maps.

    image_size must be (width, height).
    """
    if len(image_size) != 2:
        raise ValueError(f"image_size must be (width, height), got {image_size}")

    image_size = (int(image_size[0]), int(image_size[1]))

    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        calib.K0_raw,
        calib.D0,
        calib.K1_raw,
        calib.D1,
        image_size,
        calib.R_cam1_cam0,
        calib.t_cam1_cam0,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=image_size,
    )

    R1 = np.ascontiguousarray(np.array(R1, dtype=np.float64))
    R2 = np.ascontiguousarray(np.array(R2, dtype=np.float64))
    P1 = np.ascontiguousarray(np.array(P1, dtype=np.float64))
    P2 = np.ascontiguousarray(np.array(P2, dtype=np.float64))
    Q = np.ascontiguousarray(np.array(Q, dtype=np.float64))

    K_rect = np.ascontiguousarray(P1[:3, :3])

    map1x, map1y = cv2.fisheye.initUndistortRectifyMap(
        calib.K0_raw,
        calib.D0,
        R1,
        K_rect,
        image_size,
        cv2.CV_32FC1,
    )

    map2x, map2y = cv2.fisheye.initUndistortRectifyMap(
        calib.K1_raw,
        calib.D1,
        R2,
        np.ascontiguousarray(P2[:3, :3]),
        image_size,
        cv2.CV_32FC1,
    )

    fx = float(K_rect[0, 0])
    fy = float(K_rect[1, 1])
    cx = float(K_rect[0, 2])
    cy = float(K_rect[1, 2])

    if abs(P2[0, 0]) < 1e-12:
        raise ValueError("Invalid rectified projection matrix P2: P2[0,0] is too small")

    baseline = float(abs(P2[0, 3] / P2[0, 0]))

    return FisheyeRectifier(
        image_size=image_size,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        map1x=map1x,
        map1y=map1y,
        map2x=map2x,
        map2y=map2y,
        K_rect=K_rect,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        baseline=baseline,
    )


def rectify_stereo_pair(imgL_raw, imgR_raw, rectifier: FisheyeRectifier):
    """
    Rectify a raw stereo fisheye pair.
    """
    imgL_rect = cv2.remap(imgL_raw, rectifier.map1x, rectifier.map1y, cv2.INTER_LINEAR)
    imgR_rect = cv2.remap(imgR_raw, rectifier.map2x, rectifier.map2y, cv2.INTER_LINEAR)
    return imgL_rect, imgR_rect