from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def pose_matrix_from_row(row: pd.Series) -> np.ndarray:
    """
    Build a 4x4 pose matrix from one TUM-VI mocap row.
    Ground truth is given as world -> IMU/body pose.
    """
    tx = float(row[" p_RS_R_x [m]"])
    ty = float(row[" p_RS_R_y [m]"])
    tz = float(row[" p_RS_R_z [m]"])

    qw = float(row[" q_RS_w []"])
    qx = float(row[" q_RS_x []"])
    qy = float(row[" q_RS_y []"])
    qz = float(row[" q_RS_z []"])

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return T


def load_room_ground_truth(gt_file: Path):
    """
    Load raw ground-truth CSV.
    Returns:
        gt_df, timestamps_ns, xyz_body
    """
    gt_df = pd.read_csv(gt_file)
    timestamps_ns = gt_df["#timestamp [ns]"].to_numpy(dtype=np.int64)
    xyz_body = gt_df[[" p_RS_R_x [m]", " p_RS_R_y [m]", " p_RS_R_z [m]"]].to_numpy(dtype=np.float64)
    return gt_df, timestamps_ns, xyz_body


def load_gt_camera_poses(gt_file: Path, T_cam_imu: np.ndarray):
    """
    Convert GT from world->IMU pose to world->camera pose.

    GT gives T_w_imu.
    Calibration gives T_cam_imu.
    So:
        T_w_cam = T_w_imu @ inv(T_cam_imu)
    """
    gt_df = pd.read_csv(gt_file)
    timestamps_ns = gt_df["#timestamp [ns]"].to_numpy(dtype=np.int64)

    T_cam_imu = np.asarray(T_cam_imu, dtype=np.float64)
    if T_cam_imu.shape != (4, 4):
        raise ValueError(f"T_cam_imu must be 4x4, got {T_cam_imu.shape}")

    T_imu_cam = np.linalg.inv(T_cam_imu)

    poses_cam = []
    for _, row in gt_df.iterrows():
        T_w_imu = pose_matrix_from_row(row)
        T_w_cam = T_w_imu @ T_imu_cam
        poses_cam.append(T_w_cam)

    return timestamps_ns, poses_cam


def match_gt_to_timestamps(
    gt_timestamps_ns: np.ndarray,
    gt_xyz: np.ndarray,
    est_timestamps_ns: np.ndarray,
    est_xyz: np.ndarray,
):
    """
    Match GT xyz to estimated timestamps using nearest-neighbor timestamps.
    Returns:
        matched_gt_xyz, matched_est_xyz
    """
    matched_gt = []
    matched_est = []

    for ts, est_p in zip(est_timestamps_ns, est_xyz):
        idx = int(np.argmin(np.abs(gt_timestamps_ns - ts)))
        matched_gt.append(gt_xyz[idx])
        matched_est.append(est_p)

    return np.asarray(matched_gt, dtype=np.float64), np.asarray(matched_est, dtype=np.float64)


def match_timestamps(
    query_ts: np.ndarray,
    ref_ts: np.ndarray,
    ref_poses: list,
):
    """
    Match query timestamps to nearest reference poses.
    Returns:
        matched_poses
    """
    matched_poses = []
    for ts in query_ts:
        idx = int(np.argmin(np.abs(ref_ts - ts)))
        matched_poses.append(ref_poses[idx])
    return matched_poses


def poses_to_xyz(poses: list) -> np.ndarray:
    """
    Convert list of 4x4 poses to Nx3 xyz array.
    """
    return np.asarray([T[:3, 3] for T in poses], dtype=np.float64)


def rough_scale_align(gt_xyz: np.ndarray, est_xyz: np.ndarray):
    """
    Simple origin alignment + end-to-end scale alignment.
    Useful for quick visualization only.
    """
    gt0 = gt_xyz - gt_xyz[0]
    est0 = est_xyz - est_xyz[0]

    gt_len = np.linalg.norm(gt0[-1] - gt0[0])
    est_len = np.linalg.norm(est0[-1] - est0[0])

    scale = gt_len / est_len if est_len > 1e-12 else 1.0
    return gt0, est0 * scale, scale


def rigid_align_3d(A: np.ndarray, B: np.ndarray):
    """
    Rigid SE(3)-style alignment in xyz only.
    Find rotation + translation such that A_aligned ~= B.

    Args:
        A: Nx3 estimated points
        B: Nx3 ground-truth points

    Returns:
        A_aligned, R_align, t_align
    """
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    Rm = Vt.T @ U.T

    if np.linalg.det(Rm) < 0:
        Vt[-1, :] *= -1
        Rm = Vt.T @ U.T

    t = centroid_B - Rm @ centroid_A
    A_aligned = (Rm @ A.T).T + t

    return A_aligned, Rm, t


def sim3_align(A: np.ndarray, B: np.ndarray):
    """
    Similarity alignment (scale + rotation + translation), Umeyama-style.
    Align A to B.

    Args:
        A: Nx3 estimated points
        B: Nx3 ground-truth points

    Returns:
        A_aligned, scale, R_align, t_align
    """
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")

    n = A.shape[0]
    if n == 0:
        raise ValueError("Empty input arrays")

    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)

    AA = A - mu_A
    BB = B - mu_B

    cov = (BB.T @ AA) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    Rm = U @ S @ Vt

    var_A = np.mean(np.sum(AA**2, axis=1))
    if var_A < 1e-12:
        scale = 1.0
    else:
        scale = np.trace(np.diag(D) @ S) / var_A

    t = mu_B - scale * (Rm @ mu_A)
    A_aligned = (scale * (Rm @ A.T)).T + t

    return A_aligned, scale, Rm, t


def ate_rmse(est_xyz: np.ndarray, gt_xyz: np.ndarray) -> float:
    """
    Absolute Trajectory Error RMSE on xyz positions.
    """
    if est_xyz.shape != gt_xyz.shape:
        raise ValueError(f"Shape mismatch: {est_xyz.shape} vs {gt_xyz.shape}")

    err = np.linalg.norm(est_xyz - gt_xyz, axis=1)
    return float(np.sqrt(np.mean(err**2)))

def rpe_translation_rmse(est_xyz: np.ndarray, gt_xyz: np.ndarray, gap: int = 20) -> float:
    """
    Relative Pose Error translation RMSE over a fixed frame gap.

    This approximates translational RPE using xyz positions:
        error_i = ||(est[i+gap] - est[i]) - (gt[i+gap] - gt[i])||

    Args:
        est_xyz: Nx3 estimated aligned trajectory positions
        gt_xyz: Nx3 ground-truth trajectory positions
        gap: frame interval used for relative motion comparison

    Returns:
        Translational RPE RMSE in meters.
    """
    if est_xyz.shape != gt_xyz.shape:
        raise ValueError(f"Shape mismatch: {est_xyz.shape} vs {gt_xyz.shape}")

    if len(est_xyz) <= gap:
        return float("nan")

    errors = []

    for i in range(len(est_xyz) - gap):
        est_delta = est_xyz[i + gap] - est_xyz[i]
        gt_delta = gt_xyz[i + gap] - gt_xyz[i]
        errors.append(np.linalg.norm(est_delta - gt_delta))

    return float(np.sqrt(np.mean(np.square(errors))))

def poses_to_tum_array(timestamps_ns: np.ndarray, poses: list) -> np.ndarray:
    """
    Convert list of 4x4 poses to TUM trajectory format:
    timestamp tx ty tz qx qy qz qw
    """
    rows = []

    for ts, T in zip(timestamps_ns, poses):
        t = T[:3, 3]
        qx, qy, qz, qw = R.from_matrix(T[:3, :3]).as_quat()
        rows.append([ts * 1e-9, t[0], t[1], t[2], qx, qy, qz, qw])

    return np.asarray(rows, dtype=np.float64)