import cv2
import numpy as np

from .features import detect_and_match_orb


def estimate_pose_essential(pts1, pts2, K, ransac_threshold=1.0):
    if pts1 is None or pts2 is None or len(pts1) < 8:
        return None, None, None, 0

    E, _ = cv2.findEssentialMat(
        pts1,
        pts2,
        cameraMatrix=K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ransac_threshold,
    )

    if E is None:
        return None, None, None, 0

    inliers, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    return R_rel, t_rel, mask_pose, int(inliers)


def make_transform(R_rel, t_rel, unit_scale=True):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_rel

    t = np.asarray(t_rel, dtype=np.float64).ravel()
    if unit_scale:
        n = np.linalg.norm(t)
        if n > 1e-12:
            t = t / n

    T[:3, 3] = t
    return T


def _load_left_image(path, rectifier=None):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if rectifier is not None:
        img = cv2.remap(img, rectifier.map1x, rectifier.map1y, cv2.INTER_LINEAR)
    return img


def run_monocular_vo(
    left_images,
    K,
    num_frames=200,
    nfeatures=2000,
    ratio=0.75,
    top_k=300,
    ransac_threshold=1.0,
    min_inliers=15,
    rectifier=None,
):
    poses = [np.eye(4, dtype=np.float64)]
    trajectory = [np.zeros(3, dtype=np.float64)]
    stats = []

    num_frames = min(num_frames, len(left_images))

    for i in range(1, num_frames):
        img_prev = _load_left_image(left_images[i - 1], rectifier=rectifier)
        img_curr = _load_left_image(left_images[i], rectifier=rectifier)

        kp1, kp2, good_matches, pts1, pts2 = detect_and_match_orb(
            img_prev,
            img_curr,
            nfeatures=nfeatures,
            ratio=ratio,
            top_k=top_k,
        )

        R_rel, t_rel, _, inliers = estimate_pose_essential(
            pts1,
            pts2,
            K,
            ransac_threshold=ransac_threshold,
        )

        success = (
            R_rel is not None
            and t_rel is not None
            and inliers >= min_inliers
        )

        if success:
            T_rel = make_transform(R_rel, t_rel, unit_scale=True)
            T_curr = poses[-1] @ T_rel
        else:
            T_curr = poses[-1].copy()

        poses.append(T_curr)
        trajectory.append(T_curr[:3, 3].copy())

        stats.append(
            {
                "frame": i,
                "kp_prev": 0 if kp1 is None else len(kp1),
                "kp_curr": 0 if kp2 is None else len(kp2),
                "matches": len(good_matches),
                "inliers": int(inliers),
                "success": bool(success),
            }
        )

    return poses, np.asarray(trajectory, dtype=np.float64), stats