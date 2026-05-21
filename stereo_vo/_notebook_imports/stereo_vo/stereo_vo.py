import cv2
import numpy as np

from .features import detect_and_match_orb
from .calibration import rectify_stereo_pair


# =========================
# Disparity
# =========================

def create_sgbm_matcher(num_disparities=16 * 8, block_size=3):
    """
    SGBM tuned for TUM-VI 512x512 fisheye-rectified images.

    The old matcher used blockSize=7 and numDisparities=160. That was more
    expensive and often too smooth. blockSize=3 + 3WAY keeps more useful depth
    at feature locations.
    """
    if num_disparities % 16 != 0:
        num_disparities = int(np.ceil(num_disparities / 16.0) * 16)

    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=int(num_disparities),
        blockSize=int(block_size),
        P1=8 * 3 * int(block_size) ** 2,
        P2=32 * 3 * int(block_size) ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=80,
        speckleRange=16,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_disparity_rectified(
    imgL_rect,
    imgR_rect,
    matcher=None,
    use_wls=True,
    wls_lambda=2000.0,
    wls_sigma=1.0,
    min_valid_disp=0.5,
):
    """
    Compute left-image disparity in pixels.

    If cv2.ximgproc is available, WLS filtering is used because your notebook
    showed WLS gave better coverage than raw SGBM. If ximgproc is unavailable,
    the function automatically falls back to raw SGBM.
    """
    matcher = matcher or create_sgbm_matcher()

    disp_left = matcher.compute(imgL_rect, imgR_rect).astype(np.float32) / 16.0

    if use_wls and hasattr(cv2, "ximgproc"):
        try:
            right_matcher = cv2.ximgproc.createRightMatcher(matcher)
            disp_right = right_matcher.compute(imgR_rect, imgL_rect).astype(np.float32) / 16.0

            wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
            wls.setLambda(float(wls_lambda))
            wls.setSigmaColor(float(wls_sigma))

            disp = wls.filter(disp_left, imgL_rect, None, disp_right).astype(np.float32)
        except Exception:
            disp = disp_left
    else:
        disp = disp_left

    disp[disp < float(min_valid_disp)] = np.nan
    return disp


# =========================
# Feature matching
# =========================

def detect_and_match_features(
    img1,
    img2,
    feature_type="orb",
    nfeatures=2500,
    ratio=0.72,
    top_k=250,
    cross_check=True,
    use_fundamental=True,
    max_motion=90.0,
):
    """
    Temporal feature matching for consecutive left images.

    The old stereo code accepted plain ORB ratio matches. In corridor/repetitive
    scenes this can pass wrong matches into PnP. Mutual ratio + motion filtering
    + optional fundamental RANSAC is slower but much safer.
    """
    return detect_and_match_orb(
        img1,
        img2,
        nfeatures=nfeatures,
        ratio=ratio,
        top_k=top_k,
        cross_check=cross_check,
        use_fundamental=use_fundamental,
        max_motion=max_motion,
    )


# =========================
# 3D-2D construction
# =========================

def build_3d2d_correspondences(
    kp_prev,
    kp_curr,
    good_matches,
    disp_prev,
    fx,
    fy,
    cx,
    cy,
    baseline,
    min_disp=0.8,
    max_depth=8.0,
    min_depth=0.20,
    patch_radius=2,
):
    """
    Build PnP correspondences:
        3D point from previous left camera depth
        2D point from current left image

    solvePnP then estimates T_curr_prev, i.e. X_curr = R X_prev + t.
    """
    obj_points = []
    img_points = []

    if disp_prev is None or kp_prev is None or kp_curr is None:
        return np.empty((0, 3), np.float32), np.empty((0, 2), np.float32)

    h, w = disp_prev.shape

    for m in good_matches:
        kp_p = kp_prev[m.queryIdx]
        kp_c = kp_curr[m.trainIdx]

        u, v = kp_p.pt
        u_i = int(round(u))
        v_i = int(round(v))

        if v_i < 0 or v_i >= h or u_i < 0 or u_i >= w:
            continue

        x0 = max(0, u_i - patch_radius)
        x1 = min(w, u_i + patch_radius + 1)
        y0 = max(0, v_i - patch_radius)
        y1 = min(h, v_i + patch_radius + 1)

        patch = disp_prev[y0:y1, x0:x1]
        patch_valid = patch[np.isfinite(patch) & (patch >= float(min_disp))]

        if len(patch_valid) < 3:
            continue

        d = float(np.median(patch_valid))
        if not np.isfinite(d) or d < min_disp:
            continue

        z = float(fx) * float(baseline) / d

        if not np.isfinite(z) or z < min_depth or z > max_depth:
            continue

        x = (float(u) - float(cx)) * z / float(fx)
        y = (float(v) - float(cy)) * z / float(fy)

        obj_points.append([x, y, z])
        img_points.append(kp_c.pt)

    return (
        np.asarray(obj_points, dtype=np.float32),
        np.asarray(img_points, dtype=np.float32),
    )


# =========================
# Pose helpers
# =========================

def rt_to_transform(R_rel, t_rel):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R_rel, dtype=np.float64)
    T[:3, 3] = np.asarray(t_rel, dtype=np.float64).reshape(3)
    return T


def rotation_angle_deg(R_rel):
    value = (np.trace(R_rel) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def relative_rotation_deg(R_a, R_b):
    return rotation_angle_deg(np.asarray(R_a).T @ np.asarray(R_b))


def reprojection_rmse(obj_points, img_points, K_rect, R_mat, tvec):
    rvec, _ = cv2.Rodrigues(np.asarray(R_mat, dtype=np.float64))
    proj, _ = cv2.projectPoints(
        obj_points.astype(np.float64),
        rvec,
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        K_rect,
        None,
    )
    proj = proj.reshape(-1, 2)
    err = proj - img_points.astype(np.float64)
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def refine_translation_fixed_rotation(
    obj_points,
    img_points,
    K_rect,
    R_fixed,
    t_init,
    max_nfev=25,
):
    """
    Refine translation while keeping rotation fixed.
    """
    try:
        from scipy.optimize import least_squares
    except Exception:
        return np.asarray(t_init, dtype=np.float64).reshape(3, 1)

    obj_points = obj_points.astype(np.float64)
    img_points = img_points.astype(np.float64)
    R_fixed = np.asarray(R_fixed, dtype=np.float64)
    t_init = np.asarray(t_init, dtype=np.float64).reshape(3)

    rvec_fixed, _ = cv2.Rodrigues(R_fixed)

    def residuals(t):
        proj, _ = cv2.projectPoints(
            obj_points,
            rvec_fixed,
            t.reshape(3, 1),
            K_rect,
            None,
        )
        proj = proj.reshape(-1, 2)
        return (proj - img_points).reshape(-1)

    result = least_squares(
        residuals,
        t_init,
        loss="huber",
        f_scale=1.0,
        max_nfev=max_nfev,
    )

    return result.x.reshape(3, 1)


# =========================
# Hybrid stereo pose
# =========================

def estimate_hybrid_stereo_pose(
    obj_points,
    img_points,
    pts_prev,
    pts_curr,
    K_rect,
    min_inliers=30,
    reprojection_error=2.0,
    iterations_count=1000,
    confidence=0.999,
    t_norm_min=0.0003,
    t_norm_max=0.20,
    rot_angle_max=12.0,
    min_essential_inliers=60,
    use_hybrid=True,
    essential_agree_max_deg=5.0,
    max_reprojection_rmse=2.5,
):
    """
    Estimate metric stereo VO motion.

    Returns T_curr_prev, i.e. X_curr = R * X_prev + t.

    Important fix over the old version:
    Essential rotation is used only when it agrees with PnP rotation. Previously
    any essential matrix with enough inliers could replace PnP rotation, which
    can easily create large trajectory drift in corridor/repetitive scenes.
    """
    if len(obj_points) < 6 or len(img_points) < 6:
        return False, None, 0, np.nan, np.nan, np.nan

    # Use SQPNP if available; otherwise EPNP. Then refine with LM.
    pnp_flag = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_EPNP)

    success, rvec_pnp, tvec_pnp, inliers = cv2.solvePnPRansac(
        objectPoints=obj_points,
        imagePoints=img_points,
        cameraMatrix=K_rect,
        distCoeffs=None,
        flags=pnp_flag,
        reprojectionError=float(reprojection_error),
        confidence=float(confidence),
        iterationsCount=int(iterations_count),
    )

    if not success or inliers is None or len(inliers) < min_inliers:
        return False, None, 0, np.nan, np.nan, np.nan

    inlier_idx = inliers[:, 0]
    obj_in = obj_points[inlier_idx]
    img_in = img_points[inlier_idx]

    try:
        cv2.solvePnPRefineLM(
            objectPoints=obj_in,
            imagePoints=img_in,
            cameraMatrix=K_rect,
            distCoeffs=None,
            rvec=rvec_pnp,
            tvec=tvec_pnp,
        )
    except Exception:
        pass

    R_pnp, _ = cv2.Rodrigues(rvec_pnp)
    R_final = R_pnp
    tvec_final = tvec_pnp

    # Essential matrix rotation is useful, but only when it agrees with PnP.
    if use_hybrid and pts_prev is not None and pts_curr is not None and len(pts_prev) >= 8:
        E, mask_E = cv2.findEssentialMat(
            pts_prev,
            pts_curr,
            cameraMatrix=K_rect,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        if E is not None:
            essential_inliers, R_ess, _, _ = cv2.recoverPose(
                E,
                pts_prev,
                pts_curr,
                K_rect,
            )

            agree = relative_rotation_deg(R_pnp, R_ess)

            if essential_inliers >= min_essential_inliers and agree <= essential_agree_max_deg:
                R_final = R_ess
                tvec_final = refine_translation_fixed_rotation(
                    obj_in,
                    img_in,
                    K_rect,
                    R_final,
                    tvec_pnp,
                    max_nfev=25,
                )

    t_norm = float(np.linalg.norm(tvec_final))
    rot_angle = rotation_angle_deg(R_final)

    if not (float(t_norm_min) <= t_norm <= float(t_norm_max)):
        return False, None, int(len(inliers)), t_norm, rot_angle, np.nan

    if rot_angle > float(rot_angle_max):
        return False, None, int(len(inliers)), t_norm, rot_angle, np.nan

    rmse = reprojection_rmse(obj_in, img_in, K_rect, R_final, tvec_final)
    if not np.isfinite(rmse) or rmse > float(max_reprojection_rmse):
        return False, None, int(len(inliers)), t_norm, rot_angle, rmse

    T_rel = rt_to_transform(R_final, tvec_final)
    return True, T_rel, int(len(inliers)), t_norm, rot_angle, rmse


# =========================
# Stereo VO main
# =========================

def run_stereo_vo_rectified(
    left_images,
    right_images,
    rectifier,
    num_frames=150,
    feature_type="orb",
    nfeatures=2500,
    ratio=0.72,
    top_k=250,
    cross_check=True,
    use_fundamental=True,
    max_motion=90.0,
    min_disp=0.8,
    max_depth=8.0,
    min_depth=0.20,
    patch_radius=2,
    min_corr=35,
    min_inliers=30,
    t_norm_min=0.0003,
    t_norm_max=0.20,
    rot_angle_max=12.0,
    use_hybrid=True,
    use_wls=True,
    max_reprojection_rmse=2.5,
    use_ba=False,   # kept only so old notebook calls do not crash
    skip_first_n=0,
    invert_pose_update=True,
):
    """
    Run stereo VO on already-calibrated TUM-VI stereo images.

    Pose convention:
        solvePnP returns T_curr_prev.
        If poses are world_T_cam, update with inv(T_curr_prev):
            world_T_curr = world_T_prev @ inv(T_curr_prev)

    That is why invert_pose_update=True by default.
    """
    matcher = create_sgbm_matcher()
    K_rect = rectifier.K_rect

    poses = [np.eye(4, dtype=np.float64)]
    trajectory = [np.zeros(3, dtype=np.float64)]
    stats = []
    tvec_norms = []

    num_frames = min(int(num_frames), len(left_images), len(right_images))

    prev_left_rect = None
    prev_right_rect = None
    prev_disp = None

    for i in range(1, num_frames):
        if i < skip_first_n:
            T_curr = poses[-1].copy()
            poses.append(T_curr)
            trajectory.append(T_curr[:3, 3].copy())
            stats.append({
                "frame": i,
                "matches": 0,
                "corr_3d2d": 0,
                "inliers": 0,
                "success": False,
                "pnp_success": False,
                "pnp_inliers": 0,
                "t_norm": np.nan,
                "rot_angle": np.nan,
                "reproj_rmse": np.nan,
            })
            continue

        if prev_left_rect is None:
            imgL_prev_raw = cv2.imread(str(left_images[i - 1]), cv2.IMREAD_GRAYSCALE)
            imgR_prev_raw = cv2.imread(str(right_images[i - 1]), cv2.IMREAD_GRAYSCALE)
            if imgL_prev_raw is not None and imgR_prev_raw is not None:
                prev_left_rect, prev_right_rect = rectify_stereo_pair(
                    imgL_prev_raw,
                    imgR_prev_raw,
                    rectifier,
                )
                prev_disp = compute_disparity_rectified(
                    prev_left_rect,
                    prev_right_rect,
                    matcher,
                    use_wls=use_wls,
                    min_valid_disp=min_disp,
                )

        imgL_curr_raw = cv2.imread(str(left_images[i]), cv2.IMREAD_GRAYSCALE)
        imgR_curr_raw = cv2.imread(str(right_images[i]), cv2.IMREAD_GRAYSCALE)

        if prev_left_rect is None or prev_disp is None or imgL_curr_raw is None or imgR_curr_raw is None:
            T_curr = poses[-1].copy()
            poses.append(T_curr)
            trajectory.append(T_curr[:3, 3].copy())
            stats.append({
                "frame": i,
                "matches": 0,
                "corr_3d2d": 0,
                "inliers": 0,
                "success": False,
                "pnp_success": False,
                "pnp_inliers": 0,
                "t_norm": np.nan,
                "rot_angle": np.nan,
                "reproj_rmse": np.nan,
            })
            prev_left_rect = None
            prev_right_rect = None
            prev_disp = None
            continue

        imgL_curr, imgR_curr = rectify_stereo_pair(
            imgL_curr_raw,
            imgR_curr_raw,
            rectifier,
        )

        kp_prev, kp_curr, good_matches, pts_prev, pts_curr = detect_and_match_features(
            prev_left_rect,
            imgL_curr,
            feature_type=feature_type,
            nfeatures=nfeatures,
            ratio=ratio,
            top_k=top_k,
            cross_check=cross_check,
            use_fundamental=use_fundamental,
            max_motion=max_motion,
        )

        obj_points, img_points = build_3d2d_correspondences(
            kp_prev,
            kp_curr,
            good_matches,
            prev_disp,
            rectifier.fx,
            rectifier.fy,
            rectifier.cx,
            rectifier.cy,
            rectifier.baseline,
            min_disp=min_disp,
            max_depth=max_depth,
            min_depth=min_depth,
            patch_radius=patch_radius,
        )

        if len(obj_points) >= min_corr:
            success, T_rel, inlier_count, t_norm, rot_angle, reproj_rmse = estimate_hybrid_stereo_pose(
                obj_points,
                img_points,
                pts_prev,
                pts_curr,
                K_rect,
                min_inliers=min_inliers,
                t_norm_min=t_norm_min,
                t_norm_max=t_norm_max,
                rot_angle_max=rot_angle_max,
                use_hybrid=use_hybrid,
                max_reprojection_rmse=max_reprojection_rmse,
            )
        else:
            success, T_rel, inlier_count, t_norm, rot_angle, reproj_rmse = False, None, 0, np.nan, np.nan, np.nan

        if success:
            if invert_pose_update:
                T_curr = poses[-1] @ np.linalg.inv(T_rel)
            else:
                T_curr = poses[-1] @ T_rel
            tvec_norms.append(t_norm)
        else:
            T_curr = poses[-1].copy()

        poses.append(T_curr)
        trajectory.append(T_curr[:3, 3].copy())

        stats.append({
            "frame": i,
            "matches": len(good_matches),
            "corr_3d2d": len(obj_points),
            "inliers": int(inlier_count),
            "success": bool(success),
            "pnp_success": bool(success),
            "pnp_inliers": int(inlier_count),
            "t_norm": float(t_norm) if np.isfinite(t_norm) else np.nan,
            "rot_angle": float(rot_angle) if np.isfinite(rot_angle) else np.nan,
            "reproj_rmse": float(reproj_rmse) if np.isfinite(reproj_rmse) else np.nan,
        })

        # Cache current rectified pair/disparity for next frame. This saves one
        # rectification per loop and keeps the previous disparity consistent.
        prev_left_rect = imgL_curr
        prev_right_rect = imgR_curr
        prev_disp = compute_disparity_rectified(
            prev_left_rect,
            prev_right_rect,
            matcher,
            use_wls=use_wls,
            min_valid_disp=min_disp,
        )

    return (
        poses,
        np.asarray(trajectory, dtype=np.float64),
        stats,
        np.asarray(tvec_norms, dtype=np.float64),
    )
