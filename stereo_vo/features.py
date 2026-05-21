import cv2
import numpy as np


def create_orb(nfeatures=2000):
    """
    Create an ORB detector/descriptor extractor.
    Better settings for VO stability.
    """
    return cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=19,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=12,
    )


def detect_and_describe_orb(img, nfeatures=2000):
    if img is None:
        raise ValueError("Input image is None")

    orb = create_orb(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors


def keypoints_to_points(kp1, kp2, matches):
    if len(matches) < 1:
        return None, None

    pts1 = np.asarray([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts2 = np.asarray([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
    return pts1, pts2


def filter_matches_by_motion(kp1, kp2, matches, max_motion=80.0):
    """
    Remove matches with unrealistically large pixel displacement.
    Useful for consecutive-frame VO.
    """
    filtered = []

    for m in matches:
        p1 = np.array(kp1[m.queryIdx].pt)
        p2 = np.array(kp2[m.trainIdx].pt)

        if np.linalg.norm(p2 - p1) <= max_motion:
            filtered.append(m)

    return filtered


def filter_matches_fundamental(kp1, kp2, matches, reproj_thresh=1.5):
    """
    RANSAC geometric filtering using fundamental matrix.
    This removes many wrong corridor/repetitive matches.
    """
    if len(matches) < 8:
        return matches

    pts1, pts2 = keypoints_to_points(kp1, kp2, matches)

    F, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        cv2.FM_RANSAC,
        ransacReprojThreshold=reproj_thresh,
        confidence=0.99,
    )

    if F is None or mask is None:
        return matches

    mask = mask.ravel().astype(bool)
    return [m for m, keep in zip(matches, mask) if keep]


def match_descriptors_knn(des1, des2, ratio=0.75, top_k=None):
    """
    Basic KNN + Lowe ratio.
    Kept for compatibility.
    """
    if des1 is None or des2 is None:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []

    for pair in raw_matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda m: m.distance)

    if top_k is not None:
        good_matches = good_matches[:top_k]

    return good_matches


def match_descriptors_cross_ratio(des1, des2, ratio=0.75):
    """
    Ratio test in both directions, then mutual consistency.
    Stronger than one-way ratio test.
    """
    if des1 is None or des2 is None:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    raw12 = bf.knnMatch(des1, des2, k=2)
    raw21 = bf.knnMatch(des2, des1, k=2)

    good12 = []

    for pair in raw12:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good12.append(m)

    good21_pairs = set()

    for pair in raw21:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            # m.queryIdx is des2 index, m.trainIdx is des1 index
            good21_pairs.add((m.trainIdx, m.queryIdx))

    mutual = []

    for m in good12:
        if (m.queryIdx, m.trainIdx) in good21_pairs:
            mutual.append(m)

    mutual = sorted(mutual, key=lambda m: m.distance)

    return mutual


def detect_and_match_orb(
    img1,
    img2,
    nfeatures=2000,
    ratio=0.75,
    top_k=None,
    cross_check=False,
    use_fundamental=False,
    max_motion=9999.0,
):
    """
    Detect and match ORB features between two consecutive images.

    Returns:
        kp1, kp2, good_matches, pts1, pts2
    """
    if img1 is None or img2 is None:
        raise ValueError("Input image is None")

    kp1, des1 = detect_and_describe_orb(img1, nfeatures=nfeatures)
    kp2, des2 = detect_and_describe_orb(img2, nfeatures=nfeatures)

    if des1 is None or des2 is None:
        return kp1, kp2, [], None, None

    if cross_check:
        good_matches = match_descriptors_cross_ratio(des1, des2, ratio=ratio)
    else:
        good_matches = match_descriptors_knn(des1, des2, ratio=ratio, top_k=None)

    good_matches = filter_matches_by_motion(
        kp1,
        kp2,
        good_matches,
        max_motion=max_motion,
    )

    if use_fundamental:
        good_matches = filter_matches_fundamental(
            kp1,
            kp2,
            good_matches,
            reproj_thresh=1.5,
        )

    good_matches = sorted(good_matches, key=lambda m: m.distance)

    if top_k is not None:
        good_matches = good_matches[:top_k]

    if len(good_matches) < 8:
        return kp1, kp2, good_matches, None, None

    pts1, pts2 = keypoints_to_points(kp1, kp2, good_matches)

    return kp1, kp2, good_matches, pts1, pts2


def draw_matches_image(img1, kp1, img2, kp2, matches, max_draw=100):
    matches_to_draw = matches[:max_draw]
    return cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )