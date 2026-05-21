# Stereo Visual Odometry Project

This repository contains a classical visual odometry pipeline for the TUM VI stereo sequences:

- Room2
- Corridor3
- Outdoors5

The project uses only classical computer vision methods. No deep learning is used.

## Project pipeline

1. Load TUM VI stereo dataset.
2. Load the official calibration from `camchain.yaml`.
3. Rectify the fisheye stereo images.
4. Run Monocular VO baseline.
5. Run Stereo VO baseline using:
   - feature detection and matching,
   - stereo depth,
   - 3D-2D PnP,
   - RANSAC outlier rejection.
6. Save trajectories in TUM format.
7. Evaluate results:
   - Room2: full GT ATE/RPE evaluation.
   - Corridor3: start-end drift evaluation.
   - Outdoors5: start-end drift evaluation.
8. Save tables, plots, runtime information, frame statistics, and configuration files.

## Optional improvements

### Corridor3

A classical visual loop-closure extension was tested. The final optional improvement uses two verified visual loops:

- Loop 1: `560 -> 4760`
- Loop 2: `780 -> 5060`

This reduced the Stereo VO drift to approximately `6.75 m`.

### Outdoors5

A SIFT rotation-compensated visual loop was tested at:

- Loop: `6000 -> 11847`

A safe soft loop correction with gain `0.5` reduced the Stereo VO drift to approximately `18.31 m`.

Ground truth was used only for evaluation, not for loop detection or correction.

## Final results summary

| Dataset | Method | Evaluation | Result |
|---|---|---|---|
| Room2 | Monocular VO | ATE/RPE | See output tables |
| Room2 | Stereo VO | ATE/RPE | See output tables |
| Corridor3 | Raw Stereo VO | Start-end drift | 46.99 m |
| Corridor3 | Stereo VO + two visual loops | Start-end drift | 6.75 m |
| Outdoors5 | Monocular VO | Start-end drift | 2.41 m |
| Outdoors5 | Raw Stereo VO | Start-end drift | 32.29 m |
| Outdoors5 | Stereo VO + safe soft SIFT loop | Start-end drift | 18.31 m |

## Repository structure

```text
Stereo_vo/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
├── stereo_vo/
├── notebooks/
│   ├── ROOM2_FINAL.ipynb
│   ├── Corridor3_FINAL.ipynb
│   └── Outdoors5_FINAL.ipynb
└── outputs/
    ├── room2_final_submission/
    ├── corridor3_final_submission/
    └── outdoors5_final_submission/