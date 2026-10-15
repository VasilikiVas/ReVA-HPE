<div align="center">

# ReVA-HPE

### Relative 6DoF Head Pose Estimation with Synthetic Pose Supervision and Personalized Virtual Anchors

#### WACV 2027

<p>
  <a href="https://vasilikivas.github.io/">Vasiliki Vasileiou</a> ·
  <a href="https://filby89.github.io/">Panagiotis P. Filntisis</a> ·
  <a href="https://robotics.ntua.gr/members/maragos/">Petros Maragos</a> ·
  <a href="https://www.cis.upenn.edu/~kostas/">Kostas Daniilidis</a>
</p>

<p>
  <a href="https://vasilikivas.github.io/ReVA-HPE/">
    <img src="https://img.shields.io/badge/Project%20Page-Website-00CC00?style=for-the-badge" alt="Project Page">
  </a>
  <img src="https://img.shields.io/badge/WACV-2027-4F6D7A?style=for-the-badge" alt="WACV 2027">
  <img src="https://img.shields.io/badge/Code-Coming%20Soon-lightgrey?style=for-the-badge" alt="Code coming soon">
</p>

</div>

<p align="center">
  <img src="assets/teaser.png" alt="ReVA-HPE teaser" width="100%">
</p>

## Code release

The full implementation — training, evaluation, and the avatar/virtual-anchor
pipeline — will be released in this repository soon. Watch/star the repo to be
notified. Until then, see the
[project page](https://vasilikivas.github.io/ReVA-HPE/) for the paper summary
and results.

## Overview

ReVA-HPE turns relative head pose estimation into a complete **metric 6DoF**
system that needs **no ground-truth anchor pose**. A personalized Gaussian
avatar is reconstructed from a single unlabeled enrollment image and rendered
into a bank of **virtual anchors** with known poses: query **rotation** comes
from pose-consensus RANSAC over the bank, and metric **translation** from a
single enrollment landmark-PnP solve combined with the learned depth-normalized
relative translation. The relative estimator is trained purely on procedural
FLAME renderings — no real pose supervision, pseudo-labels, domain adaptation,
or fine-tuning on the evaluation sets. A fast variant (**VA-fast**) uses the
bank once at enrollment and then needs only one real-to-real prediction per
query.

<p align="center">
  <img src="assets/method.png" alt="ReVA-HPE method overview" width="100%">
</p>

## Results

Shared MTCNN protocol; rotation in degrees, translation in millimeters, lower
is better. "First frame" fixes each subject's first available frame as
enrollment; "α ≤ 15°" selects a near-frontal enrollment. Methods marked †
require the ground-truth pose of the enrollment image at test time.

**BIWI**

| Method | Yaw ↓ | Pitch ↓ | Roll ↓ | MAEr ↓ | MAEt ↓ | Pose supervision |
|---|---|---|---|---|---|---|
| 6DRepNet | 3.74 | 4.95 | 3.04 | 3.91 | — | mixed |
| TokenHPE-v1 | 5.57 | 6.23 | 3.79 | 5.20 | — | mixed |
| TRG* | 4.58 | 7.18 | 3.68 | 5.15 | 48.71 | mixed |
| SemiUHPE | 3.30 | 3.71 | 2.60 | 3.20 | — | real |
| OPAL | 3.74 | 7.23 | 3.25 | 4.74 | — | mixed |
| VGGT-HPE-Abs | 4.90 | 7.01 | 3.53 | 5.15 | — | synthetic |
| **ReVA-HPE (ours, first frame)** | 2.20 | 2.94 | 2.47 | **2.53** | 17.10 | synthetic |
| **ReVA-HPE (ours, α ≤ 15°)** | 2.01 | 2.84 | 2.61 | **2.49** | 18.94 | synthetic |
| ReVA-HPE-GT† (ours) | 1.60 | 1.92 | 2.72 | 2.08 | 9.57 | synthetic |

**FaMoS**

| Method | Yaw ↓ | Pitch ↓ | Roll ↓ | MAEr ↓ | MAEt ↓ | Pose supervision |
|---|---|---|---|---|---|---|
| 6DRepNet | 4.54 | 8.03 | 5.36 | 5.98 | — | mixed |
| TokenHPE-v1 | 8.62 | 5.69 | 4.46 | 6.26 | — | mixed |
| TRG* | 6.18 | 6.26 | 4.32 | 5.59 | 34.66 | mixed |
| SemiUHPE | 1.98 | 3.65 | 2.94 | 2.86 | — | real |
| OPAL | 8.33 | 5.38 | 5.14 | 6.28 | — | mixed |
| VGGT-HPE-Abs | 34.41 | 10.60 | 9.73 | 18.29 | 40.46 | synthetic |
| **ReVA-HPE (ours, first frame)** | 1.34 | 6.69 | 3.31 | 3.78 | 31.27 | synthetic |
| **ReVA-HPE (ours, α ≤ 15°)** | 1.21 | 3.39 | 2.80 | **2.47** | 23.71 | synthetic |
| ReVA-HPE-GT† (ours) | 1.38 | 1.77 | 2.70 | 1.95 | 9.87 | synthetic |

### Qualitative results

<p align="center">
  <img src="assets/results_biwi.jpg" alt="Qualitative results on BIWI" width="100%">
</p>

<p align="center">
  <img src="assets/results_famos.jpg" alt="Qualitative results on FaMoS" width="100%">
</p>

## Citation

If you use this work, please cite (pages and the official PDF link will be
added when the WACV 2027 proceedings are published):

```bibtex
@inproceedings{vasileiou2027revahpe,
  title={ReVA-HPE: Relative 6DoF Head Pose Estimation with Synthetic Pose Supervision and Personalized Virtual Anchors},
  author={Vasileiou, Vasiliki and Filntisis, Panagiotis P. and Maragos, Petros and Daniilidis, Kostas},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2027}
}
```

## Links

- Project page: https://vasilikivas.github.io/ReVA-HPE/

## Acknowledgements

Built on [VGGT](https://github.com/facebookresearch/vggt),
[FaceLift](https://github.com/weijielyu/FaceLift), and
[FLAME](https://flame.is.tue.mpg.de/). We thank the EuroHPC Joint Undertaking
for access to Leonardo (CINECA).
