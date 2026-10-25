# Checkpoints

## ReVA-HPE

| Checkpoint | Size | Download |
|---|---:|---|
| ReVA-HPE `checkpoint_410.pt` (relative 6DoF estimator) | 6.3 GB | [Hugging Face](https://huggingface.co/Sila97/ReVA-HPE) |

Download:

```bash
hf download Sila97/ReVA-HPE checkpoint_410.pt --local-dir checkpoints/reva_hpe
# or: wget https://huggingface.co/Sila97/ReVA-HPE/resolve/main/checkpoint_410.pt
```

Contains VGGT-1B-derived weights, released under the VGGT research
(non-commercial) license (`LICENSE.txt`). Set
`REVA_HPE_CHECKPOINT` in `assets.env`.

## Third-party (official sources)

| Asset | Source | Notes |
|---|---|---|
| VGGT-1B `model.pt` | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | frozen base |
| FaceLift (GS-LRM + mvdiffusion weights) | [weijielyu/FaceLift](https://github.com/weijielyu/FaceLift) | avatar reconstruction, used frozen |
| MTCNN | `facenet-pytorch==2.6.0` | auto-downloaded |
| FAN 2D landmarks | `face-alignment==1.5.0` | auto-downloaded |
| Baselines | [6DRepNet](https://github.com/thohemp/6DRepNet), [TokenHPE](https://github.com/zc2023/TokenHPE), [TRG](https://github.com/asw91666/TRG-Release), [SemiUHPE](https://github.com/hnuzhy/SemiUHPE), [OPAL](https://github.com/pcr-upm/opal23_headpose), [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface), [6dof_face (PerspNet)](https://github.com/cbsropenproject/6dof_face) | official weights from each repo |
