# Training

ReVA-HPE training implementation, derived from the VGGT training framework.
Hydra configs live under `config/`, the paper model is
`config/flame_fromvggt_largepairs_fixed_fov_unified_crop_depth_norm_delta_t_rot1_vggtinit.yaml`.
Launch through `scripts/train_paper_model.sh`, pair-manifest details are in
`DATA.md`.

Notes:

- The trainer loads VGGT-1B with `checkpoint.strict: false` and remaps base
  weights into the LoRA-wrapped linears (`<module>.weight ->
  <module>.base.weight`), so the frozen backbone starts from the pretrained
  weights while only the LoRA adapters and the camera head train.
- Supervision is applied to frame 1 (the query) only; translation targets are
  depth-normalized (dx/z0, dy/z0, log z/z0).
- For the original VGGT training framework and its CO3D/VKitti instructions,
  see [facebookresearch/vggt](https://github.com/facebookresearch/vggt).
