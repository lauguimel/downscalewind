#!/bin/bash
set -euo pipefail

qsub configs/hpc/train_v2_fno3d_resid_s4.pbs
qsub configs/hpc/train_v2_fno3d_resid_s4_agl.pbs
qsub configs/hpc/train_v2_vit_base_resid_s4_geo.pbs
qsub configs/hpc/train_v2_vit_base_resid_s4_geo_agl.pbs
qsub configs/hpc/train_v2_vit_large_resid_s4_geo.pbs
qsub configs/hpc/train_v2_vit_large_resid_s4_geo_agl.pbs
