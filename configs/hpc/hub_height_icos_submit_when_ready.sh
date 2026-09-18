#!/bin/bash -l
# Wait for the ICOS-zone ERA5 downloads to finish, then submit the three remaining towers.
LOG=/scratch/maitreje/dsw/hub_height/era5_icos_new.log
ERA5=$HOME/dsw/data/raw/era5_icos_new_jja2020.zarr
U100=$HOME/dsw/data/raw/era5_icos_new_jja2020_u100.zarr
while ! grep -q "ICOS_NEW_U100_DONE" "$LOG"; do
    pgrep -u "$USER" -f era5_icos_new.sh >/dev/null || { echo "[$(date)] download script died before finishing"; break; }
    sleep 600
done
[ -d "$ERA5/coords" ] || { echo "[$(date)] MISSING $ERA5 — nothing submitted"; exit 1; }
[ -d "$U100/coords" ] || { echo "[$(date)] no u100 store — submitting without ERA5 100 m"; U100=""; }
cd "$HOME/dsw"
for T in oxk toh kre; do
    echo "[$(date)] qsub $T: $(qsub -v TOWER=$T,ERA5=$ERA5,U100=$U100 configs/hpc/hub_height_icos_tower.pbs)"
done
echo "[$(date)] WATCHER_DONE"
