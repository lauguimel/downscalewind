#!/bin/bash -l
# Login-node chain (network I/O only, no compute): ERA5 downloads for the hub-height benchmark,
# then qsub the Penmanshiel GPU job once its store exists. Run with:
#   nohup bash ~/dsw/configs/hpc/hub_height_chain.sh > /scratch/maitreje/dsw/hub_height/chain.log 2>&1 &
set -u
BASE="$HOME/dsw"
cd "$BASE/services/data-ingestion"
module load Miniconda3/24.9.2-0 || module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate fuxicfd
ING="python -u ingest_era5_europe_hourly.py --max-days-per-req 7"

echo "[$(date)] ERA5 Penmanshiel 2018 (10 levels + surface)"
$ING --output ../../data/raw/era5_penmanshiel_2018.zarr --start 2018-01 --end 2018-12 --bbox "54.5,-3.75,57.0,-0.75"
RC=$?; echo "[$(date)] RC=$RC"
if [ "$RC" -eq 0 ]; then
  cd "$BASE" && JOB=$(qsub configs/hpc/hub_height_penmanshiel.pbs) && echo "[$(date)] qsub penmanshiel → $JOB"
  cd "$BASE/services/data-ingestion"
fi

echo "[$(date)] ERA5 Penmanshiel 2018 u100/v100"
python -u ingest_era5_europe_hourly.py --output ../../data/raw/era5_penmanshiel_2018_u100.zarr --start 2018-01 --end 2018-12 \
  --bbox "54.5,-3.75,57.0,-0.75" --surface-only --surface-vars "100m_u_component_of_wind,100m_v_component_of_wind" --max-days-per-req 31
echo "[$(date)] RC=$?"

echo "[$(date)] ERA5 Alaiz 2017-07 → 2019-07 (10 levels + surface)"
$ING --output ../../data/raw/era5_alaiz_2017_2019.zarr --start 2017-07 --end 2019-07 --bbox "41.5,-3.0,44.0,0.0"
echo "[$(date)] RC=$?"

echo "[$(date)] ERA5 Alaiz u100/v100"
python -u ingest_era5_europe_hourly.py --output ../../data/raw/era5_alaiz_2017_2019_u100.zarr --start 2017-07 --end 2019-07 \
  --bbox "41.5,-3.0,44.0,0.0" --surface-only --surface-vars "100m_u_component_of_wind,100m_v_component_of_wind" --max-days-per-req 31
echo "[$(date)] RC=$? — CHAIN DONE"
