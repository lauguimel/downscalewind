#!/usr/bin/env bash
# sync_to_uga.sh — Upload scripts + templates + data to UGA for server-side campaign
#
# Usage:
#   cd services/module2a-cfd
#   bash sync_to_uga.sh

set -euo pipefail

REMOTE=UGA
DSW=/home/guillaume/dsw

echo "=== Syncing to $REMOTE:$DSW ==="

# 1. Python scripts (all .py in module2a-cfd root)
echo "  [1/5] Scripts..."
rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='_archive' \
    *.py "$REMOTE:$DSW/scripts/"

# 2. Templates (Jinja2 — includes new q.j2)
echo "  [2/5] Templates..."
rsync -avz templates/openfoam/ "$REMOTE:$DSW/scripts/templates/openfoam/"

# 3. Campaign config + timestamps
echo "  [3/5] Configs + timestamps..."
ssh $REMOTE "mkdir -p $DSW/configs/sites"
rsync -avz ../../configs/sites/perdigao.yaml "$REMOTE:$DSW/configs/sites/"
rsync -avz ../../data/campaign/sf_poc/timestamps_100.csv "$REMOTE:$DSW/"

# 4. Campaign script
echo "  [4/5] Campaign script..."
rsync -avz run_campaign_q.sh "$REMOTE:$DSW/scripts/"

# 5. Shared module (for data_io, logging)
echo "  [5/5] Shared module..."
rsync -avz --exclude='__pycache__' ../../shared/ "$REMOTE:$DSW/shared/"

echo ""
echo "=== Sync complete ==="
echo ""
echo "To launch:"
echo "  ssh $REMOTE"
echo "  cd $DSW"
echo "  nohup bash scripts/run_campaign_q.sh > log_campaign_q.txt 2>&1 &"
echo "  tail -f log_campaign_q.txt"
echo ""
echo "To monitor:"
echo "  ssh $REMOTE 'tail -20 $DSW/log_campaign_q.txt'"
echo ""
echo "To rsync results back:"
echo "  rsync -avz $REMOTE:$DSW/cases/poc_100ts_q/ ../../data/cases/poc_100ts_q/"
