#!/bin/bash
# transfer_all_to_oci.sh — Tar everything in 2 archives, transfer to OCI.
#
# Use ONLY if you have ~250 GB free on both UGA and OCI.
# Sequential: campaign first (created → transferred → deleted local), then training.
#
# Usage (run on UGA, ideally inside tmux):
#   bash transfer_all_to_oci.sh

set -euo pipefail

SSH_KEY="$HOME/.ssh/id_oci"
OCI_TARGET="enjoy@wx-outdoor.com"
OCI_DST="/data/tars"

UGA_TAR_DIR="$HOME/dsw/tars_tmp"
mkdir -p "$UGA_TAR_DIR"

SSH_OPTS="ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new"

$SSH_OPTS "$OCI_TARGET" "mkdir -p $OCI_DST"

echo "==> Free space check"
echo "UGA:" && df -h "$HOME/dsw" | tail -1
echo "OCI:" && $SSH_OPTS "$OCI_TARGET" "df -h /data | tail -1"

# ─────────────────────────────────────────────────────────────────────
# Part 1: campaign_9k → ~83 GB tar
# ─────────────────────────────────────────────────────────────────────
TAR_RAW="$UGA_TAR_DIR/campaign_9k_full.tar"
echo
echo "==> [1/2] Creating $TAR_RAW (~83 GB)"
cd ~/dsw
tar -cf "$TAR_RAW" campaign_9k/
echo "    Tar size: $(du -h "$TAR_RAW" | cut -f1)"

echo "==> [1/2] Transferring to OCI..."
rsync -av --progress -e "$SSH_OPTS" "$TAR_RAW" "$OCI_TARGET:$OCI_DST/"

echo "==> [1/2] Removing local tar"
rm "$TAR_RAW"

# ─────────────────────────────────────────────────────────────────────
# Part 2: training_9k → ~164 GB tar
# ─────────────────────────────────────────────────────────────────────
TAR_TRAIN="$UGA_TAR_DIR/training_9k_full.tar"
echo
echo "==> [2/2] Creating $TAR_TRAIN (~164 GB)"
cd ~/dsw/data/cfd-database
tar -cf "$TAR_TRAIN" training_9k/
echo "    Tar size: $(du -h "$TAR_TRAIN" | cut -f1)"

echo "==> [2/2] Transferring to OCI..."
rsync -av --progress -e "$SSH_OPTS" "$TAR_TRAIN" "$OCI_TARGET:$OCI_DST/"

echo "==> [2/2] Removing local tar"
rm "$TAR_TRAIN"

echo
echo "==> All transfers complete"
echo "==> On Windows VM: rclone copy --http-url http://wx-outdoor.com:8080 :http:tars H:\\dsw\\tars\\ --multi-thread-streams 8 --progress"
