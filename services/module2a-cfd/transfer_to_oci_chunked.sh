#!/bin/bash
# transfer_to_oci_chunked.sh — Tar campaign_9k + training_9k on UGA, transfer to OCI in chunks.
#
# Strategy: tar locally on UGA (fast, native), then rsync the tar file to OCI.
# A single large file transfers MUCH faster than thousands of small Zarr chunks.
#
# Usage (run on UGA, ideally inside tmux):
#   bash transfer_to_oci_chunked.sh <chunk_id>
#   bash transfer_to_oci_chunked.sh 1     # Sites 0..149
#   bash transfer_to_oci_chunked.sh 2     # Sites 150..299
#   bash transfer_to_oci_chunked.sh 3     # Sites 300..449
#   bash transfer_to_oci_chunked.sh 4     # Sites 450..599
#
# Each chunk (~150 sites) produces ~21 GB raw tar + ~43 GB training tar = ~64 GB
# stays well under the 150 GB / 250 GB OCI disk limit.
#
# Workflow:
#   1. UGA tars one part (raw or training) → rsync to OCI → delete local tar
#   2. Repeat for second part
#   3. After chunk N: download to Aqua VM, delete tars on OCI, then run chunk N+1

set -euo pipefail

CHUNK="${1:?Usage: $0 <chunk_id 1|2|3|4>}"

SSH_KEY="$HOME/.ssh/id_oci"
OCI_TARGET="enjoy@wx-outdoor.com"
OCI_DST="/data/tars"

UGA_TAR_DIR="$HOME/dsw/tars_tmp"
mkdir -p "$UGA_TAR_DIR"

case "$CHUNK" in
    1) START=0;   END=149 ; LABEL='0..149'  ;;
    2) START=150; END=299 ; LABEL='150..299';;
    3) START=300; END=449 ; LABEL='300..449';;
    4) START=450; END=599 ; LABEL='450..599';;
    *) echo "Error: chunk_id must be 1..4"; exit 1 ;;
esac

# Build list of site IDs for this chunk (zero-padded 5 digits)
SITE_IDS=()
for i in $(seq $START $END); do
    SITE_IDS+=("$(printf 'site_%05d' $i)")
done

SSH_OPTS="ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new"

echo "==> Chunk $CHUNK: sites $LABEL"
echo "==> Strategy: tar on UGA → rsync single file → OCI"

# Ensure target directory exists on OCI
$SSH_OPTS "$OCI_TARGET" "mkdir -p $OCI_DST"

# Check OCI free space
OCI_FREE=$($SSH_OPTS "$OCI_TARGET" "df -BG /data | awk 'NR==2 {print \$4}' | tr -d 'G'")
echo "==> OCI free space: ${OCI_FREE} GB"
if [ "$OCI_FREE" -lt 70 ]; then
    echo "WARNING: less than 70 GB free on OCI. Make sure previous chunks were deleted."
    read -p "Continue anyway? (y/N) " confirm
    [ "$confirm" = "y" ] || exit 1
fi

# ─────────────────────────────────────────────────────────────────────
# Helper: tar one part, rsync, delete local
# ─────────────────────────────────────────────────────────────────────
tar_and_transfer() {
    local label="$1"           # "campaign" or "training"
    local source_dir="$2"      # path relative to cwd (e.g. "campaign_9k")
    local mode="$3"            # "exact" (campaign) or "prefix" (training)
    local tar_file="$4"        # output tar path

    echo
    echo "==> [$label] Building file list from ${#SITE_IDS[@]} site IDs..."

    > /tmp/tar_list_$$.txt
    for sid in "${SITE_IDS[@]}"; do
        if [ "$mode" = "exact" ]; then
            # campaign_9k/site_XXXXX (one dir per site)
            local d="$source_dir/$sid"
            [ -d "$d" ] && echo "$d" >> /tmp/tar_list_$$.txt
        else
            # training_9k/site_XXXXX_case_tsYYY (multiple dirs per site)
            for d in "$source_dir/${sid}"_*; do
                [ -d "$d" ] && echo "$d" >> /tmp/tar_list_$$.txt
            done
        fi
    done

    local n=$(wc -l < /tmp/tar_list_$$.txt)
    echo "    Items to archive: $n"

    if [ "$n" -eq 0 ]; then
        echo "    Nothing to archive, skipping"
        rm -f /tmp/tar_list_$$.txt
        return
    fi

    echo "==> [$label] Creating $tar_file"
    tar -cf "$tar_file" --files-from=/tmp/tar_list_$$.txt
    rm /tmp/tar_list_$$.txt
    echo "    Tar size: $(du -h "$tar_file" | cut -f1)"

    echo "==> [$label] Transferring to OCI..."
    rsync -av --progress -e "$SSH_OPTS" "$tar_file" "$OCI_TARGET:$OCI_DST/"

    echo "==> [$label] Removing local tar (UGA)"
    rm "$tar_file"
}

# ─────────────────────────────────────────────────────────────────────
# Part 1: campaign_9k (raw OF + stacked Zarr)
# ─────────────────────────────────────────────────────────────────────
cd ~/dsw
tar_and_transfer "campaign" "campaign_9k" "exact" \
    "$UGA_TAR_DIR/campaign_9k_chunk${CHUNK}.tar"

# ─────────────────────────────────────────────────────────────────────
# Part 2: training_9k (per-case grid+unstructured Zarr)
# ─────────────────────────────────────────────────────────────────────
cd ~/dsw/data/cfd-database
tar_and_transfer "training" "training_9k" "prefix" \
    "$UGA_TAR_DIR/training_9k_chunk${CHUNK}.tar"

# Bonus: include dataset.yaml on chunk 1 (small files)
if [ "$CHUNK" = "1" ]; then
    echo
    echo "==> [meta] Transferring dataset.yaml + dataset.csv"
    rsync -av --progress -e "$SSH_OPTS" \
        training_9k/dataset.yaml training_9k/dataset.csv \
        "$OCI_TARGET:$OCI_DST/" 2>/dev/null || true
fi

echo
echo "==> Chunk $CHUNK transfer complete"
echo "==> Tar files now on OCI under $OCI_DST/"
echo "==> Verify:  ssh -i $SSH_KEY $OCI_TARGET 'ls -lh $OCI_DST/'"
echo
echo "==> NEXT STEPS:"
echo "    1. On Windows VM: rclone copy --http-url http://wx-outdoor.com:8080 :http:tars H:\\dsw\\tars\\ --transfers 4 --multi-thread-streams 8 --progress"
echo "    2. Extract on Aqua: tar -xf H:/dsw/tars/*chunk${CHUNK}.tar -C H:/dsw/"
echo "    3. Delete on OCI: ssh -i $SSH_KEY $OCI_TARGET 'rm $OCI_DST/*chunk${CHUNK}.tar'"
echo "    4. Run next chunk: bash $0 $((CHUNK+1))"
