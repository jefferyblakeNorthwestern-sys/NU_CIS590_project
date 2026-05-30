#!/usr/bin/env bash
# run_all.sh
# Full BATADAL detection pipeline — cleans, trains, detects, scores, and
# runs per-window analysis across all 8 experimental configurations.
#
# Usage: bash run_all.sh

set -e
PYTHON="python3.11"
SCRIPTS="anomaly-detection/scripts"
DATA="data"
REPORTS="reports"

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

header()  { echo -e "\n${BLUE}══════════════════════════════════════════════${NC}"; echo -e "${BLUE}  $1${NC}"; echo -e "${BLUE}══════════════════════════════════════════════${NC}"; }
success() { echo -e "${GREEN}  ✓  $1${NC}"; }
info()    { echo -e "${YELLOW}  →  $1${NC}"; }

# ── Step 1: Clean everything ──────────────────────────────────────────────────
header "Step 1 — Clean previous outputs"
read -p "  Delete all reports/ and trained_model*/ directories? (y/n): " confirm
if [[ "$confirm" != "y" ]]; then
    echo "  Aborted."
    exit 0
fi

rm -rf reports/
rm -rf trained_model/
rm -rf trained_model_d2_labeled/
rm -rf trained_model_test_labeled/

mkdir -p reports/run1
mkdir -p reports/run4
mkdir -p reports/run5
mkdir -p reports/run8
mkdir -p reports/window_8
mkdir -p reports/window_9
mkdir -p reports/window_10
mkdir -p reports/window_11
mkdir -p reports/window_12
mkdir -p reports/window_13
mkdir -p reports/window_14

success "All previous outputs cleared and directories created"

# ── Step 2: Generate windows JSON files ───────────────────────────────────────
header "Step 2 — Generate windows JSON files"
if [[ -f "$DATA/BATADAL_dataset2_windows.json" && -f "$DATA/BATADAL_test_windows.json" ]]; then
    info "Windows JSON files already exist — skipping"
else
    $PYTHON $SCRIPTS/make_windows.py \
        --dataset2-labels $DATA/BATADAL_dataset2_attack_list.csv \
        --test-labels     $DATA/BATADAL_test_attack_list.csv \
        --output-dir      $DATA/
    success "Windows JSON files generated"
fi

# ── Step 3: Training ──────────────────────────────────────────────────────────
header "Step 3 — Training"
read -p "  Run all three training configurations? (y/n): " train_confirm
if [[ "$train_confirm" != "y" ]]; then
    echo "  Skipping training — using existing trained_model/ directories"
else
    info "Training Run 1/3 — baseline (no labels) ..."
    $PYTHON $SCRIPTS/train.py \
        --input  $DATA/BATADAL_training_dataset_1.CSV \
        --output trained_model/
    success "trained_model/ complete"

    info "Training Run 2/3 — Dataset 2 labeled ..."
    $PYTHON $SCRIPTS/train.py \
        --input            $DATA/BATADAL_training_dataset_1.CSV \
        --output           trained_model_d2_labeled/ \
        --exclude-windows  $DATA/BATADAL_dataset2_windows.json
    success "trained_model_d2_labeled/ complete"

    info "Training Run 3/3 — Test dataset labeled ..."
    $PYTHON $SCRIPTS/train.py \
        --input            $DATA/BATADAL_training_dataset_1.CSV \
        --output           trained_model_test_labeled/ \
        --exclude-windows  $DATA/BATADAL_test_windows.json
    success "trained_model_test_labeled/ complete"
fi

# ── Step 4: Detection — 8 experimental runs ───────────────────────────────────
header "Step 4 — Detection (8 experimental runs)"

info "Run 1 — Dataset 2, no labels ..."
$PYTHON $SCRIPTS/detect.py \
    --input  $DATA/BATADAL_training_dataset_2.CSV \
    --model  trained_model/ \
    --output reports/run1/
success "Run 1 complete"

info "Run 4 — Dataset 2, with labels ..."
$PYTHON $SCRIPTS/detect.py \
    --input  $DATA/BATADAL_training_dataset_2.CSV \
    --model  trained_model_d2_labeled/ \
    --output reports/run4/
success "Run 4 complete"

info "Run 5 — Test dataset, no labels ..."
$PYTHON $SCRIPTS/detect.py \
    --input  $DATA/BATADAL_test_dataset.CSV \
    --model  trained_model/ \
    --output reports/run5/
success "Run 5 complete"

info "Run 8 — Test dataset, with labels ..."
$PYTHON $SCRIPTS/detect.py \
    --input  $DATA/BATADAL_test_dataset.CSV \
    --model  trained_model_test_labeled/ \
    --output reports/run8/
success "Run 8 complete"

# ── Step 5: Windowed detection — per attack window ────────────────────────────
header "Step 5 — Windowed detection (per attack window, test dataset)"

info "Window 8  — 2017-01-16 09:00 to 2017-01-19 06:00 ..."
$PYTHON $SCRIPTS/detect.py --input $DATA/BATADAL_test_dataset.CSV --model trained_model/ --output reports/window_8/  --window-start "2017-01-16 09:00" --window-end "2017-01-19 06:00"
success "Window 8 complete"

info "Window 9  — 2017-01-30 08:00 to 2017-02-02 00:00 ..."
$PYTHON $SCRIPTS/detect.py --input $DATA/BATADAL_test_dataset.CSV --model trained_model/ --output reports/window_9/  --window-start "2017-01-30 08:00" --window-end "2017-02-02 00:00"
success "Window 9 complete"

info "Window 10 — 2017-02-09 03:00 to 2017-02-10 09:00 ..."
$PYTHON $SCRIPTS/detect.py --input $DATA/BATADAL_test_dataset.CSV --model trained_model/ --output reports/window_10/ --window-start "2017-02-09 03:00" --window-end "2017-02-10 09:00"
success "Window 10 complete"

info "Window 11 — 2017-02-12 01:00 to 2017-02-13 07:00 ..."
$PYTHON $SCRIPTS/detect.py --input $DATA/BATADAL_test_dataset.CSV --model trained_model/ --output reports/window_11/ --window-start "2017-02-12 01:00" --window-end "2017-02-13 07:00"
success "Window 11 complete"

info "Window 12 — 2017-02-24 05:00 to 2017-02-28 08:00 ..."
$PYTHON $SCRIPTS/detect.py --input $DATA/BATADAL_test_dataset.CSV --model trained_model/ --output reports/window_12/ --window-start "2017-02-24 05:00" --window-end "2017-02-28 08:00"
success "Window 12 complete"

info "Window 13 — 2017-03-10 14:00 to 2017-03-13 21:00 ..."
$PYTHON $SCRIPTS/detect.py --input $DATA/BATADAL_test_dataset.CSV --model trained_model/ --output reports/window_13/ --window-start "2017-03-10 14:00" --window-end "2017-03-13 21:00"
success "Window 13 complete"

info "Window 14 — 2017-03-25 20:00 to 2017-03-27 01:00 ..."
$PYTHON $SCRIPTS/detect.py --input $DATA/BATADAL_test_dataset.CSV --model trained_model/ --output reports/window_14/ --window-start "2017-03-25 20:00" --window-end "2017-03-27 01:00"
success "Window 14 complete"

# ── Step 6: Scoring — ablation ────────────────────────────────────────────────
header "Step 6 — Scoring (ablation study)"

info "Scoring Run 1 ..."
$PYTHON $SCRIPTS/score.py \
    --reports reports/run1/ \
    --labels  $DATA/BATADAL_dataset2_attack_list.csv \
    --data    $DATA/BATADAL_training_dataset_2.CSV \
    --ablation --detail

info "Scoring Run 4 ..."
$PYTHON $SCRIPTS/score.py \
    --reports reports/run4/ \
    --labels  $DATA/BATADAL_dataset2_attack_list.csv \
    --data    $DATA/BATADAL_training_dataset_2.CSV \
    --ablation --detail

info "Scoring Run 5 ..."
$PYTHON $SCRIPTS/score.py \
    --reports reports/run5/ \
    --labels  $DATA/BATADAL_test_attack_list.csv \
    --data    $DATA/BATADAL_test_dataset.CSV \
    --ablation --detail

info "Scoring Run 8 ..."
$PYTHON $SCRIPTS/score.py \
    --reports reports/run8/ \
    --labels  $DATA/BATADAL_test_attack_list.csv \
    --data    $DATA/BATADAL_test_dataset.CSV \
    --ablation --detail

success "All ablation scoring complete"

# ── Step 7: Scoring — disagreement analysis ───────────────────────────────────
header "Step 7 — Disagreement analysis"

info "Disagreement — Run 1 (Dataset 2) ..."
$PYTHON $SCRIPTS/score.py \
    --reports reports/run1/ \
    --labels  $DATA/BATADAL_dataset2_attack_list.csv \
    --data    $DATA/BATADAL_training_dataset_2.CSV \
    --disagreement

info "Disagreement — Run 5 (Test dataset) ..."
$PYTHON $SCRIPTS/score.py \
    --reports reports/run5/ \
    --labels  $DATA/BATADAL_test_attack_list.csv \
    --data    $DATA/BATADAL_test_dataset.CSV \
    --disagreement

info "Disagreement — Windowed reports (per attack window) ..."
for window in 8 9 10 11 12 13 14; do
    $PYTHON $SCRIPTS/score.py \
        --reports reports/window_${window}/ \
        --labels  $DATA/BATADAL_test_attack_list.csv \
        --data    $DATA/BATADAL_test_dataset.CSV \
        --disagreement
done

success "All disagreement analysis complete"

# ── Step 8: Collect all JSON outputs into one file ────────────────────────────
header "Step 8 — Collecting all results into reports/all_results.json"

python3.11 << 'PYEOF'
import json, os
from pathlib import Path

all_results = {
    "ablation": {},
    "disagreement": {},
    "windowed_disagreement": {}
}

# Ablation results
for run in ["run1", "run4", "run5", "run8"]:
    p = Path(f"reports/{run}/ablation_results.json")
    if p.exists():
        with open(p) as f:
            all_results["ablation"][run] = json.load(f)
        print(f"  collected ablation/{run}")

# Disagreement results — full runs
for run in ["run1", "run5"]:
    p = Path(f"reports/{run}/disagreement_results.json")
    if p.exists():
        with open(p) as f:
            all_results["disagreement"][run] = json.load(f)
        print(f"  collected disagreement/{run}")

# Disagreement results — per window
for window in range(8, 15):
    p = Path(f"reports/window_{window}/disagreement_results.json")
    if p.exists():
        with open(p) as f:
            all_results["windowed_disagreement"][f"window_{window}"] = json.load(f)
        print(f"  collected windowed_disagreement/window_{window}")

out = Path("reports/all_results.json")
with open(out, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n  Saved -> {out}  ({out.stat().st_size // 1024}KB)")
PYEOF

success "reports/all_results.json ready for analysis"

# ── Done ──────────────────────────────────────────────────────────────────────
header "Pipeline complete"
echo -e "  Upload ${GREEN}reports/all_results.json${NC} to Claude for analysis."
echo ""
