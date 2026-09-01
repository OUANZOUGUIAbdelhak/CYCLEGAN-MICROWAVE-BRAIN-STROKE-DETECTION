#!/bin/bash
# Self-resuming training driver for Phase 4. It resumes from the last checkpoint
# until it reaches --iters, so any stop only costs the iters since the last save.
# Progress is printed to the terminal and also saved to a log. Leave the tab open.
cd "$(dirname "$0")"
CKDIR=outputs/4_cyclegan
LOG=$CKDIR/main_train.log
PY=.venv/bin/python
TARGET=50000

reached() {
  [ -f $CKDIR/main_latest.pt ] && \
  $PY -c "import torch;print(torch.load('$CKDIR/main_latest.pt',map_location='cpu')['iter'])" 2>/dev/null
}

echo "=== Self-resuming CycleGAN training -> $TARGET iterations ==="
echo "    (leave this tab open; it resumes automatically if interrupted)"
echo
for attempt in $(seq 1 40); do
  it=$(reached); it=${it:-0}
  if [ "$it" -ge "$TARGET" ]; then
     echo ""; echo "=========================================="
     echo "  REACHED $it >= $TARGET.  TRAINING DONE."
     echo "=========================================="
     break
  fi
  if [ "$it" -gt 0 ]; then RESUME="--resume"; else RESUME=""; fi
  echo "[driver] attempt $attempt: starting/resuming from iter $it ..."
  PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -dimsu $PY src/phase4_train.py \
    --iters $TARGET --batch 2 --f_out 128 --lambda_cyc 10 --lambda_id 5 --lambda_sup 15 \
    --lr 2e-4 --clip 1.0 --inst_noise 0.05 $RESUME \
    --tag main --sample_every 2500 --ckpt_every 1000 2>&1 | tee -a "$LOG"
  echo "[driver] training exited at $(date); will check progress and resume if needed."
  sleep 3
done
