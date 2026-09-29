#!/usr/bin/env bash
# Two-direction ablation of the submission on the reference GPU (run from the repo root after bench_l40s_v2.sh).
#   speedup/record/ablate_l40s.sh [reps=3] [epochs=10]
# configs: base (upstream #10 file), full (submission), all-off, full minus X (leave-one-out), all-off plus X (add-one)
# for X in ATTN GRAPH MUONF BF16 COMPILE. Timing only (the ablation copy skips evaluation; base keeps its untimed eval): epoch 1 and median steady epoch per run.
set -uo pipefail
reps=${1:-3}; epochs=${2:-10}
tag=abl-$(date +%y%m%d-%H%M%S)-$(hostname -s)
UV=${UV:-$(command -v uv || echo ~/.local/bin/uv)}
X="ATTN GRAPH MUONF BF16 COMPILE"
configs="base full alloff"; for x in $X; do configs+=" -$x +$x"; done
envs() {  # config -> ABL_ env assignments
  case $1 in
    full) echo "";;
    alloff) for x in $X; do printf 'ABL_%s=0 ' $x; done;;
    -*) echo "ABL_${1#-}=0";;
    +*) for x in $X; do [ "$x" != "${1#+}" ] && printf 'ABL_%s=0 ' $x; done;;
  esac
}
run() {  # $1 config $2 name
  if [ "$1" = base ]; then $UV run python train_nano.py --name "$2" --epochs "$epochs"
  else env ABL_EVAL=0 $(envs "$1") $UV run python records/20260928launchbound/train_nano_abl.py --name "$2" --epochs "$epochs"; fi
}
echo "== warmup (not counted)"
for cfg in $configs; do run "$cfg" "$tag-warm$cfg" > /dev/null 2>&1; echo "warm $cfg exit=$?"; done
for r in $(seq 1 "$reps"); do
  for cfg in $configs; do
    run "$cfg" "$tag-$cfg-r$r" > /dev/null 2>&1
    log=$(ls -t workdir/experiments/"$tag-$cfg-r$r"/*/*-log.txt 2>/dev/null | head -1)
    e1=$(grep -E '^e:1/' "$log" | grep -oE ' e_t:[0-9.]+' | cut -d: -f2)
    st=$(grep -E '^e:' "$log" | tail -n +3 | grep -oE ' e_t:[0-9.]+' | cut -d: -f2 | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
    echo "rep $r $cfg: epoch1=$e1 steady=$st"
  done
done
