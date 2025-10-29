#!/usr/bin/env python3
import json, sys, re
from pathlib import Path
from statistics import mean

MIN_DELTA = 1e-3
PATIENCE  = 30
MA_WINDOW = 5

def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)

def find_training_log(dir_path: Path):
    cands = list(dir_path.rglob("training_log.json"))
    return cands[0] if cands else None

def find_metrics_json(dir_path: Path):
    cands = list(dir_path.rglob("metrics.json"))
    return cands[0] if cands else None

def extract_dice_series_from_training_log(obj):
    series = []
    if isinstance(obj, list):
        for ep in sorted(obj, key=lambda x: x.get('epoch', 0)):
            dice_keys = [k for k in ep.keys()
                         if re.search(r"dice", k, re.I)]
            if dice_keys:
                for pref in ["pseudo_dice", "mean_fg_dice", "val_dice", "mean_dice", "dice"]:
                    if pref in ep:
                        series.append(float(ep[pref]))
                        break
                else:
                    series.append(float(ep[dice_keys[0]]))
    elif isinstance(obj, dict):
        def flatten_dice(d):
            seq = None
            for k, v in d.items():
                if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v) and "dice" in k.lower():
                    seq = v; break
            if seq is None:
                for v in d.values():
                    if isinstance(v, dict):
                        inner = flatten_dice(v)
                        if inner is not None: return inner
            return seq
        seq = flatten_dice(obj)
        if seq: series = list(map(float, seq))
    return series

def extract_dice_series_from_metrics(obj):
    for k in ["mean_fg_dice", "mean_dice", "dice", "pseudo_dice", "val_dice"]:
        if k in obj and isinstance(obj[k], list):
            return list(map(float, obj[k]))
        if k in obj and isinstance(obj[k], (int, float)):
            return [float(obj[k])]
    def dfs(o):
        if isinstance(o, dict):
            for kk, vv in o.items():
                if isinstance(vv, list) and all(isinstance(x, (int,float)) for x in vv) and "dice" in kk.lower():
                    return list(map(float, vv))
                got = dfs(vv)
                if got: return got
        return None
    got = dfs(obj)
    return got or []

def moving_average(xs, w):
    if w <= 1: return xs[:]
    out = []
    for i in range(len(xs)):
        l = max(0, i - w + 1)
        out.append(mean(xs[l:i+1]))
    return out

def find_best_epoch(dice):
    best_idx = max(range(len(dice)), key=lambda i: dice[i])
    return best_idx, dice[best_idx]

def find_early_stop_epoch(dice_ma, min_delta=MIN_DELTA, patience=PATIENCE):
    best = -1e9
    stale = 0
    for i, v in enumerate(dice_ma):
        if v > best + min_delta:
            best = v
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                return max(0, i - patience + 1)
    return None 

def main():
    if len(sys.argv) != 2:
        print("Usage: python find_convergence.py /path/to/.../fold_X", file=sys.stderr)
        sys.exit(1)
    fold_dir = Path(sys.argv[1]).resolve()
    if not fold_dir.exists():
        print(f"[ERR] Not found: {fold_dir}", file=sys.stderr); sys.exit(2)

    log_path = find_training_log(fold_dir)
    met_path = find_metrics_json(fold_dir)

    dice = []
    src = None
    if log_path:
        try:
            dice = extract_dice_series_from_training_log(load_json(log_path))
            src = f"{log_path.name}"
        except Exception as e:
            print(f"[WARN] Failed to parse {log_path}: {e}")
    if not dice and met_path:
        try:
            dice = extract_dice_series_from_metrics(load_json(met_path))
            src = f"{met_path.name}"
        except Exception as e:
            print(f"[WARN] Failed to parse {met_path}: {e}")

    if not dice:
        print("[ERR] Could not find a dice series in logs/metrics.", file=sys.stderr)
        sys.exit(3)

    best_ep, best_val = find_best_epoch(dice)
    dice_ma = moving_average(dice, MA_WINDOW)
    es_ep = find_early_stop_epoch(dice_ma, MIN_DELTA, PATIENCE)

    print("=== Convergence Report ===")
    print(f"source              : {src}")
    print(f"num_epochs(parsed)  : {len(dice)}")
    print(f"best_epoch          : {best_ep} (0-based), dice={best_val:.4f}")
    if es_ep is not None:
        print(f"early_stop_epoch    : {es_ep} (0-based, MA window={MA_WINDOW}, min_delta={MIN_DELTA}, patience={PATIENCE})")
        print(f"early_stop_dice(MA) : {dice_ma[es_ep]:.4f}")
    else:
        print("early_stop_epoch    : None (did not meet patience/min_delta criteria)")
    suggest = min(best_ep, es_ep) if es_ep is not None else best_ep
    print(f"suggested_stop_epoch: {suggest} (0-based)")
    print(f"(params: MA_WINDOW={MA_WINDOW}, MIN_DELTA={MIN_DELTA}, PATIENCE={PATIENCE})")

if __name__ == "__main__":
    main()
