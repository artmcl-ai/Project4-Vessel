import json, sys, pathlib

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "metrics.json")
with path.open() as f:
    m = json.load(f)

print("=== Dice Summary ===")
fg = m.get("foreground_mean", {})
if "Dice" in fg:
    print(f"Foreground mean Dice: {fg['Dice']:.6f}")

mean = m.get("mean", {})
for cls_id, stats in mean.items():
    if isinstance(stats, dict) and "Dice" in stats:
        print(f"Class {cls_id} mean Dice: {stats['Dice']:.6f}")

cases = m.get("metric_per_case", [])
if cases:
    print("\nPer-case (first 5):")
    for c in cases[:5]:
        pf = pathlib.Path(c.get("prediction_file", "")).name
        d1 = c["metrics"].get("1", {}).get("Dice", None)
        d2 = c["metrics"].get("2", {}).get("Dice", None)
        print(f"  {pf}: class1={d1:.4f}  class2={d2:.4f}")
