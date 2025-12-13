import json
import matplotlib.pyplot as plt

with open("data/nnUNet_results/Dataset001_nnunet/preds/ensemble_seed1-9/metrics.json") as f:
    m = json.load(f)

dice1 = [c["metrics"]["1"]["Dice"] for c in m["metric_per_case"]]
dice2 = [c["metrics"]["2"]["Dice"] for c in m["metric_per_case"]]

plt.figure(figsize=(8,4))
plt.plot(dice1, "o-", label="Artery")
plt.plot(dice2, "s-", label="Vein")
plt.axhline(0.8856, color="gray", linestyle="--", label="Mean Dice (0.8856)")
plt.title("Per-case Dice Scores (10-Seed Soft-Voting Ensemble)")
plt.xlabel("Test Case Index")
plt.ylabel("Dice Coefficient")
plt.legend()
plt.tight_layout()
plt.savefig("ensemble_dice_curve.png", dpi=300)
plt.show()
