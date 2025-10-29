# generate_seeds.py
import random
import os
from datetime import datetime


def generate_seeds(n=10, min_seed=0, max_seed=1_000_000):
    return random.sample(range(min_seed, max_seed), n)


if __name__ == "__main__":
    seeds = generate_seeds()

    configs_dir = os.path.join(os.getcwd(), "configs")
    os.makedirs(configs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"seeds_{timestamp}.txt"
    filepath = os.path.join(configs_dir, filename)

    with open(filepath, "w") as f:
        for s in seeds:
            f.write(f"{s}\n")

    print(f"Generated {len(seeds)} seeds and saved to: {filepath}")
    print("Seeds:", seeds)
