import os, argparse, runpy
from pathlib import Path

# set headless backend BEFORE importing pyplot
os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt  # after backend set

p = argparse.ArgumentParser()
p.add_argument("--file", required=True)        # e.g., src/PartA1.py
p.add_argument("--out", default="outputs")     # output folder
args = p.parse_args()

print(f"[runner] CWD={os.getcwd()}")
print(f"[runner] running {args.file}")

# run your script as __main__ so code under that guard executes
runpy.run_path(args.file, run_name="__main__")

# save any open figures
fig_nums = plt.get_fignums()
print(f"[runner] figures found: {fig_nums}")
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
for n in fig_nums:
    plt.figure(n).savefig(out / f"figure_{n}.png", dpi=150, bbox_inches="tight")
plt.close("all")

if fig_nums:
    print(f"[runner] saved to {out.resolve()}")
else:
    print("[runner] no figures to save")
