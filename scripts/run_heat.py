import os, argparse, runpy
from pathlib import Path


os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt  

p = argparse.ArgumentParser()
p.add_argument("--file", required=True)        
p.add_argument("--out", default="outputs")     
args = p.parse_args()

print(f"[runner] CWD={os.getcwd()}")
print(f"[runner] running {args.file}")


runpy.run_path(args.file, run_name="__main__")


fig_nums = plt.get_fignums()
print(f"[runner] figures found: {fig_nums}")
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
for n in fig_nums:
    plt.figure(n).savefig(out / f"figure_{n}.png", dpi=150, bbox_inches="tight")
plt.close("all")

print(f"[runner] figure saved to {out.resolve()}")