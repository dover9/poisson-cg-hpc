# utils.py
from pathlib import Path
import matplotlib.pyplot as plt

# Start from the directory this file lives in
HERE = Path(__file__).resolve().parent

# Try to detect the project root:
# - If there's a "results" directory right next to this file, use HERE.
# - Else, if there's a "results" directory one level up, use the parent.
if (HERE / "results").is_dir():
    ROOT = HERE
elif (HERE.parent / "results").is_dir():
    ROOT = HERE.parent
else:
    # Fallback: just use HERE for everything
    ROOT = HERE

RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

# Make sure figures/ exists; results/ is optional but recommended
FIG_DIR.mkdir(parents=True, exist_ok=True)

def savefig(name: str):
    """Save current Matplotlib figure into figures/ with high DPI."""
    path = FIG_DIR / name
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")