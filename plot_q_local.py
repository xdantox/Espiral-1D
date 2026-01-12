import os
import numpy as np
import matplotlib.pyplot as plt


def compute_q_local_from_last_frame(spin_history_path: str) -> np.ndarray:
    sh = np.load(spin_history_path, mmap_mode="r")
    spins = np.array(sh[-1], dtype=float)  # (N,3)
    theta = np.unwrap(np.arctan2(spins[:, 2], spins[:, 1]))
    q_local = np.diff(theta)
    return q_local


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    q_path = "spin_history_q_local.npy"
    sh_path = "J_perp n=1198 relax.npy"

    if os.path.exists(q_path):
        q_local = np.load(q_path)
        source = q_path
    elif os.path.exists(sh_path):
        q_local = compute_q_local_from_last_frame(sh_path)
        source = f"{sh_path} (last frame)"
    else:
        raise FileNotFoundError("Expected spin_history_q_local.npy or spin_history.npy in current folder")

    q_local = np.asarray(q_local, dtype=float)
    x = np.arange(q_local.size)

    plt.figure(figsize=(10, 4.2))
    plt.plot(x, q_local, lw=1.0)
    plt.axhline(0.0, color="k", lw=0.7, alpha=0.6)
    plt.title(f"q_local por enlace (fuente: {source})")
    plt.xlabel("índice de enlace i (entre sitio i e i+1)")
    plt.ylabel("q_local [rad]")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    out = "q_local.png"
    plt.savefig(out, dpi=200)
    plt.show()
    print(f"Saved: {out} | q_local shape={q_local.shape}")

    return q_local
if __name__ == "__main__":
    q=main()
