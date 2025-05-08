#!/usr/bin/env python3
import os, re, subprocess

# —————— user‐configurable ranges ——————
BK_vals = [8, 16, 32, 64]
TM_vals = [4, 8, 16, 32]
TN_vals = [4, 8, 16, 32]
BM_vals = [64, 128, 256]
BN_vals = [64, 128, 256]

# path to your runner
RUNNER = os.path.join("src", "runner.cu")

# your build + run commands
BUILD_CMD = "cmake -B build/ -S . && cmake --build build/"
RUN_CMD   = "./build/gemm 4"

# GPU limit on SMEM
MAX_SHARED = 48 * 1024  # bytes

def patch_runner(bk, tm, tn, bm, bn):
    with open(RUNNER, "r") as f:
        txt = f.read()
    # rewrite each tile_* const
    txt = re.sub(r"const uint tile_BK\s*=\s*\d+;",
                 f"const uint tile_BK = {bk};", txt)
    txt = re.sub(r"const uint tile_TM\s*=\s*\d+;",
                 f"const uint tile_TM = {tm};", txt)
    txt = re.sub(r"const uint tile_TN\s*=\s*\d+;",
                 f"const uint tile_TN = {tn};", txt)
    txt = re.sub(r"const uint tile_BM\s*=\s*\d+;",
                 f"const uint tile_BM = {bm};", txt)
    txt = re.sub(r"const uint tile_BN\s*=\s*\d+;",
                 f"const uint tile_BN = {bn};", txt)
    with open(RUNNER, "w") as f:
        f.write(txt)

def main():
    os.chdir(os.path.dirname(__file__) or ".")
    total = 0
    for bk in BK_vals:
      for tm in TM_vals:
        for tn in TN_vals:
          for bm in BM_vals:
            for bn in BN_vals:
              # 1) tile dims must divide evenly
              if (bm*bn) % (tm*tn) != 0: continue
              T = (bm*bn)//(tm*tn)
              # 2) reasonable thread‐count & quantization checks
              if not (0 < T <= 1024): continue
              if (T*4) % bk != 0:   continue
              if (T*4) % bn != 0:   continue
              if bn % (16*tn)  != 0: continue
              if bm % (16*tm)  != 0: continue
              if (bm*bk) % (4*T) != 0: continue
              if (bn*bk) % (4*T) != 0: continue
              # 3) shared‐mem requirement
              shared_bytes = (bm + bn) * bk * 4
              if shared_bytes > MAX_SHARED:
                print(f" SKIP BK={bk} TM={tm} TN={tn} BM={bm} BN={bn} → "
                      f"{shared_bytes//1024} KiB > 48 KiB")
                continue

              total += 1
    print(f"Will try {total} configs\n")

    config_i = 0
    for bk in BK_vals:
      for tm in TM_vals:
        for tn in TN_vals:
          for bm in BM_vals:
            for bn in BN_vals:
              if (bm*bn) % (tm*tn) != 0: continue
              T = (bm*bn)//(tm*tn)
              if not (0 < T <= 1024): continue
              if (T*4) % bk != 0:   continue
              if (T*4) % bn != 0:   continue
              if bn % (16*tn)  != 0: continue
              if bm % (16*tm)  != 0: continue
              if (bm*bk) % (4*T) != 0: continue
              if (bn*bk) % (4*T) != 0: continue
              shared_bytes = (bm + bn) * bk * 4
              if shared_bytes > MAX_SHARED: continue

              config_i += 1
              print(f"[{config_i}/{total}] BK={bk} TM={tm} TN={tn} BM={bm} BN={bn}")
              patch_runner(bk, tm, tn, bm, bn)
              try:
                  subprocess.run(BUILD_CMD, shell=True, check=True)
                  subprocess.run(RUN_CMD,   shell=True, check=True)
              except subprocess.CalledProcessError:
                  print("    → build or run failed, skipping\n")
                  continue

if __name__ == "__main__":
    main()
