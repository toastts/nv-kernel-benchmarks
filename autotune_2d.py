#!/usr/bin/env python3
import itertools
import subprocess

# path to your runner.cu
RUNNER = "src/runner.cu"
LOGFILE = "tune_2d_results.txt"

# ranges to sweep
BMs = [64, 128, 256]
BNs = [64, 128, 256]
BKs = [8, 16, 32, 64]
TMs = [4, 8, 16, 32]
TNs = [4, 8, 16, 32]

def is_valid(BM, BN, BK, TM, TN):
    total = BM * BN
    tile_sz = TM * TN
    if total % tile_sz != 0: return False
    T = total // tile_sz
    if T <= 0 or T > 1024: return False
    if (T*4) % BK != 0: return False
    if (T*4) % BN != 0: return False
    if BN % (16*TN) != 0: return False
    if BM % (16*TM) != 0: return False
    if (BM*BK) % (4*T) != 0: return False
    if (BN*BK) % (4*T) != 0: return False
    return True

# start fresh log
with open(LOGFILE, "w") as log:
    log.write("=== 2D blocktile autotune (2048x2048 only) ===\n")

for BM, BN, BK, TM, TN in itertools.product(BMs, BNs, BKs, TMs, TNs):
    if not is_valid(BM, BN, BK, TM, TN):
        continue

    # read runner.cu
    with open(RUNNER) as f:
        lines = f.readlines()

    # rewrite just the tile_* lines
    out = []
    for l in lines:
        if "const uint tile_BK" in l:
            out.append(f"  const uint tile_BK = {BK};\n")
        elif "const uint tile_TM" in l:
            out.append(f"  const uint tile_TM = {TM};\n")
        elif "const uint tile_TN" in l:
            out.append(f"  const uint tile_TN = {TN};\n")
        elif "const uint tile_BM" in l:
            out.append(f"  const uint tile_BM = {BM};\n")
        elif "const uint tile_BN" in l:
            out.append(f"  const uint tile_BN = {BN};\n")
        else:
            out.append(l)
    with open(RUNNER, "w") as f:
        f.writelines(out)

    # rebuild
    print(f"Building BM={BM} BN={BN} BK={BK} TM={TM} TN={TN}...")
    subprocess.run("cmake -B build/ -S . && cmake --build build/", shell=True, check=True)

    # run kernel #4 (2D blocktiling) on the fixed 4096×4096 workload
    print("Running ./build/gemm 4 ...")
    proc = subprocess.run("./build/gemm 4", shell=True, capture_output=True, text=True)

    # append to log
    with open(LOGFILE, "a") as log:
        log.write(f"\n--- BM={BM} BN={BN} BK={BK} TM={TM} TN={TN} ---\n")
        log.write(proc.stdout)
        if proc.stderr.strip():
            log.write("\n[stderr]\n")
            log.write(proc.stderr)

print("autotune complete, results in", LOGFILE)
