#!/usr/bin/env python3
import subprocess, itertools, shutil

RUNNER = "src/runner.cu"
LOGFILE = "tune_2d_results.txt"

# parameter ranges
BMs = [128, 256]
BNs = [128, 256]
BKs = [8, 16, 32]
TMs = [8, 16, 32]
TNs = [8, 16, 32]

# read & backup original
with open(RUNNER, 'r') as f:
    orig = f.readlines()
shutil.copyfile(RUNNER, RUNNER + ".bak")

# prepare log
with open(LOGFILE, 'w') as log:
    log.write("=== 2D blocktile autotuning ===\n")

# loop through all combos
for BM, BN, BK, TM, TN in itertools.product(BMs, BNs, BKs, TMs, TNs):
    # same validity checks as in your C
    total = BM*BN
    tile  = TM*TN
    if total % tile:               continue
    T = total // tile
    if T<=0 or T>1024:             continue
    if (T*4)%BK:                   continue
    if (T*4)%BN:                   continue
    if BN%(16*TN):                 continue
    if BM%(16*TM):                 continue
    if (BM*BK)%(4*T):              continue
    if (BN*BK)%(4*T):              continue

    # patch lines 61-65 (0-based indices 60–64)
    mod = orig.copy()
    mod[60] = f"  const uint BK = {BK};\n"
    mod[61] = f"  const uint TM = {TM};\n"
    mod[62] = f"  const uint TN = {TN};\n"
    mod[63] = f"  const uint BM = {BM};\n"
    mod[64] = f"  const uint BN = {BN};\n"
    with open(RUNNER, 'w') as f:
        f.writelines(mod)

    # rebuild
    subprocess.run("cmake -B build/ -S . && cmake --build build/", shell=True, check=True)

    # run kernel #4 (2D blocktiling)
    proc = subprocess.run("./build/gemm 4", shell=True, capture_output=True, text=True)

    # log results
    with open(LOGFILE, 'a') as log:
        log.write(f"\n--- BM={BM} BN={BN} BK={BK} TM={TM} TN={TN} ---\n")
        log.write(proc.stdout)
        if proc.stderr.strip():
            log.write("\n[stderr]\n")
            log.write(proc.stderr)

# restore original runner.cu
shutil.move(RUNNER + ".bak", RUNNER)

print(f"done. results in {LOGFILE}")
