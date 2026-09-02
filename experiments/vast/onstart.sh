#!/bin/bash
# Vast.ai worker bootstrap for jax-solitons fleet runs.
#
# Rebuilt 2026-06-13: the old path (probe GitHub speed; `pip install jax[cuda12]`
# from PyPI) bit the RUNBOOK's documented #1 gotcha -- PyPI/Fastly AND GitHub are
# throttled on Vast datacenter IPs (~KB/s), while Cloudflare/CDN run at MB/s on the
# SAME hosts. So the GitHub probe FALSE-REJECTED good hosts and the PyPI install
# HUNG. This version uses the delivery paths that actually work:
#   - JAX+CUDA via conda-forge (anaconda.org rides Cloudflare, tens of MB/s)
#   - miniconda installer from repo.anaconda.com (CDN), since the nvidia/cuda
#     image has no conda
#   - probe the CONDA CDN (the path we use), not GitHub
# The engine (jax-solitons) is small, so pip-from-GitHub is tolerable.
# Env lands at /workspace/jaxenv -> driver runs /workspace/jaxenv/bin/python.
set -uo pipefail
mkdir -p /workspace
L=/workspace/onstart.log
log(){ echo "$(date +%T) $*" >> "$L"; }
log "onstart start"

MC=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 1. Probe the conda CDN (what we actually pull from). Bail on a throttled host.
SPD=$(curl -s -o /dev/null -w "%{speed_download}" --max-time 15 -L "$MC")
log "conda-cdn ${SPD}B/s"
if [ "${SPD%.*}" -lt 2000000 ]; then
    touch /tmp/worker-bad-network; log "BAD NET (conda cdn < 2MB/s)"; exit 1
fi

# 1b. A C toolchain. The nvidia/cuda *runtime* image ships none, and pyknotid
# pulls `planarity` -- a C extension with no manylinux wheel, so pip builds it from
# source. Without gcc that build fails, the `&&` chain in the leg command breaks
# before `cat > leg.sh`, and the leg dies as `rc=127: bash: leg.sh: No such file or
# directory` -- a compiler error wearing a shell error's clothes. Measured
# 2026-08-05: every leg of the first N_link ladder attempt died this way.
# Cheap and once per box, versus per-leg in the command.
#
# The GL libraries are for the same dependency and are just as non-optional.
# pyknotid.spacecurves imports pyknotid.visualise unconditionally, which imports
# vispy, which resolves a GL ES 2.0 backend AT IMPORT. On a headless CUDA container
# that raises `OSError: GL ES 2.0 library not found` and the knot preflight exits
# 90 -- so a knot-determinant run dies on a *graphics* library it never draws with.
# Measured 2026-08-05 on the second ladder attempt, and only visible because
# stream_progress teed the box's stdout home before the host was destroyed.
apt-get update -qq >> "$L" 2>&1 || log "apt update warn"
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    build-essential libgles2-mesa libegl1 libgl1 >> "$L" 2>&1 || log "apt install warn"
command -v gcc >/dev/null && log "gcc $(gcc -dumpversion)" || log "NO GCC -- pyknotid will fail"
ldconfig -p 2>/dev/null | grep -q libGLESv2 && log "libGLESv2 present" \
    || log "NO libGLESv2 -- pyknotid import will fail"

# 2. Miniconda + mamba (fast solver)
curl -s -L "$MC" -o /tmp/mc.sh
bash /tmp/mc.sh -b -p /workspace/conda >> "$L" 2>&1
source /workspace/conda/etc/profile.d/conda.sh
# Accept Anaconda channel ToS. Since 2024 conda HARD-FAILS any solve that even
# references the defaults channels (pkgs/main, pkgs/r) until their ToS is
# accepted (CondaToSNonInteractiveError). We only pull from conda-forge, but the
# Miniconda base env has defaults configured, so `conda install ... mamba` trips
# the gate and aborts -> jaxenv never builds. Accept once, non-interactively.
# (Measured 2026-06-14: this was the real onstart failure, NOT the IP throttle --
#  the same host clocked 109 MB/s to the conda CDN.)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >> "$L" 2>&1 || log "tos main warn"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >> "$L" 2>&1 || log "tos r warn"
conda install -n base -y -c conda-forge mamba >> "$L" 2>&1 || log "mamba install warn"

# 3. JAX + CUDA from conda-forge (Cloudflare), into a prefix env
SOLVER=mamba; command -v mamba >/dev/null || SOLVER=conda
# NB: `pip` MUST be in this list -- step 4 installs the engine via
# /workspace/jaxenv/bin/pip, which conda does NOT provide unless asked.
#
# ⚠ PINNED, and the pin is the FARM INTEGRATOR'S IDENTITY -- not parity with home.
# Unpinned (`"jaxlib=*=*cuda*" jax`), this line resolved whatever conda-forge shipped
# on the day of the rental. Two legs of the same campaign, weeks apart, at the SAME
# engine commit, could therefore run different solvers, and nothing in the manifest
# or the launch record would show it: the stage-2 rungs land within 4.3% of EHN, and
# a solver change is comfortably inside that. Under a pin, solver drift is a VERDICT
# FAILURE (the leg exits 91 before relaxing) rather than a silent change of physics.
#
# The skew is REAL and now explicit: home runs jax 0.11.0 from PyPI; conda-forge's
# newest is 0.10.2, so the farm has always been a version behind. Measured, not
# assumed -- jaxlib 0.10.2 has cuda129 builds on conda-forge for py311-py314.
# Moving this number is a deliberate act that re-dates every farm result after it.
JAX_PIN=0.10.2
$SOLVER create -y -p /workspace/jaxenv -c conda-forge \
    "jaxlib=${JAX_PIN}=*cuda*" "jax=${JAX_PIN}" scipy numpy pip >> "$L" 2>&1

# 4. The engine (small; GitHub pip is tolerable)
/workspace/jaxenv/bin/pip install -q \
    "git+https://github.com/JimGalasyn/jax-solitons" >> "$L" 2>&1

# 5. Verify JAX sees the GPU, then signal ready (else bail loudly)
if /workspace/jaxenv/bin/python -c \
    "import jax; d=jax.devices(); print(d); assert d and d[0].platform=='gpu'" >> "$L" 2>&1; then
    touch /tmp/worker-ready; log "worker ready (GPU ok)"
else
    touch /tmp/worker-bad-network; log "jax/GPU verify FAILED"; exit 1
fi
