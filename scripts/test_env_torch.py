#!/usr/bin/env python
"""Minimal sanity check for a Slurm GPU job using PyTorch.

Prints which interpreter is actually running, which environment it belongs to,
and whether PyTorch can see and use the GPU. Run this once before launching
a long job so a wrong interpreter or an invisible GPU fails in seconds instead
of after an hour.
"""

import os
import platform
import socket
import sys


def section(title):
    print(f"\n--- {title} ---", flush=True)


section("interpreter")
print("executable :", sys.executable)
print("version    :", platform.python_version())
print("prefix     :", sys.prefix)
print("conda env  :", os.environ.get("CONDA_DEFAULT_ENV", "<none>"))
print("virtual env:", os.environ.get("VIRTUAL_ENV", "<none>"))

section("node")
print("hostname   :", socket.gethostname())
print("cwd        :", os.getcwd())
print("job id     :", os.environ.get("SLURM_JOB_ID", "<not under slurm>"))
print("cpus       :", os.environ.get("SLURM_CPUS_PER_TASK", "<unset>"))
print("visible gpu:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))

section("packages")
for name in ("numpy", "torch", "torchvision", "torchaudio", "transformers", "sentence_transformers"):
    try:
        mod = __import__(name)
    except ImportError as exc:
        print(f"{name:<20} MISSING ({exc.__class__.__name__})")
    else:
        print(f"{name:<20} {getattr(mod, '__version__', 'unknown')}")

section("gpu")
try:
    import torch
except ImportError:
    print("torch not installed - nothing to check")
    sys.exit(1)

print("torch version  :", torch.__version__)
print("built with cuda:", torch.version.cuda)
print("cudnn version  :", torch.backends.cudnn.version())

gpu_available = torch.cuda.is_available()
print("cuda available :", gpu_available)
print("device count   :", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("  ", i, torch.cuda.get_device_name(i))

if not gpu_available:
    print("\nNo GPU visible to PyTorch.")
    print("Check: -G/--gres in the job script, CUDA_VISIBLE_DEVICES above,")
    print("and whether this build is torch (GPU/CUDA) or torch (CPU-only).")
    sys.exit(1)

# Real allocation + compute: a listed device is not always a usable one.
device = torch.device("cuda:0")
a = torch.randn(1000, 1000, device=device)
c = a @ a
ok = bool(torch.isfinite(c).all().item())

print("matmul ok      :", ok)
print("result device  :", c.device)

print("\nOK")
