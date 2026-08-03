"""Break down TQD's per-step cost into resharding, local gate-application
(interchange + bmm), and measurement-sampling time, using torch.profiler with
named record_function regions around the three suspected cost sources:

    tqd.device.DistributedQuantumDevice.maybe_reshard
    tqd.functional.apply_unitary_bmm
    tqd.measure.sampler_diff_approx

Nothing in the tqd package itself is modified -- the target functions are
monkeypatched at import time to wrap the original call in a named
record_function block, so the profiler trace attributes time to each of them
distinctly instead of a flat list of aten:: ops and NCCL kernels.

Usage (matches scripts/scaling.py's launch convention):
    torchrun --nproc_per_node=2 scripts/profile_cost_breakdown.py <batch> <nq> <world_sz>
"""

import argparse
import functools
import os

import torch
from torch.profiler import ProfilerActivity, profile

import tqd
import tqd.device as tqd_device
import tqd.functional as tqd_functional
import tqd.measure as tqd_measure
import tqd.module


def _record(fn, name):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function(name):
            return fn(*args, **kwargs)

    return wrapped


tqd_device.DistributedQuantumDevice.maybe_reshard = _record(  # type: ignore[method-assign]
    tqd_device.DistributedQuantumDevice.maybe_reshard, "maybe_reshard"
)
tqd_functional.apply_unitary_bmm = _record(
    tqd_functional.apply_unitary_bmm, "apply_unitary_bmm"
)
tqd_measure.sampler_diff_approx = _record(
    tqd_measure.sampler_diff_approx, "sampler_diff_approx"
)


def build_circuit(nq):
    """Same shape of circuit as scripts/scaling.py: encoder + layered CX-ladder/RY."""
    func_list = [{"func": "ry", "wires": [i], "input_idx": [i]} for i in range(nq)]
    enc = tqd.GeneralEncoder(func_list)
    base_mod = [enc]
    for _ in range(3):
        base_mod = (
            base_mod
            + [tqd.CX(wires=[i, (i + 1) % nq]) for i in range(nq)]
            + [tqd.RY(wires=[i]) for i in range(nq)]
        )
    return tqd.module.InvertibleUnitary(base_mod)


def profile_run(batch, nq, world_sz, steps=10, shots=1000):
    rank = os.environ["RANK"]
    qdev = tqd.DistributedQuantumDevice(
        nq, bsz=batch, device="cuda", world_sz=world_sz, invertible=True
    )
    mod = build_circuit(nq)
    mod.train()
    opt = torch.optim.Adam(mod.parameters())

    x_i = torch.rand([batch, nq], device=f"cuda:{rank}") * torch.pi / 3

    def step():
        opt.zero_grad()
        qdev.reset_states()
        mod(qdev, x_i)
        # differentiable approximate sampler path (training=True), the one actually
        # used in practice for backprop through measurement during VMC training
        with torch.profiler.record_function("measure_allZ_diff"):
            meas = tqd.measure_allZ(qdev, shots=shots, training=True)
        with torch.profiler.record_function("backward"):
            loss = meas.abs().sum()
            loss.backward()
        opt.step()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(steps):
            step()
            torch.cuda.synchronize()

    if rank == "0":
        print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=25))

        totals = {}
        for evt in prof.key_averages():
            if evt.key in (
                "maybe_reshard",
                "apply_unitary_bmm",
                "sampler_diff_approx",
                "measure_allZ_diff",
                "backward",
            ):
                totals[evt.key] = evt.self_device_time_total
        print(
            "\n--- Named-region self-device-time summary (us, summed over",
            steps,
            "steps) ---",
        )
        for name, us in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f"{name:25s} {us:12.1f}")


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("batch", type=int)
    arg_parser.add_argument("nq", type=int)
    arg_parser.add_argument("world_sz", type=int)
    arg_parser.add_argument("--steps", type=int, default=10)
    arg_parser.add_argument("--shots", type=int, default=1000)
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profile_run(args.batch, args.nq, args.world_sz, steps=args.steps, shots=args.shots)
