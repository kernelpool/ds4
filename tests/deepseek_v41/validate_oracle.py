"""Checks the mini oracle is trustworthy: deterministic, exercising every mechanism, and
sensitive to the simulated KV quantisation.  Also runs a decode step so the ring buffer and the
partial-compressor-group path are covered, not just prefill."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_mini_model import install_reference, mini_args, init_weights, DEFAULT_SNAP

snap = os.environ.get("DS41_SNAPSHOT", DEFAULT_SNAP)

def build(quant, seed=0):
    ref = install_reference(snap, quantize=quant)
    import ds41_kernels_cpu as k; k.QUANTIZE = quant
    torch.manual_seed(seed); torch.set_default_dtype(torch.bfloat16); torch.set_default_device("cpu")
    return ref, init_weights(ref.Transformer(mini_args(ref)), seed).eval()

def run(model, ids, start=0):
    with torch.inference_mode():
        _, logits, _ = model(ids, start)
    return logits.detach().float().cpu().numpy()

ref, m = build(True)
ids = torch.randint(0, 512, (1, 24))
a, b = run(m, ids), run(m, ids)
print(f"determinism (same model, same input): max|d| = {np.abs(a-b).max():.3e}")

ref2, m2 = build(True)
c = run(m2, ids)
print(f"reproducible from scratch (fresh build, same seed): max|d| = {np.abs(a-c).max():.3e}")

# decode: continue one token past the prefill, exercising the SWA ring and partial compressor group
ref3, m3 = build(True)
_ = run(m3, ids, 0)
nxt = torch.randint(0, 512, (1, 1))
d1 = run(m3, nxt, 24); d2 = run(m3, torch.randint(0, 512, (1, 1)), 25)
print(f"decode step 24 -> logits {d1.shape} finite={np.isfinite(d1).all()}; step 25 finite={np.isfinite(d2).all()}")

# quantisation must actually change the numbers, or the shim is a no-op
ref4, m4 = build(False)
e = run(m4, ids)
print(f"quant on vs off: max|d| = {np.abs(a-e).max():.3e}  (0 would mean the quant shim is inert)")

z = np.load("tests/deepseek_v41/mini/oracle_prefill.npz")
comp = [k for k in z.files if "compressor" in k]
idxr = [k for k in z.files if "indexer" in k]
print(f"\nmechanisms exercised: compressor layers {comp}, indexer layers {idxr}")
for k in comp + idxr:
    v = z[k]; print(f"   {k:22s} {str(v.shape):18s} finite={np.isfinite(v).all()} range=[{v.min():.3f},{v.max():.3f}]")
print(f"   hc stream shape (layer out): {z['layer0.out'].shape}  -> [b, s, hc_mult, dim]")
