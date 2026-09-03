"""Expand a trained v2 surrogate checkpoint from 24 to 32 AGL output levels.

The ViT backbone is level-independent; only three tensors per output head depend
on nz:

    heads.<h>.vert_basis          (1, feat, 1, 1, nz)
    heads.<h>.film_net.2.weight   (feat * nz * 2, 128)
    heads.<h>.film_net.2.bias     (feat * nz * 2,)

`GeoFiLMVerticalHead.forward` reshapes the FiLM output as ``view(B, C, nz, 2)``,
so the flat layout is row-major (feat, nz, 2) and padding along the nz axis is a
plain slice-copy.

New levels are initialised from the topmost trained level (100 m) rather than at
random: the surrogate then starts out predicting a constant profile above 100 m,
which is far closer to the truth than noise and lets the fine-tune spend its
budget on the 110-200 m band alone.

Usage:
    python expand_surrogate_heads_v3.py --src <v2 best.pt> --dst <v3 init.pt>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _expand_nz(t: torch.Tensor, *, axis: int, n_old: int, n_new: int) -> torch.Tensor:
    """Grow `axis` from n_old to n_new, repeating the last slice."""
    idx = list(range(n_old)) + [n_old - 1] * (n_new - n_old)
    return t.index_select(axis, torch.tensor(idx, device=t.device))


def expand_state_dict(sd: dict, *, nz_old: int, nz_new: int) -> tuple[dict, list[str]]:
    out, touched = {}, []
    for k, v in sd.items():
        if k.endswith("vert_basis") and v.ndim == 5 and v.shape[-1] == nz_old:
            out[k] = _expand_nz(v, axis=-1, n_old=nz_old, n_new=nz_new)
            touched.append(k)
        elif "film_net" in k and k.endswith(("weight", "bias")) and v.shape[0] % (nz_old * 2) == 0 \
                and v.shape[0] // (nz_old * 2) > 1:
            feat = v.shape[0] // (nz_old * 2)
            tail = v.shape[1:]
            w = v.reshape(feat, nz_old, 2, *tail)
            w = _expand_nz(w, axis=1, n_old=nz_old, n_new=nz_new)
            out[k] = w.reshape(feat * nz_new * 2, *tail)
            touched.append(k)
        else:
            out[k] = v
    return out, touched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--nz-old", type=int, default=24)
    ap.add_argument("--nz-new", type=int, default=32)
    args = ap.parse_args()

    ck = torch.load(str(args.src), map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    new_sd, touched = expand_state_dict(sd, nz_old=args.nz_old, nz_new=args.nz_new)

    if not touched:
        raise SystemExit(f"no nz={args.nz_old} tensor found in {args.src} — wrong checkpoint?")

    if "model" in ck:
        ck["model"] = new_sd
    else:
        ck = new_sd
    ck.pop("optimizer", None)   # optimiser moments are stale after the reshape
    ck.pop("scheduler", None)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, str(args.dst))

    print(f"expanded {len(touched)} tensors {args.nz_old} -> {args.nz_new} levels")
    for k in touched:
        print(f"  {k}: {tuple(sd[k].shape)} -> {tuple(new_sd[k].shape)}")
    print(f"carried over unchanged: {len(new_sd) - len(touched)} tensors")
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
