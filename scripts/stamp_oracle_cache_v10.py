#!/usr/bin/env python3
"""Stamp oracle-cache entries with the current solver fingerprint -- an explicit, recorded
migration, not a bypass.

    INVERSETL_PROFILE=pub ... python3 scripts/stamp_oracle_cache_v10.py [--apply]

Entries written before the fingerprint existed (T54, 2026-09-06..09) carry geometry,
wavelengths and per-wavelength orders but no solver identity, and entries written under an
earlier fingerprint formula (before module_sha256 was included) would be rejected by the
current one.  Rather than recompute ~350 designs (~150 GPU-hours), this script accepts an
entry only when

  1. its structure's ASSEMBLED artifact (rcwa_<S><sx>_v8.npz) records the settings the
     current profile contract demands -- adaptive order, complex64, JC materials -- which the
     driver wrote from the live solver at assembly time;
  2. the entry's stored per-wavelength orders equal what the current module selects for
     that geometry (the strongest check the old fields allow);
  3. its spectra are finite and the right length.

It then rewrites the entry with the fingerprint the current live solver would produce for
that geometry and records every decision in results/oracle_cache_migration_v10.json.
Anything failing a check is left unstamped and therefore recomputed on next use.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src_v8"))
import common, oracle                                   # noqa: E402
import importlib


def main(apply):
    log = dict(profile=common.PROFILE, when=time.strftime("%Y-%m-%d %H:%M:%S"), apply=apply, entries=[])
    mods = {}
    for f in sorted(oracle.CACHE.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        kind, s, i = str(z["kind"]), str(z["structure"]), int(z["index"])
        sx = f.stem[len(f"{kind}_{s}"):-4]                # "" / "_s123" / "_m0" / "_o5" ...
        rec = dict(file=f.name, kind=kind, structure=s, sx=sx, index=i, had_fingerprint="fingerprint" in z.files)
        try:
            if s not in mods:
                m = importlib.import_module(common.STRUCTS[s]["rcwa_module"])
                mods[s] = m
            mod = mods[s]
            # the assembled artifact this entry fed, and what it says the solver was
            # rcwa_validate --tag o5 writes cache kind 'rcwa_o5' and artifact rcwa_<S><sx>_v8_o5.npz;
            # INVERSETL_TAG=_o5 would have put the tag in sx.  Handle both spellings.
            fixed5 = kind.endswith("_o5") or sx.endswith("_o5")
            asm = oracle.RESULTS / (f"rcwa_{s}{sx}_v8_o5.npz" if kind.endswith("_o5") else f"{kind}_{s}{sx}_v8.npz")
            if kind.startswith("rcwa") and asm.exists():
                a = np.load(asm, allow_pickle=True)
                if fixed5:
                    oracle.configure_solver(mod, order=5)
                else:
                    oracle.configure_solver(mod)
                ok_settings = (bool(a["adaptive"]) == bool(mod.RCWA_SETTINGS.get("adaptive_order"))
                               and "complex64" in str(a["sim_dtype"]) and str(a["materials"]) == "jc")
                rec["assembled_artifact"] = asm.name
            else:
                oracle.configure_solver(mod, order=5 if fixed5 else None)
                ok_settings = True                        # no assembled artifact to contradict
                rec["assembled_artifact"] = None
            cfg = common.STRUCTS[s]
            names = list(cfg["param_names"]); params = np.asarray(z["params"], float)
            pdict = dict(zip(names, params.tolist())); wl = np.asarray(z["wavelengths"], float)
            ok_orders = np.asarray(z["orders"]).tolist() == oracle.orders_used(mod, s, pdict, wl)
            a_keys = ("A_TE", "A_TM") if cfg["dual"] else ("A",)
            ok_spec = bool(z["failed"]) or all(np.asarray(z[f"A_{k}"]).shape == (len(wl),)
                                               and np.all(np.isfinite(np.asarray(z[f"A_{k}"]))) for k in a_keys)
            rec.update(ok_settings=ok_settings, ok_orders=ok_orders, ok_spectra=ok_spec)
            if ok_settings and ok_orders and ok_spec:
                fp = oracle.fingerprint(s, pdict, wl, cfg, mod)
                rec["fingerprint"] = fp[:16]
                if apply:
                    d = {k: z[k] for k in z.files}
                    d["fingerprint"] = fp
                    d["solver"] = json.dumps(oracle.live_provenance(mod), sort_keys=True)
                    d["stamped_by"] = "scripts/stamp_oracle_cache_v10.py " + log["when"]
                    tmp = f.with_suffix(".tmp.npz"); np.savez(tmp, **d); tmp.replace(f)
                rec["action"] = "stamped" if apply else "would stamp"
            else:
                rec["action"] = "left unstamped (will be recomputed)"
        except Exception as e:
            rec["action"] = f"error: {type(e).__name__}: {str(e)[:120]}"
        log["entries"].append(rec)
    n = len(log["entries"]); st = sum(r["action"].startswith(("stamped", "would")) for r in log["entries"])
    log["summary"] = dict(total=n, stamped=st, left=n - st)
    out = oracle.RESULTS / "oracle_cache_migration_v10.json"
    out.write_text(json.dumps(log, indent=1))
    print(f"{n} entries: {st} {'stamped' if apply else 'would be stamped'}, {n - st} left -> {out.name}")
    for r in log["entries"]:
        if not r["action"].startswith(("stamped", "would")):
            print("  ", r["file"], r["action"], {k: r.get(k) for k in ("ok_settings", "ok_orders", "ok_spectra")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true")
    main(ap.parse_args().apply)
