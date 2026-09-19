#!/usr/bin/env python3
"""T26 — assert every number in the manuscript's Table 3 against the evidence JSONs.
Usage: python3 scripts/check_table3_v10.py [paper/manuscript_v10.md] [results_v8]"""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
M = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper/manuscript_v10.md"
R = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "results_v8")
syn = json.load(open(R / "synthesis_v8.json")); pooled = json.load(open(R / "pooled_v8.json"))
st = json.load(open(R / "stats_supplement_v9.json"))
txt = M.read_text()
row = {ln.split("|")[1].strip(): [c.strip() for c in ln.strip().strip("|").split("|")]
       for ln in txt.splitlines() if ln.startswith("| H1") or ln.startswith("| — |")}
ps, fails = syn["per_structure"], []


def chk(name, got, want):
    if got != want:
        fails.append(f"{name}: manuscript {got!r} != evidence {want!r}")


# seed-42 counts in the pre-registered order A, C, B
chk("H1a seed-42", row["H1a"][1].split("(")[0].strip(),
    "oracle success " + " / ".join(str(ps[s]["oracle_success"]) for s in ("A", "C", "B")))
chk("H1b seed-42", row["H1b"][1].split("(")[0].strip(),
    "Δ-flag rate " + " / ".join(str(ps[s]["flagged"]) for s in ("A", "C", "B")))
chk("H1c seed-42", row["H1c"][1].strip(),
    "ρ(RCWA, Surr) " + " / ".join(f"{ps[s]['rho_rcwa_surr']:+.2f}".replace("-", "−") for s in ("A", "C", "B")))
chk("max Δ", row["—"][1].strip(),
    "worst-case severity max Δ " + " / ".join(f"{ps[s]['max_delta']:.1f}" for s in ("A", "C", "B")) + " %")
# pooled counts
pool = pooled["per_structure"]
chk("H1a pooled", re.search(r"\(([^)]*)\)", row["H1a"][1]).group(1),
    ", ".join(f"{pool[s]['oracle_pass']}/{pool[s]['n_valid']}" for s in ("A", "C", "B")))
# monotonicity flags
mono_pre = {k[:3]: v["monotone"] for k, v in syn["hypotheses"].items()}
mono_pub = {k[:3]: v["monotone"] for k, v in json.load(open(R / "synthesis_v8_printed.json"))["hypotheses"].items()}
for h, key in (("H1a", "H1a"), ("H1b", "H1b"), ("H1c", "H1c")):
    chk(f"{h} monotone preliminary", row[h][2], "yes" if mono_pre[key] else "no")
    chk(f"{h} monotone published", row[h][3], "yes" if mono_pub[key] else "no")
# smallest pairwise p (nominal; Holm)
holm = pooled["multiplicity"]["fisher_p_holm"]
for h, metric in (("H1a", "oracle_success"), ("H1b", "flagged")):
    nom = min(syn["fisher_exact_p"][metric].values())
    hkey = min(((k, v) for k, v in syn["fisher_exact_p"][metric].items()), key=lambda kv: kv[1])[0]
    chk(f"{h} p", row[h][4], f"{nom:.3f} ({holm[f'{metric}:{hkey}']:.3f})")
rho_p = min(ps[s]["rho_p"] for s in "ABC")
chk("H1c p", row["H1c"][4].split(" ")[0], f"{rho_p:.3f}")
# trend and power sentences
for want in (f"nominal *p* = {st['ca_trend']['conventions']['preliminary']['pretender']['nominal']['p']:.2f}".replace("0.62", "0.62"),
             f"{st['ca_trend']['conventions']['printed']['pretender']['nominal']['p']:.3f}",
             f"{st['ca_trend']['conventions']['printed']['pretender']['deff_corrected']['p']:.3f}",
             f"{st['power']['fisher_20v20_mde_at_0p80']*100:.0f} pp",
             f"{st['power']['spearman_n20_mde_at_0p80']:.2f}"):
    if want not in txt:
        fails.append(f"missing from manuscript: {want!r}")
# common-threshold counts
ct = st["common_threshold_h1b"]["thr_from_A"]
if f"{ct['flagged']['B']} / {ct['flagged']['A']} / {ct['flagged']['C']}" not in txt:
    fails.append("common-threshold counts at 9.69 % not found in the stated B/A/C order")
print("\n".join(fails) if fails else "Table 3 check: all values match the evidence JSONs")
sys.exit(1 if fails else 0)
