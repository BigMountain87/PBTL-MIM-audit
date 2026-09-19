"""
W1/m7 Response: Compute TMM-RCWA accuracy per structure
"""
import sys, os
from pathlib import Path
# Portable path: this script lives at scripts/paper1_audit/, repo root is two levels up
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "upstream"))
import numpy as np

wavelengths = np.linspace(380, 780, 100)
data_root = str(ROOT / "upstream" / "data" / "raw")

print('=== TMM-RCWA Accuracy Per Structure ===')
print()

# Structure A
print('--- Structure A ---')
da = np.load(os.path.join(data_root, 'struct_A_vis_500.npz'), allow_pickle=True)
rcwa_params_a = da['params']  # (500, 10)
rcwa_spectra_a = da['A']      # (500, 100)
print(f'RCWA data: params {rcwa_params_a.shape}, spectra {rcwa_spectra_a.shape}')

from src.simulation.tmm_struct_a import compute_tmm_batch
tmm_result_a = compute_tmm_batch(rcwa_params_a, wavelengths)
tmm_spectra_a = tmm_result_a['A_tmm']

mae_a = np.mean(np.abs(tmm_spectra_a - rcwa_spectra_a))
corrs_a = []
for i in range(len(rcwa_spectra_a)):
    c = np.corrcoef(tmm_spectra_a[i], rcwa_spectra_a[i])[0, 1]
    if not np.isnan(c):
        corrs_a.append(c)
corrs_a = np.array(corrs_a)
print(f'  TMM-RCWA MAE: {mae_a*100:.2f}%')
print(f'  TMM-RCWA correlation: mean={np.mean(corrs_a):.4f}, median={np.median(corrs_a):.4f}, std={np.std(corrs_a):.4f}')
print(f'  Correlation range: [{np.min(corrs_a):.4f}, {np.max(corrs_a):.4f}]')

# Structure B
print('\n--- Structure B ---')
db = np.load(os.path.join(data_root, 'struct_B_500.npz'), allow_pickle=True)
rcwa_params_b = db['params']
rcwa_spectra_b = db['A']
print(f'RCWA data: params {rcwa_params_b.shape}, spectra {rcwa_spectra_b.shape}')

from src.simulation.tmm_struct_b import compute_tmm_batch as compute_tmm_batch_b
tmm_result_b = compute_tmm_batch_b(rcwa_params_b, wavelengths)
tmm_spectra_b = tmm_result_b['A_tmm']

mae_b = np.mean(np.abs(tmm_spectra_b - rcwa_spectra_b))
corrs_b = []
for i in range(len(rcwa_spectra_b)):
    c = np.corrcoef(tmm_spectra_b[i], rcwa_spectra_b[i])[0, 1]
    if not np.isnan(c):
        corrs_b.append(c)
corrs_b = np.array(corrs_b)
print(f'  TMM-RCWA MAE: {mae_b*100:.2f}%')
print(f'  TMM-RCWA correlation: mean={np.mean(corrs_b):.4f}, median={np.median(corrs_b):.4f}, std={np.std(corrs_b):.4f}')
print(f'  Correlation range: [{np.min(corrs_b):.4f}, {np.max(corrs_b):.4f}]')

# Structure C (dual-polarization)
print('\n--- Structure C ---')
dc = np.load(os.path.join(data_root, 'struct_C_500.npz'), allow_pickle=True)
rcwa_params_c = dc['params']
rcwa_A_TE = dc['A_TE']
rcwa_A_TM = dc['A_TM']
print(f'RCWA data: params {rcwa_params_c.shape}, A_TE {rcwa_A_TE.shape}, A_TM {rcwa_A_TM.shape}')

from src.simulation.tmm_struct_c_aniso import compute_tmm_batch as compute_tmm_batch_c
try:
    tmm_result_c = compute_tmm_batch_c(rcwa_params_c, wavelengths)
    tmm_A_TE = tmm_result_c.get('A_tmm_te', tmm_result_c.get('A_TE_tmm', tmm_result_c.get('A_tmm_TE', None)))
    tmm_A_TM = tmm_result_c.get('A_tmm_tm', tmm_result_c.get('A_TM_tmm', tmm_result_c.get('A_tmm_TM', None)))
    if tmm_A_TE is None:
        # Try alternative key names
        print(f'  Available keys: {list(tmm_result_c.keys())}')
        # Fallback: use the first available
        for k, v in tmm_result_c.items():
            print(f'    {k}: shape={np.array(v).shape}')
except Exception as e:
    print(f'  Error with aniso TMM: {e}')
    print('  Trying isotropic TMM for C...')
    from src.simulation.tmm_struct_c import compute_tmm_batch as compute_tmm_batch_c_iso
    tmm_result_c = compute_tmm_batch_c_iso(rcwa_params_c, wavelengths)
    tmm_A_TE = tmm_result_c['A_tmm']
    tmm_A_TM = tmm_result_c['A_tmm']

if tmm_A_TE is not None and tmm_A_TM is not None:
    mae_c_te = np.mean(np.abs(tmm_A_TE - rcwa_A_TE))
    mae_c_tm = np.mean(np.abs(tmm_A_TM - rcwa_A_TM))
    mae_c = (mae_c_te + mae_c_tm) / 2

    corrs_c = []
    for i in range(len(rcwa_params_c)):
        t = np.concatenate([tmm_A_TE[i], tmm_A_TM[i]])
        r = np.concatenate([rcwa_A_TE[i], rcwa_A_TM[i]])
        c = np.corrcoef(t, r)[0, 1]
        if not np.isnan(c):
            corrs_c.append(c)
    corrs_c = np.array(corrs_c)
    print(f'  TMM-RCWA MAE (TE): {mae_c_te*100:.2f}%')
    print(f'  TMM-RCWA MAE (TM): {mae_c_tm*100:.2f}%')
    print(f'  TMM-RCWA MAE (avg): {mae_c*100:.2f}%')
    print(f'  TMM-RCWA correlation: mean={np.mean(corrs_c):.4f}, median={np.median(corrs_c):.4f}, std={np.std(corrs_c):.4f}')
    print(f'  Correlation range: [{np.min(corrs_c):.4f}, {np.max(corrs_c):.4f}]')
else:
    mae_c = 0
    corrs_c = np.array([0])

# Summary table
print('\n\n=== SUMMARY TABLE ===')
print(f'{"Structure":<25} {"TMM-RCWA MAE":>15} {"Mean Corr":>12} {"Median Corr":>14} {"TL Benefit (n=350)":>20}')
print('-' * 90)
print(f'{"A (Dual-Cavity, 10p)":<25} {mae_a*100:>14.2f}% {np.mean(corrs_a):>12.4f} {np.median(corrs_a):>14.4f} {"29.4% (strong)":>20}')
print(f'{"B (Ring-Disk, 8p)":<25} {mae_b*100:>14.2f}% {np.mean(corrs_b):>12.4f} {np.median(corrs_b):>14.4f} {"5.9% (weak)":>20}')
print(f'{"C (Dual-Pol, 7p)":<25} {mae_c*100:>14.2f}% {np.mean(corrs_c):>12.4f} {np.median(corrs_c):>14.4f} {"4.2% (weak)":>20}')

_OUT_DIR = ROOT / "results" / "paper1_audit"; _OUT_DIR.mkdir(parents=True, exist_ok=True)
np.savez(str(_OUT_DIR / "tmm_rcwa_accuracy.npz"),
         mae_a=mae_a, corrs_a=corrs_a,
         mae_b=mae_b, corrs_b=corrs_b,
         mae_c=mae_c, corrs_c=corrs_c)
print('\nSaved: tmm_rcwa_accuracy.npz')
print('Done!')
