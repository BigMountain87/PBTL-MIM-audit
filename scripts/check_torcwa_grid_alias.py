"""How much does the 64x64 real-space raster change torcwa's permittivity convolution
matrix at the orders the pub pipeline uses?

Builds one 'hard' rectangle (f_min < 150 nm, aspect ratio > 2, so adaptive_order picks
N = 17) on a 64x64 and a 256x256 grid, forms the (2N+1)^2-square convolution matrix
with torcwa's own _material_conv, and compares the two entry-by-entry, grouped by the
harmonic difference max(|mx-nx|, |my-ny|).  Two effects are separated:

  * wrap-around aliasing: an nx-point FFT holds harmonics -nx/2 .. nx/2-1, the matrix
    needs differences up to 2N, so nx >= 4N+2 is required (70 at N = 17; 64 gives a
    clean maximum of N = 15) -- rows with max|m-n| >= 32;
  * raster resolution: pixel = P/nx (9.4 nm at P = 600) with a soft sigmoid edge
    (torcwa.rcwa_geo default edge_sharpness = 100, ~1-pixel transition), so the raster
    under-resolves the geometry -- visible in the low-difference rows, at any N.

NB: the raw 64-vs-256 comparison includes a physically inert sub-pixel translation phase
(torcwa samples at pixel centres), which inflates the Frobenius difference ~2x (13.6 % raw
vs 7.6 % de-phased at N = 17). See docs/notes_rcwa_grid_truncation_v11.md §4 and §6.

Usage: python scripts/check_torcwa_grid_alias.py [N]       (default 17; try 15)
CPU only, seconds.  See docs/notes_rcwa_grid_truncation_v11.md.
"""
import sys
import torch
import torcwa

torch.set_default_dtype(torch.float64)
dev = torch.device("cpu")
dt = torch.complex128

P, Wx, Wy = 600.0, 120.0, 400.0          # f_min = 120 < 150, ar = 3.3  ->  N = 17 in adaptive_order
eps_metal = complex(-3.0, 12.0)          # Cr-like near 1 um; only the contrast matters here
eps_diel = 2.1
N = int(sys.argv[1]) if len(sys.argv) > 1 else 17


def convmat(nx):
    torcwa.rcwa_geo.dtype = torch.float64
    torcwa.rcwa_geo.device = dev
    torcwa.rcwa_geo.Lx = P
    torcwa.rcwa_geo.Ly = P
    torcwa.rcwa_geo.nx = nx
    torcwa.rcwa_geo.ny = nx
    torcwa.rcwa_geo.grid()
    g = torcwa.rcwa_geo.rectangle(Wx=Wx, Wy=Wy, Cx=P / 2, Cy=P / 2)
    eps = eps_diel + (eps_metal - eps_diel) * g.to(dt)
    sim = torcwa.rcwa(freq=1 / 1000.0, order=[N, N], L=[P, P], dtype=dt, device=dev)
    return sim._material_conv(eps), sim


C64, sim = convmat(64)
C256, _ = convmat(256)

ox = torch.meshgrid(sim.order_x, sim.order_y, indexing="ij")[0].reshape(-1)
oy = torch.meshgrid(sim.order_x, sim.order_y, indexing="ij")[1].reshape(-1)
dmax = torch.maximum((ox[:, None] - ox[None, :]).abs(), (oy[:, None] - oy[None, :]).abs())

diff = (C64 - C256).abs()
ref = C256.abs()
print(f"N={N}: convolution matrix {C64.shape[0]}x{C64.shape[0]}, grid 64 vs 256, "
      f"P={P:.0f} Wx={Wx:.0f} Wy={Wy:.0f} nm (pixel {P/64:.1f} vs {P/256:.1f} nm)")
print(f"{'max|m-n|':>9} {'entries':>8} {'median|C256|':>13} {'median|C64-C256|':>17} {'rel':>7}")
for d in sorted(set(dmax.flatten().tolist())):
    m = dmax == d
    r = ref[m].median().item()
    e = diff[m].median().item()
    print(f"{int(d):>9} {int(m.sum()):>8} {r:>13.4e} {e:>17.4e} {e / max(r, 1e-30):>7.2f}")

band = dmax >= 32
print(f"\nentries with max|m-n|>=32 (aliased on a 64 grid): {int(band.sum())} of {band.numel()} "
      f"({100 * band.float().mean():.2f} %)")
print(f"||C64-C256||_F / ||C256||_F, all entries      : {(diff.norm() / ref.norm()).item():.3e}")
print(f"||C64-C256||_F / ||C256||_F, max|m-n|<=31 only: "
      f"{(diff[~band].norm() / ref[~band].norm()).item():.3e}")
