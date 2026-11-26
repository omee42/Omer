#!/usr/bin/env python3
"""bolge_ag — modelin yalnız bir bölgesini hızlıca ağa çevirir (tam üretim 1-2 saat; bölge 5-20 dk).

    python3 bolge_ag.py <arac_betigi.py> <cikis_onek> <h_mm> x0 x1 y0 y1 z0 z1 [mod]

Betik `arac_sdf(Blok, mod)` (y >= 0 yarısında SDF) ve isteğe bağlı `asimetrik(tam, x, y, z, mod)` tanımlamalı.
y0, y1 >= 0 bir |y| bandıdır; iki yan ayrı yazılır: <onek>_L.npz (-y, sürücü tarafı) ve <onek>_R.npz (+y).
Çıktı gerçek mm'dir; gorsel.py zebra / kamera.py bindir doğrudan okur. h ~3 mm genelde yeter.
Çekirdek sayısı NP ortam değişkeniyle (varsayılan 4)."""
import importlib.util
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from skimage.measure import marching_cubes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arac_cekirdek as C  # noqa: E402

modp, pre, h = sys.argv[1], sys.argv[2], float(sys.argv[3])
x0, x1, y0, y1, z0, z1 = map(float, sys.argv[4:10])
MOD = sys.argv[10] if len(sys.argv) > 10 else "tek"
M = None


def load():
    spec = importlib.util.spec_from_file_location("arac_modeli", modp)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def work(a):
    global M
    if M is None:
        M = load()
    xs, yh, z = a
    return M.arac_sdf(C.Blok(xs, yh, z), MOD).astype(np.float32)


if __name__ == "__main__":
    t0 = time.time()
    X0 = float(os.environ.get("ARAC_X0", "-40")); Z0 = float(os.environ.get("ARAC_Z0", "-20"))
    # tam üretim ızgarasıyla hizalı (x X0'dan, y 0'dan, z Z0'dan)
    x = X0 + h * np.arange(np.ceil((x0 - X0) / h), np.floor((x1 - X0) / h) + 1)
    yh = h * np.arange(np.ceil(y0 / h), np.floor(y1 / h) + 1)
    z = Z0 + h * np.arange(np.ceil((z0 - Z0) / h), np.floor((z1 - Z0) / h) + 1)
    jobs = [(x[i:i + 6], yh, z) for i in range(0, len(x), 6)]
    with Pool(int(os.environ.get("NP", "4"))) as P:
        yar = np.concatenate(P.map(work, jobs), 0)
    Mm = load()
    for side in ("L", "R"):
        if side == "L":
            vol = np.ascontiguousarray(yar[:, ::-1, :]); y = -yh[::-1]
        else:
            vol = yar.copy(); y = yh
        if hasattr(Mm, "asimetrik"):
            Mm.asimetrik(vol, x, y, z, MOD)
        vol[0] = vol[-1] = 1e3; vol[:, 0] = vol[:, -1] = 1e3; vol[:, :, 0] = vol[:, :, -1] = 1e3
        if vol.min() >= 0:
            continue
        v, f, _, _ = marching_cubes(vol, level=0.0137, allow_degenerate=False)
        v = v * h + np.array([x[0], y[0], z[0]])
        if C._ters(v, f):
            f = f[:, ::-1]
        np.savez(f"{pre}_{side}.npz", v=v.astype(np.float32), f=f.astype(np.int32))
        print(pre, side, len(f), "üçgen", flush=True)
        del vol
    print("bitti", round(time.time() - t0), "sn")
