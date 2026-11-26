#!/usr/bin/env python3
"""
Opel Astra J Sedan (2012–2018, 4 kapı, makyajlı ön) — 3D baskı maketi üretici.

Yöntem ve araçlar: .claude/skills/arac-3d (SKILL.md). Ortak çekirdek: scripts/arac_cekirdek.py.
Tüm geometri gerçek aracın milimetre ölçüleriyle tanımlanır, baskı ölçeğine en sonda küçültülür.
    boy 4658, genişlik 1814 (aynasız), yükseklik 1500, aks mesafesi 2685, iz ön 1544 / arka 1558,
    lastik 225/50 R17 (Cosmo / Sport donanım).
Koordinatlar: x ön tampon ucundan arkaya, +y sağ, z yukarı (yer z=0).

Çıktılar:
    astra_sedan_tek_parca.stl   tekerlekler gövdeye bağlı, tek parça
    astra_sedan_govde.stl       tekerleksiz gövde (aks delikli)      ┐ kit
    astra_sedan_teker.stl       tekerlek (4 adet basılır)            ┘

Kullanım:
    python3 astra_sedan_olustur.py --parca kit --h 0.07
    python3 astra_sedan_olustur.py --parca tek --h 0.08
"""
import argparse
import gc
import math
import os
import sys

import numpy as np

_BURASI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("ARAC3D_SCRIPTS", os.path.join(_BURASI, "..", ".claude", "skills", "arac-3d", "scripts")))
from arac_cekirdek import (Egri, YPoligon, Blok, p, smin, smax, sstep, sstep5, sd_cizgi, sd_kutu2,  # noqa: E402
                           sd_elipsoid, oyuk, doldur, petek, yumusat, duz_yazi_2d, govde_mesh, sadelestir,
                           kaydet, teker_mesh)

# ============================================================ ana ölçüler (gerçek mm)
L = 4658.0
GEN_YARI = 907.0          # 1814 / 2
YUKSEKLIK = 1500.0
AKS_MES = 2685.0
AKS_ON = 925.0
AKS_ARKA = AKS_ON + AKS_MES
TEKER_R = 328.0           # 225/50 R17 -> çap 656.8
JANT_R = 216.0            # 17 inç oturma yarıçapı
JANT_DIS = 232.0          # görünen jant dudağı
TEKER_YG = 112.0          # lastik yarı genişliği
TEKER_Y_ON, TEKER_Y_ARKA = 772.0, 779.0   # iz / 2
AKS_Z = TEKER_R
CAMUR_R = 362.0
KUYU_Y = 640.0
AKS_DELIK_R = 1.05        # kit, baskı mm: 1.75 mm filament gövdede serbest döner
TEKER_DELIK_R = 0.90      # kit, baskı mm: tekerlekte sıkı geçme
KUTU = (-40.0, L + 60.0, 1000.0, -20.0, YUKSEKLIK + 40.0)   # üretim hacmi (x0, x1, y_yari, z0, z1)


# ============================================================ profiller (gerçek mm)
# İLK TAHMİN: fotoğraf ölçümleriyle değiştirilecek (her tablonun üstüne kaynağı yazılır).
# yan profil, orta hat: alt gövde üst çizgisi (tampon üstü -> kaput -> torpido; arkada bagaj -> arka yüz)
Z_UST = Egri([(0, 560), (40, 650), (110, 730), (220, 780), (500, 830), (900, 890), (1250, 955), (1300, 965),
              (3700, 1050), (3900, 1062), (4200, 1072), (4450, 1075), (4580, 1068), (4630, 1040), (4650, 980),
              (4658, 900)])
KAPUT_BOMBE = Egri([(0, 0), (300, 15), (800, 40), (1250, 55), (1350, 20), (3800, 20), (4000, 45), (4500, 60), (4658, 60)])
Z_ALT = Egri([(0, 380), (60, 250), (200, 190), (400, 160), (600, 150), (1200, 150), (3400, 150), (4100, 160),
              (4400, 220), (4560, 320), (4658, 400)])
HW_PLAN = Egri([(0, 880), (925, 904), (1600, 905), (2400, 905), (3610, 907), (4300, 895), (4658, 870)])
F_DIKEY = Egri([(100, 0.90), (150, 0.93), (250, 0.965), (400, 0.99), (550, 1.0), (750, 0.995), (880, 0.975),
                (980, 0.94), (1080, 0.88)])
ON_PLAN = Egri([(0, 0), (200, 15), (400, 60), (550, 115), (680, 190), (780, 275), (850, 365), (890, 450), (907, 540)])
ON_YAN = Egri([(150, 70), (250, 25), (350, 5), (450, 0), (550, 0), (650, 15), (750, 50), (830, 100)])
ARKA_PLAN = Egri([(0, 0), (250, 8), (450, 30), (600, 70), (720, 130), (810, 210), (870, 310), (900, 420), (907, 520)])
ARKA_YAN = Egri([(150, 80), (250, 25), (350, 5), (450, 0), (700, 0), (850, 6), (950, 20), (1020, 45), (1075, 90)])
# kabin (cam evi): orta hat tavan / ön cam / arka cam çizgisi ve bel hizasında yarı genişlik
Z_KABIN = Egri([(1150, 900), (1300, 975), (1500, 1100), (1750, 1255), (2000, 1385), (2200, 1450), (2450, 1490),
                (2700, 1500), (2950, 1490), (3200, 1455), (3450, 1375), (3650, 1270), (3800, 1175), (3900, 1100),
                (3960, 1050)])
HW_KABIN = Egri([(1250, 690), (1500, 760), (1900, 790), (2600, 800), (3200, 790), (3600, 750), (3850, 690), (3950, 650)])
EGIM_X = Egri([(1300, 0.45), (2600, 0.50), (3400, 0.55), (3900, 0.60)])   # cam evi yan eğimi (tumblehome) dy/dz
Z_BEL = 1000.0            # bel (cam alt) hizası referansı


def hw_kabin(x, z):
    return HW_KABIN(x) - EGIM_X(x) * (z - Z_BEL)


def govde_temel(X, Y, Z):
    """Detaysız gövde SDF'i (alt gövde + cam evi)."""
    hwx, dhwx = HW_PLAN(X), HW_PLAN.t(X)
    fz, dfz = F_DIKEY(Z), F_DIKEY.t(Z)
    hw = hwx * fz
    for xa in (AKS_ON, AKS_ARKA):                         # teker kemeri kabarıklığı + kıvrık dudak
        e = np.sqrt((X - xa) ** 2 + (Z - AKS_Z) ** 2) - CAMUR_R
        kab = sstep(-12.0, 0.0, e) * (1 - sstep(30.0, 120.0, e))
        dudak = 3.5 * np.exp(-((e - 6.0) / 5.0) ** 2)
        hw = hw + (10.0 * kab + dudak) * sstep(150, 250, Z)
    d_yan = (np.abs(Y) - hw) / np.sqrt(1 + (dhwx * fz) ** 2 + (hwx * dfz) ** 2)
    bombe = KAPUT_BOMBE(X)
    zu = Z_UST(X) - bombe * (Y / GEN_YARI) ** 2
    d_ust = (Z - zu) / np.sqrt(1 + Z_UST.t(X) ** 2 + (2 * bombe * Y / GEN_YARI ** 2) ** 2)
    d_alt = (Z_ALT(X) - Z) / np.sqrt(1 + Z_ALT.t(X) ** 2)
    d_on = (ON_PLAN(np.abs(Y)) + ON_YAN(Z) - X) / np.sqrt(1 + ON_PLAN.t(np.abs(Y)) ** 2 + ON_YAN.t(Z) ** 2)
    d_arka = (X - (L - ARKA_PLAN(np.abs(Y)) - ARKA_YAN(Z))) / np.sqrt(1 + ARKA_PLAN.t(np.abs(Y)) ** 2 + ARKA_YAN.t(Z) ** 2)
    alt = smax(smax(d_yan, d_ust, p(3.5)), d_on, p(3.0))
    alt = smax(alt, d_arka, p(3.0))
    alt = smax(alt, d_alt, p(1.2))
    zk = Z_KABIN(X) - 45.0 * (Y / 650.0) ** 2
    d_kust = (Z - zk) / np.sqrt(1 + Z_KABIN.t(X) ** 2 + (90.0 * Y / 650.0 ** 2) ** 2)
    eg = EGIM_X(X)
    d_kyan = (np.abs(Y) - hw_kabin(X, Z)) / np.sqrt(1 + eg ** 2 + HW_KABIN.t(X) ** 2)
    kabin = smax(d_kyan, d_kust, p(2.6))
    kabin = smax(kabin, (Z_UST(X) - 40.0) - Z, p(0.5))
    kabin = smax(kabin, np.maximum(1150.0 - X, X - 3990.0), 40.0)
    return smin(alt, kabin, p(0.9))


def camurluk_kuyusu(X, Y, Z, xa):
    dx, dz = X - xa, Z - AKS_Z
    daire = np.sqrt(dx * dx + dz * dz) - CAMUR_R
    damla = np.maximum((dz + np.abs(dx)) / math.sqrt(2) - CAMUR_R, dz - 470.0)
    return np.maximum(smin(daire, damla, 30.0), KUYU_Y - Y)


# ============================================================ tekerlek (İLK TAHMİN: jant fotoğraftan çizilecek)
def tekerlek_sdf(dx, dz, t, yon=None):
    """t: dışa doğru eksen (gerçek mm), teker merkezine göre."""
    r = np.sqrt(dx * dx + dz * dz)
    lastik = smax(r - TEKER_R, np.abs(t) - TEKER_YG, 25.0)
    lastik = smax(lastik, JANT_R - 6.0 - r, 4.0)
    jant = np.maximum(r - JANT_DIS, np.abs(t + 15.0) - 85.0)
    return np.minimum(lastik, jant)


# ============================================================ detaylar
def detaylar(d, d0, b):
    return d


def asimetrik(d, x, y, z, mod):
    return d


def arac_sdf(b, mod):
    X, Y, Z = b.x[:, None, None], b.y[None, :, None], b.z[None, None, :]
    d0 = govde_temel(X, Y, Z)
    d = detaylar(d0.copy(), d0, b)
    for xa, ty in ((AKS_ON, TEKER_Y_ON), (AKS_ARKA, TEKER_Y_ARKA)):
        g = b.bolge(xa - 520, xa + 520, KUYU_Y - 30, 1100, -30, 820)
        if g is None:
            continue
        sl, Xs, Ys, Zs = g
        d[sl] = smax(d[sl], -camurluk_kuyusu(Xs, Ys, Zs, xa), p(0.25))
        r = np.sqrt((Xs - xa) ** 2 + (Zs - AKS_Z) ** 2)
        if mod == "tek":
            stub = np.maximum(r - p(2.0), np.maximum(615.0 - Ys, Ys - 700.0))
            d[sl] = np.minimum(d[sl], stub)
            d[sl] = np.minimum(d[sl], tekerlek_sdf(Xs - xa, Zs - AKS_Z, Ys - ty))
        else:
            boss = np.maximum(r - p(2.3), np.maximum(KUYU_Y - 12.0 - Ys, Ys - (ty - TEKER_YG - 5.0)))
            d[sl] = np.minimum(d[sl], boss)
            delik = np.sqrt((Xs - xa) ** 2 + (Zs - AKS_Z) ** 2) - p(AKS_DELIK_R)
            d[sl] = np.maximum(d[sl], -np.broadcast_to(delik, d[sl].shape))
    if mod == "tek":
        d = np.maximum(d, 6.0 - Z)   # lastik tabanında düzlük (tablaya oturma)
    return d


# ============================================================ ana program
def main():
    ap = argparse.ArgumentParser(description="Opel Astra J Sedan 3D baskı maketi")
    ap.add_argument("--olcek", type=float, default=24.0, help="ölçek paydası (24 -> 1:24)")
    ap.add_argument("--parca", choices=["tek", "kit", "hepsi"], default="hepsi")
    ap.add_argument("--h", type=float, default=0.15, help="baskı ölçeğinde ızgara aralığı (mm)")
    ap.add_argument("--tolerans", type=float, default=0.005, help="sadeleştirme toleransı (mm)")
    ap.add_argument("--onek", default="astra_sedan")
    a = ap.parse_args()
    n = a.olcek
    for parca, ad in (("tek", "tek_parca"), ("kit", "govde")):
        if a.parca not in (parca, "hepsi"):
            continue
        print(f"[{ad}]")
        v, f = govde_mesh(arac_sdf, parca, a.h, n, KUTU, asimetrik=asimetrik)
        print(f"  {len(f)} üçgen -> sadeleştiriliyor (tolerans {a.tolerans} mm)")
        vs, fs = sadelestir(v, f, a.tolerans)
        del v, f
        gc.collect()
        kaydet(vs, fs, f"{a.onek}_{ad}.stl")
    if a.parca in ("kit", "hepsi"):
        print("[teker]")
        teker_mesh(lambda dx, dz, t: tekerlek_sdf(dx, dz, t, yon=1.0), min(a.h, 0.07), n, a.tolerans,
                   f"{a.onek}_teker.stl", yaricap=TEKER_R + 20.0, yari_gen=TEKER_YG + 30.0, delik_r=TEKER_DELIK_R)


if __name__ == "__main__":
    main()
