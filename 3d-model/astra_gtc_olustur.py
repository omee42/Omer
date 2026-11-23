#!/usr/bin/env python3
"""
Opel Astra J GTC (2011–2018, 3 kapı, OPC donanım) — 3D baskı maketi üretici.

Tüm geometri GERÇEK aracın milimetre ölçüleriyle tanımlanır, baskı ölçeğine en sonda küçültülür.
    boy 4466, genişlik 1840 (aynalarla 2020), yükseklik 1482, aks mesafesi 2695,
    iz genişliği ön 1584 / arka 1588, yerden yükseklik ~112, lastik 245/35 R20 (OPC).
Gövde işaretli mesafe alanı (SDF) ile modellenir ve marching cubes ile pürüzsüz yüzeye çevrilir.

Çıktılar:
    astra_gtc_tek_parca.stl   tekerlekler gövdeye bağlı, tek parça
    astra_gtc_govde.stl       tekerleksiz gövde (alt yüzeyi düz, aks delikli)  ┐ kit
    astra_gtc_teker.stl       tek tekerlek (4 adet basılır)                     ┘

Kullanım:
    pip install numpy scipy scikit-image trimesh manifold3d
    python3 astra_gtc_olustur.py                    # 1:24, ızgara 0.15 mm, hepsi
    python3 astra_gtc_olustur.py --olcek 18         # 1:18 (≈ 248 mm)
    python3 astra_gtc_olustur.py --parca tek --h 0.12

Marka logosu (Opel şimşeği) ve yazılar eklenmez; rozet yerleri boş halka olarak bırakılır.
"""
import argparse
import math
import time

import numpy as np
from scipy.interpolate import PchipInterpolator
from manifold3d import CrossSection, JoinType

K = 24.0  # detay ölçüleri 1:24 baskıdaki mm olarak verilir; p() gerçek mm'ye çevirir


def p(v):
    return v * K


# ============================================================ ana ölçüler (gerçek mm)
L = 4466.0
AKS_ON, AKS_ARKA = 935.0, 3630.0
TEKER_R = 340.0          # 245/35 R20 -> çap 679.5
JANT_R = 254.0           # 20 inç
TEKER_YG = 122.0         # lastik yarı genişliği
TEKER_Y = 792.0          # iz genişliği / 2
AKS_Z = TEKER_R
CAMUR_R = 372.0
KUYU_Y = 640.0           # çamurluk kuyusunun iç duvarı
AKS_DELIK_R = 1.05       # kit, BASKI mm: 1.75 mm filament gövdede serbest döner
TEKER_DELIK_R = 0.90     # kit, BASKI mm: tekerlekte sıkı geçme


# ============================================================ yardımcılar
class Egri:
    def __init__(self, pts):
        a = np.asarray(pts, float)
        self.lo, self.hi = a[0, 0], a[-1, 0]
        self.f = PchipInterpolator(a[:, 0], a[:, 1])
        self.df = self.f.derivative()

    def __call__(self, x):
        return self.f(np.clip(x, self.lo, self.hi))

    def t(self, x):
        xc = np.clip(x, self.lo, self.hi)
        return np.where((x > self.lo) & (x < self.hi), self.df(xc), 0.0)


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def sd_poligon(px, py, v):
    v = np.asarray(v, float)
    d = (px - v[0, 0]) ** 2 + (py - v[0, 1]) ** 2
    s = np.ones(np.broadcast(px, py).shape)
    for i in range(len(v)):
        j = i - 1
        ex, ey = v[j, 0] - v[i, 0], v[j, 1] - v[i, 1]
        wx, wy = px - v[i, 0], py - v[i, 1]
        h = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0, 1)
        bx, by = wx - ex * h, wy - ey * h
        d = np.minimum(d, bx * bx + by * by)
        c1 = py >= v[i, 1]
        c2 = py < v[j, 1]
        c3 = ex * wy > ey * wx
        s = np.where((c1 & c2 & c3) | (~c1 & ~c2 & ~c3), -s, s)
    return s * np.sqrt(d)


class YPoligon:
    """Köşeleri r ile yuvarlatılmış poligon (SDF)."""

    def __init__(self, pts, r):
        a = np.asarray(pts, float)
        alan = 0.5 * np.sum(a[:, 0] * np.roll(a[:, 1], -1) - np.roll(a[:, 0], -1) * a[:, 1])
        if alan < 0:
            a = a[::-1]
        self.ic = np.asarray(CrossSection([a.tolist()]).offset(-r, JoinType.Miter).to_polygons()[0])
        self.r = r

    def __call__(self, px, py):
        return sd_poligon(px, py, self.ic) - self.r


def sd_cizgi(px, py, pts):
    pts = np.asarray(pts, float)
    d = None
    for a, b in zip(pts[:-1], pts[1:]):
        ex, ey = b - a
        wx, wy = px - a[0], py - a[1]
        h = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0, 1)
        dd = (wx - ex * h) ** 2 + (wy - ey * h) ** 2
        d = dd if d is None else np.minimum(d, dd)
    return np.sqrt(d)


def sd_kutu2(px, py, cx, cy, hx, hy, r=0.0):
    qx = np.abs(px - cx) - hx + r
    qy = np.abs(py - cy) - hy + r
    return np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - r


def sd_kutu3(x, y, z, c, h, r=0.0):
    qx = np.abs(x - c[0]) - h[0] + r
    qy = np.abs(y - c[1]) - h[1] + r
    qz = np.abs(z - c[2]) - h[2] + r
    dis = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2)
    return dis + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0) - r


def sd_elipsoid(x, y, z, c, r):
    px, py, pz = (x - c[0]) / r[0], (y - c[1]) / r[1], (z - c[2]) / r[2]
    k0 = np.sqrt(px * px + py * py + pz * pz)
    k1 = np.sqrt((px / r[0]) ** 2 + (py / r[1]) ** 2 + (pz / r[2]) ** 2)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def petek(py, pz, s):
    """Bal peteği ağının kenarına uzaklık (s: altıgen yarıçapı)."""
    w = math.sqrt(3) * s
    hz = 3.0 * s
    ay = py - w * np.round(py / w)
    az = pz - hz * np.round(pz / hz)
    by_ = py - w / 2 - w * np.round((py - w / 2) / w)
    bz = pz - 1.5 * s - hz * np.round((pz - 1.5 * s) / hz)
    sec = (ay * ay + az * az) < (by_ * by_ + bz * bz)
    qy = np.where(sec, ay, by_)
    qz = np.where(sec, az, bz)
    k = np.maximum(np.abs(qy), np.maximum(np.abs(0.5 * qy + 0.8660254 * qz), np.abs(0.5 * qy - 0.8660254 * qz)))
    return s * 0.8660254 - k


def oyuk(d, d0, maske, derinlik):
    s = np.maximum(-(d0 + derinlik), maske)
    return np.maximum(d, -s)


def doldur(d, d0, maske, seviye):
    return np.minimum(d, np.maximum(d0 - seviye, maske))


# ============================================================ profiller (gerçek mm)
Z_ALT_UST = Egri([(0, 470), (20, 590), (60, 680), (120, 750), (200, 790), (400, 830), (700, 890),
                  (1000, 945), (1250, 985), (1350, 1000), (1600, 1012), (2000, 1035), (2600, 1066),
                  (3000, 1088), (3400, 1118), (3700, 1132), (3950, 1118), (4130, 1092), (4250, 1075),
                  (4330, 1062), (4375, 1050), (4410, 1018), (4435, 962), (4452, 880), (4462, 790),
                  (4466, 700)])
KAPUT_BOMBE = Egri([(0, 0), (250, 8), (600, 22), (1000, 35), (1300, 40), (1450, 10), (4000, 10), (4300, 30)])
Z_ALT = Egri([(0, 300), (30, 190), (80, 140), (200, 118), (400, 112), (4100, 112), (4250, 150),
              (4380, 230), (4440, 300), (4466, 380)])
HW_PLAN = Egri([(0, 900), (935, 905), (1500, 893), (2300, 885), (3000, 892), (3630, 900), (4466, 888)])
F_DIKEY = Egri([(100, 0.89), (150, 0.915), (220, 0.95), (340, 0.985), (480, 0.998), (620, 1.0),
                (740, 0.992), (860, 0.968), (960, 0.938), (1100, 0.88)])
ON_PLAN = Egri([(0, 0), (200, 20), (400, 75), (550, 138), (680, 212), (770, 288), (840, 368), (885, 440),
                (905, 495), (918, 550), (926, 600)])
ON_YAN = Egri([(100, 90), (150, 45), (200, 20), (300, 5), (400, 0), (520, 0), (600, 8), (680, 28),
               (760, 62), (840, 120)])
ARKA_PLAN = Egri([(0, 0), (200, 5), (400, 20), (550, 50), (680, 100), (780, 175), (850, 270), (895, 380),
                  (915, 480), (925, 560)])
ARKA_YAN = Egri([(100, 95), (150, 62), (200, 35), (250, 18), (300, 8), (400, 2), (450, 0), (700, 0),
                 (800, 4), (880, 13), (960, 30), (1015, 55), (1050, 90), (1100, 170)])
Z_KABIN = Egri([(900, 600), (1150, 880), (1300, 975), (1400, 1030), (1600, 1145), (1800, 1255),
                (2000, 1350), (2150, 1410), (2300, 1447), (2450, 1466), (2600, 1472), (2800, 1466),
                (3100, 1444), (3400, 1412), (3650, 1377), (3800, 1352), (3880, 1335), (3940, 1310),
                (4020, 1262), (4120, 1196), (4220, 1128), (4300, 1075), (4360, 1030), (4420, 900)])
HW_KABIN = Egri([(1100, 665), (1350, 752), (1700, 788), (2300, 794), (2800, 795), (3200, 797),
                 (3600, 798), (3900, 772), (4250, 702)])
EGIM_X = Egri([(1500, 0.47), (2400, 0.50), (2900, 0.54), (3300, 0.64), (3650, 0.74), (4000, 0.80)])


def hw_kabin(x, z):
    """Kabin (cam) yan yüzeyi: içe eğim arkaya doğru artar (C direği içe yatar)."""
    return HW_KABIN(x) - (z - 1000.0) * EGIM_X(x)


def omuz_z(x):
    return 860.0 + 0.048 * (x - 600.0)


# ============================================================ detay şekilleri (gerçek mm)
ON_CAM = YPoligon([(1610, 1018), (2000, 1032), (2500, 1056), (2845, 1080), (2838, 1390), (2600, 1392),
                   (2380, 1374), (2200, 1316), (2000, 1222)], 22)
ARKA_YAN_CAM = YPoligon([(2905, 1084), (3100, 1104), (3250, 1145), (3360, 1205), (3440, 1270),
                         (3300, 1344), (3100, 1368), (2900, 1384)], 18)
FAR_ON = YPoligon([(398, 632), (470, 636), (600, 646), (720, 662), (820, 684), (885, 710), (960, 738),
                   (960, 836), (870, 822), (760, 806), (640, 788), (545, 762), (480, 726), (432, 688)], 8)
FAR_YAN = YPoligon([(-60, 600), (250, 620), (480, 680), (640, 742), (760, 808), (800, 838), (740, 848),
                    (500, 850), (-60, 850)], 8)
UST_IZGARA = YPoligon([(-470, 726), (-360, 734), (-200, 731), (0, 729), (200, 731), (360, 734), (470, 726),
                       (440, 702), (360, 674), (220, 652), (0, 644), (-220, 652), (-360, 674), (-440, 702)], 10)
ALT_IZGARA = YPoligon([(-630, 505), (-520, 463), (-380, 440), (-200, 428), (0, 425), (200, 428), (380, 440),
                       (520, 463), (630, 505), (602, 410), (565, 290), (535, 192), (-535, 192), (-565, 290),
                       (-602, 410)], 14)
KOSE_HAVA = YPoligon([(718, 612), (790, 580), (832, 490), (845, 382), (832, 286), (790, 220), (680, 188),
                      (722, 262), (736, 360), (726, 462), (706, 552)], 10)
STOP_ARKA = YPoligon([(385, 1002), (470, 984), (600, 960), (740, 938), (850, 924), (910, 934),
                      (930, 990), (905, 1040), (780, 1046), (620, 1044), (490, 1032)], 10)
STOP_YAN = YPoligon([(4060, 1012), (4200, 978), (4330, 950), (4500, 930), (4500, 1046),
                     (4330, 1046), (4170, 1034)], 10)
OPC_KOSE = YPoligon([(640, 235), (770, 235), (815, 295), (820, 465), (780, 478), (695, 370)], 14)
DIFUZOR = YPoligon([(-610, 150), (610, 150), (585, 345), (-585, 345)], 25)
EGZOZ = YPoligon([(85, 195), (255, 195), (240, 285), (100, 285)], 12)
KAPI_CIZGILERI = [
    [(1340, 300), (1350, 700), (1390, 960), (1445, 1002), (1640, 1080), (2000, 1252), (2200, 1340),
     (2400, 1392), (2600, 1404), (2878, 1400)],
    [(2878, 1400), (2882, 1080), (2900, 700), (2900, 300)],
    [(1340, 300), (2900, 300)],
    [(705, 818), (722, 720), (700, 615)],     # ön tampon - çamurluk (far arka ucundan teker kemerine)
    [(4185, 958), (4060, 800), (3900, 640)],  # arka tampon - çamurluk (stop alt köşesine)
]


def kaput_poligon():
    # ön kenar yüzey üzerinde önden görünüşte z = 745 (orta) ... 815 (far üst kenarı) çizgisine oturur
    on = [(116, 0), (120, 150), (134, 300), (156, 400), (197, 500), (263, 600), (340, 700), (406, 760)]
    yan = [(1330, 705)]
    sag = on + yan
    return np.array([(x, y) for x, y in sag] + [(x, -y) for x, y in reversed(sag) if y > 0])


KAPUT = kaput_poligon()


# ============================================================ gövde
def govde_temel(X, Y, Z):
    hwx, dhwx = HW_PLAN(X), HW_PLAN.t(X)
    fz, dfz = F_DIKEY(Z), F_DIKEY.t(Z)
    hw = hwx * fz
    # omuz çizgisi
    zc = omuz_z(X)
    hw = hw + p(0.3) * np.maximum(0, 1 - np.abs(Z - zc) / p(2.0)) * sstep(500, 900, X) * (1 - sstep(4250, 4400, X))
    # GTC "bıçak" çizgisi: kapının önünde eşiğe yakın başlar, arka çamurluğa doğru yükselir
    zb = 365.0 + (X - 1430.0) * 0.158
    alt_b = zb - Z
    oy = p(0.75) * sstep(0, p(0.12), alt_b) * (1 - sstep(p(1.5), p(6.5), alt_b))
    hw = hw - oy * sstep(1400, 1900, X) * (1 - sstep(3150, 3330, X))
    # OPC marşpiyel
    hw = hw + p(0.35) * sstep(120, 150, Z) * (1 - sstep(250, 280, Z)) * sstep(1300, 1420, X) * (1 - sstep(3150, 3280, X))
    # çamurluk kabarıklıkları (arka daha belirgin)
    for xa, a in ((AKS_ON, 0.6), (AKS_ARKA, 0.85)):
        r = np.sqrt((X - xa) ** 2 + (Z - AKS_Z) ** 2)
        hw = hw + p(a) * sstep(CAMUR_R, CAMUR_R + p(0.4), r) * (1 - sstep(CAMUR_R + p(1.2), CAMUR_R + p(3.8), r)) * sstep(150, 250, Z)
    # GTC arka basen (haunch): arka teker üstünde omuz bölgesi dışa şişer
    hw = hw + 32.0 * sstep(2950, 3550, X) * (1 - sstep(4000, 4400, X)) * sstep(700, 900, Z) * (1 - sstep(1010, 1120, Z))
    d_yan = (Y - hw) / np.sqrt(1 + (dhwx * fz) ** 2 + (hwx * dfz) ** 2)

    bombe = KAPUT_BOMBE(X)
    zu = Z_ALT_UST(X) - bombe * (Y / 900.0) ** 2
    kaput = sstep(120, 260, X) * (1 - sstep(1250, 1350, X))
    zu = zu + kaput * (p(1.1) * np.maximum(0, 1 - Y / 330.0) ** 2.2 + p(0.3) * np.maximum(0, 1 - np.abs(Y - 690) / 110.0))
    d_ust = (Z - zu) / np.sqrt(1 + Z_ALT_UST.t(X) ** 2 + (2 * bombe * Y / 900.0 ** 2) ** 2)
    d_alt = (Z_ALT(X) - Z) / np.sqrt(1 + Z_ALT.t(X) ** 2)
    d_on = (ON_PLAN(Y) + ON_YAN(Z) - X) / np.sqrt(1 + ON_PLAN.t(Y) ** 2 + ON_YAN.t(Z) ** 2)
    d_arka = (X - (L - ARKA_PLAN(Y) - ARKA_YAN(Z))) / np.sqrt(1 + ARKA_PLAN.t(Y) ** 2 + ARKA_YAN.t(Z) ** 2)

    alt = smax(d_yan, d_ust, p(4.5))
    alt = smax(alt, d_on, p(3.0))
    alt = smax(alt, d_arka, p(3.5))
    alt = smax(alt, d_alt, p(1.2))

    zk = Z_KABIN(X) - 40.0 * (Y / 650.0) ** 2
    d_kust = (Z - zk) / np.sqrt(1 + Z_KABIN.t(X) ** 2 + (80.0 * Y / 650.0 ** 2) ** 2)
    eg = EGIM_X(X)
    d_kyan = (Y - hw_kabin(X, Z)) / np.sqrt(1 + eg ** 2 + (HW_KABIN.t(X) - (Z - 1000.0) * EGIM_X.t(X)) ** 2)
    kabin = smax(d_kyan, d_kust, p(2.6))
    kabin = smax(kabin, (Z_ALT_UST(X) - 35.0) - Z, p(0.5))
    kabin = np.maximum(kabin, X - 4380.0)
    kabin = np.maximum(kabin, 950.0 - X)

    d = smin(alt, kabin, p(0.9))
    # tavan spoyleri: tavanın devamı, 28 mm kalın dudak, hafif dışbükey arka kenar, camın üstüne taşar
    yy = np.minimum((Y / 540.0) ** 2, 1.0)
    sp_ust = 1352.0 - 0.10 * np.maximum(X - 3800.0, 0) - 30.0 * yy
    sp = np.maximum(X - (3965.0 + 20.0 * (1 - yy)), 3790.0 - X)
    sp = smax(sp, Y - (hw_kabin(X, sp_ust) - 12.0), 15.0)
    sp = smax(sp, np.maximum(Z - sp_ust, (sp_ust - 28.0) - Z), 8.0)
    return smin(d, sp, p(0.5))


def camurluk_kuyusu(X, Y, Z, xa):
    dx, dz = X - xa, Z - AKS_Z
    daire = np.sqrt(dx * dx + dz * dz) - CAMUR_R
    damla = np.maximum((dz + np.abs(dx)) / math.sqrt(2) - CAMUR_R, dz - 470.0)
    damla = np.maximum(damla, -dz)
    ic = np.minimum(daire, damla)
    return np.maximum(np.where(Y < 800.0, ic, daire), KUYU_Y - Y)


# ============================================================ tekerlek (gerçek mm, yerel)
def tekerlek_sdf(dx, dz, t):
    r = np.sqrt(dx * dx + dz * dz)
    th = np.arctan2(dz, dx)
    # lastik
    lastik = sd_kutu2(r, t, (JANT_R + TEKER_R) / 2 + 4, 0.0, (TEKER_R - JANT_R) / 2 - 4, TEKER_YG, p(1.6))
    for tc in (-80.0, -28.0, 28.0, 80.0):
        kanal = np.maximum(np.abs(t - tc) - p(0.22), (TEKER_R - p(0.55)) - r)
        lastik = np.maximum(lastik, -kanal)
    s = np.mod(r * th, p(1.7)) - p(0.85)
    eg = s + 0.25 * (np.abs(t) - 95.0)
    enine = np.maximum(np.maximum(np.abs(eg) - p(0.18), (TEKER_R - p(0.5)) - r), 100.0 - np.abs(t))
    lastik = np.maximum(lastik, -enine)
    halka = np.maximum(np.abs(r - 300.0) - p(0.12), (TEKER_YG - p(0.25)) - t)
    lastik = np.maximum(lastik, -halka)

    # jant
    arka = np.maximum(np.maximum(r - 262.0, -TEKER_YG - t), t + 15.0)
    dudak = sd_kutu2(r, t, 251.0, 92.0, 16.0, 22.0, 6.0)
    kovan = sd_kutu2(r, t, 242.0, 35.0, 18.0, 52.0, 3.0)
    disk = np.maximum(r - 182.0, np.abs(t + 8.0) - 12.0)
    disk_del = np.inf
    for k in range(18):  # delikli disk
        a = k * 2 * math.pi / 18
        for rr in (125.0, 155.0):
            aa = a + (0.0 if rr < 140 else math.pi / 18)
            dd = np.sqrt((dx - rr * math.cos(aa)) ** 2 + (dz - rr * math.sin(aa)) ** 2) - p(0.28)
            disk_del = np.minimum(disk_del, dd)
    disk = np.maximum(disk, -np.maximum(disk_del, np.abs(t + 8.0) - 30.0))
    sapka = np.maximum(r - 85.0, np.abs(t - 15.0) - 30.0)
    dth = np.angle(np.exp(1j * (th - 2.4)))
    kaliper = np.maximum(np.maximum(np.abs(r - 172.0) - 42.0, np.abs(dth) * r - 85.0), np.abs(t - 15.0) - 28.0)
    kaliper = smax(kaliper, -np.maximum(np.abs(r - 172.0) - 18.0, 30.0 - np.abs(dth) * r + 200.0), 2.0)
    # 5 çift kol
    a = np.mod(th + math.pi, 2 * math.pi / 5) - math.pi / 5
    # göbekte birleşen, janta doğru açılan V çift kollar; göbekte kalın, uca doğru incelir
    delta = np.clip(0.00115 * (r - 75.0), 0.0, None)
    kalin = 21.0 - 0.045 * (r - 70.0)
    kol_ang = r * np.abs(np.abs(a) - delta) - kalin
    yuz = 114.0 - 0.16 * (r - 70.0)            # derin içbükey jant yüzü
    kol = np.maximum(np.maximum(kol_ang, r - 244.0), 60.0 - r)
    kol = np.maximum(kol, np.maximum(t - yuz, (yuz - 46.0) - t))
    gobek = np.maximum(r - 78.0, np.maximum(25.0 - t, t - 108.0))
    kapak = np.maximum(r - 40.0, np.abs(t - 108.0) - 3.0)
    gobek = smin(gobek, kapak, 3.0)
    for k in range(5):
        ang = k * 2 * math.pi / 5 + math.pi / 5 - math.pi
        bx, bz = 56.0 * math.cos(ang), 56.0 * math.sin(ang)
        bij = np.maximum(np.sqrt((dx - bx) ** 2 + (dz - bz) ** 2) - p(0.32), 100.0 - t)
        gobek = np.maximum(gobek, -bij)

    jant = np.minimum(arka, dudak)
    jant = np.minimum(jant, kovan)
    jant = np.minimum(jant, disk)
    jant = np.minimum(jant, sapka)
    jant = np.minimum(jant, kaliper)
    jant = smin(jant, kol, 6.0)
    jant = smin(jant, gobek, 8.0)
    return np.minimum(lastik, jant)


# ============================================================ blok değerlendirme
class Blok:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def bolge(self, x0, x1, y0, y1, z0, z1):
        def aralik(a, lo, hi):
            return np.searchsorted(a, lo, "left"), np.searchsorted(a, hi, "right")
        ix, iy, iz = aralik(self.x, x0, x1), aralik(self.y, y0, y1), aralik(self.z, z0, z1)
        if ix[1] <= ix[0] or iy[1] <= iy[0] or iz[1] <= iz[0]:
            return None
        sl = (slice(*ix), slice(*iy), slice(*iz))
        return sl, self.x[sl[0]][:, None, None], self.y[sl[1]][None, :, None], self.z[sl[2]][None, None, :]


def detaylar(d, d0, b):
    def uygula(bb, fn):
        g = b.bolge(*bb)
        if g is not None:
            sl, X, Y, Z = g
            d[sl] = fn(d[sl], d0[sl], X, Y, Z)

    CAM = p(0.45)

    # ---------- camlar
    def on_cam(dd, z0, X, Y, Z):
        m = smax(smax(1365 - X, X - 2160, p(1.5)), Y - (hw_kabin(X, Z) - p(2.4)), p(1.5))
        return oyuk(dd, z0, np.maximum(m, 1032 - Z), CAM)
    uygula((1340, 2200, 0, 900, 990, 1500), on_cam)

    def arka_cam(dd, z0, X, Y, Z):
        m = smax(smax(3950 - X, X - 4285, p(1.2)), Y - (hw_kabin(X, Z) - p(2.6)), p(1.2))
        return oyuk(dd, z0, np.maximum(m, 1088 - Z), CAM)
    uygula((3900, 4320, 0, 900, 1060, 1400), arka_cam)

    def yan_cam(dd, z0, X, Y, Z):
        m = np.minimum(ON_CAM(X, Z), ARKA_YAN_CAM(X, Z))
        return oyuk(dd, z0, np.maximum(m, (hw_kabin(X, Z) - p(1.2)) - Y), CAM)
    uygula((1420, 3500, 450, 900, 1010, 1440), yan_cam)

    # ---------- kaput çizgisi (üstten)
    def kaput(dd, z0, X, Y, Z):
        s = sd_poligon(X, Y, KAPUT)
        m = np.maximum(np.maximum(np.abs(s) - p(0.22), 690 - Z), X - 1320)
        return oyuk(dd, z0, m, p(0.5))
    uygula((90, 1340, 0, 790, 680, 1020), kaput)

    # ---------- ön: üst ızgara, krom kanat çıtası, rozet halkası
    def ust_izgara(dd, z0, X, Y, Z):
        m = np.maximum(UST_IZGARA(Y, Z), X - 300)
        dd = oyuk(dd, z0, m, p(1.0))
        for zc in (664.0, 712.0):                       # ince yatay çıtalar (ızgara içi)
            dd = doldur(dd, z0, np.maximum(np.abs(Z - zc) - p(0.2), m), -p(0.45))
        # krom kanat çıtası: rozette kalın (yarı kalınlık 13), uçlara doğru incelir (6) ve yukarı kıvrılır
        ay = np.abs(Y)
        kanat = sd_cizgi(ay, Z, [(0, 688), (150, 690), (280, 700), (380, 716), (455, 728)]) - (13.0 - 7.0 * np.clip(ay / 455.0, 0, 1))
        dd = doldur(dd, z0, np.maximum(kanat, X - 300), p(0.3))
        rr = np.sqrt(Y ** 2 + (Z - 688) ** 2)
        dd = doldur(dd, z0, np.maximum(np.abs(rr - 50) - p(0.25), X - 300), p(0.4))   # rozet halkası Ø~105
        dd = doldur(dd, z0, np.maximum(rr - 44, X - 300), p(0.15))                    # rozet zemini
        return dd
    uygula((-40, 300, 0, 500, 620, 760), ust_izgara)

    # ---------- farlar (bumerang): önden FAR_ON ∩ yandan FAR_YAN
    def farlar(dd, z0, X, Y, Z):
        m = np.maximum(FAR_ON(Y, Z), FAR_YAN(X, Z))
        dd = oyuk(dd, z0, m, p(0.5) + p(0.7) * sstep(640.0, 800.0, Z))   # taban aşağı bakar: üst kenar hood altına girer
        # üst kenar boyunca kanat DRL ışık kılavuzu (yükseltilmiş şerit)
        drl = sd_cizgi(Y, Z, [(445, 690), (520, 735), (640, 770), (760, 790), (870, 806), (960, 818)]) - p(0.25)
        dd = doldur(dd, z0, np.maximum(drl, m), -p(0.3))
        # projektör (iç) ve uzun far (dış) çanakları
        for yc, zc, r in ((650.0, 716.0, 40.0), (800.0, 738.0, 30.0)):
            mer = np.sqrt((Y - yc) ** 2 + (Z - zc) ** 2)
            dd = oyuk(dd, z0, np.maximum(mer - (r - p(0.3)), m), p(1.1))             # mercek çukuru
        return dd
    uygula((-20, 820, 380, 960, 620, 870), farlar)

    # ---------- ön tampon: plaka, petek ızgara, köşe hava girişleri, splitter
    def on_tampon(dd, z0, X, Y, Z):
        plaka = sd_kutu2(Y, Z, 0, 520, 262, 56, 10)
        dd = oyuk(dd, z0, np.abs(plaka + p(0.15)) - p(0.15), p(0.3))   # plaka çevresinde gölge çizgisi
        dd = doldur(dd, z0, plaka + p(0.3), p(0.2))                     # plaka yüzü
        m = ALT_IZGARA(Y, Z)
        dd = doldur(dd, z0, np.abs(m + p(0.3)) - p(0.3), p(0.3))         # orta girişin çerçeve dudağı ("gülümseme")
        dd = oyuk(dd, z0, m, p(1.4))
        dd = doldur(dd, z0, np.maximum(petek(Y, Z, p(0.8)) - p(0.16), m), -p(0.5))
        k = KOSE_HAVA(Y, Z)
        dd = doldur(dd, z0, np.maximum(np.abs(k + p(0.25)) - p(0.25), X - 700), p(0.35))  # bıçak kenar
        dd = oyuk(dd, z0, np.maximum(k, X - 700), p(1.4))
        dd = doldur(dd, z0, np.maximum(petek(Y, Z, p(0.7)) - p(0.15), np.maximum(k, X - 700)), -p(0.55))
        # splitter dudağı: öne p(1.0) taşan keskin plaka, uçları köşe girişlerine kıvrılır
        dudak = np.maximum(sd_cizgi(np.abs(Y), Z, [(0, 160), (600, 162), (720, 176), (790, 205)]) - p(0.45), X - 600)
        dd = doldur(dd, z0, dudak, p(1.0))
        golge = np.maximum(sd_cizgi(np.abs(Y), Z, [(0, 186), (560, 188)]) - p(0.2), X - 400)
        dd = oyuk(dd, z0, golge, p(0.4))
        return dd
    uygula((-60, 700, 0, 940, 120, 640), on_tampon)

    # ---------- yan: kapı aralıkları, kol
    def kapilar(dd, z0, X, Y, Z):
        m = None
        for c in KAPI_CIZGILERI:
            s = sd_cizgi(X, Z, c)
            m = s if m is None else np.minimum(m, s)
        return oyuk(dd, z0, np.maximum(m - p(0.22), 480 - Y), p(0.5))
    uygula((300, 4200, 480, 960, 250, 1480), kapilar)

    def kol(dd, z0, X, Y, Z):
        xc, zc = 2650.0, omuz_z(2650.0) - 15
        cep = np.sqrt(((X - xc) / 85) ** 2 + ((Z - zc) / 28) ** 2) - 1.0
        dd = oyuk(dd, z0, np.maximum(cep * 28, 700 - Y), p(0.55))
        k = sd_kutu2(X, Z, xc, zc + 4, 70, 12, 11)
        return doldur(dd, z0, np.maximum(k, 700 - Y), p(0.3))
    uygula((2540, 2760, 700, 960, 820, 960), kol)

    # ---------- arka
    def arka(dd, z0, X, Y, Z):
        bagaj = sd_cizgi(Y, Z, [(585, 1100), (575, 1058), (500, 1052), (420, 1036), (372, 1008),
                                (365, 960), (372, 760), (352, 712), (310, 697), (0, 695)])
        dd = oyuk(dd, z0, np.maximum(bagaj - p(0.25), 4250 - X), p(0.6))
        st = np.maximum(STOP_ARKA(Y, Z), 4150 - X)
        dd = oyuk(dd, z0, st, p(0.6))
        g = sd_cizgi(Y, Z, [(420, 1000), (600, 976), (760, 956), (870, 950)]) - p(0.25)
        dd = doldur(dd, z0, np.maximum(g, st), -p(0.2))
        g2 = sd_cizgi(Y, Z, [(480, 1022), (640, 1026), (800, 1026), (870, 1020)]) - p(0.25)
        dd = doldur(dd, z0, np.maximum(g2, st), -p(0.2))
        trim = np.maximum(sd_cizgi(Y, Z, [(0, 992), (340, 990)]) - p(0.4), 4300 - X)
        dd = doldur(dd, z0, trim, p(0.2))
        rr = np.sqrt(Y ** 2 + (Z - 995) ** 2)
        dd = doldur(dd, z0, np.maximum(np.abs(rr - 72) - p(0.35), 4300 - X), p(0.3))
        dd = doldur(dd, z0, np.maximum(rr - 58, 4300 - X), p(0.12))
        plaka = np.maximum(sd_kutu2(Y, Z, 0, 790, 272, 66, 14), 4300 - X)
        dd = oyuk(dd, z0, plaka, p(0.5))
        dd = doldur(dd, z0, np.maximum(sd_kutu2(Y, Z, 0, 790, 255, 55, 8), 4300 - X), -p(0.2))
        dif = np.maximum(DIFUZOR(Y, Z), 4300 - X)
        dd = oyuk(dd, z0, dif, p(0.8))
        for yc in (300.0, 400.0, 500.0):
            dd = doldur(dd, z0, np.maximum(np.abs(Y - yc) - p(0.25), dif), -p(0.15))
        eg = np.maximum(EGZOZ(Y, Z), 4300 - X)
        dd = doldur(dd, z0, np.maximum(np.abs(eg + p(0.25)) - p(0.25), 4300 - X), p(0.35))
        dd = oyuk(dd, z0, eg + p(0.5), p(3.0))
        tampon = sd_cizgi(Y, Z, [(372, 700), (600, 704), (720, 730), (800, 790), (860, 880)]) - p(0.25)
        dd = oyuk(dd, z0, np.maximum(np.maximum(tampon, 4150 - X), Y - 860), p(0.6))
        kose = np.maximum(OPC_KOSE(Y, Z), 4150 - X)
        dd = oyuk(dd, z0, kose, p(0.9))
        ref = np.maximum(sd_kutu2(Y, Z, 772, 362, 15, 78, 10), 4150 - X)
        dd = doldur(dd, z0, np.maximum(ref, kose), -p(0.3))
        return dd
    uygula((4100, 4520, 0, 960, 130, 1150), arka)

    def stop_yan(dd, z0, X, Y, Z):
        st = np.maximum(STOP_YAN(X, Z), 780 - Y)
        dd = oyuk(dd, z0, st, p(0.6))
        g = sd_cizgi(X, Z, [(4090, 1008), (4300, 975), (4480, 968)]) - p(0.25)
        dd = doldur(dd, z0, np.maximum(g, st), -p(0.2))
        g = sd_cizgi(X, Z, [(4180, 1026), (4330, 1028), (4480, 1024)]) - p(0.25)
        return doldur(dd, z0, np.maximum(g, st), -p(0.15))
    uygula((4040, 4520, 760, 960, 910, 1070), stop_yan)

    # ---------- köpekbalığı anten
    def anten(dd, z0, X, Y, Z):
        return smin(dd, sd_elipsoid(X, Y, Z, (3560.0, 0.0, 1395.0), (p(3.6), p(0.85), p(2.5))), p(0.5))
    uygula((3450, 3680, 0, 60, 1320, 1470), anten)

    # ---------- aynalar: kapıya bağlı ince ayak + damla planlı kapak (önü bombeli, arkası düz cam yüzü)
    def ayna(dd, z0, X, Y, Z):
        def kapak(Xq):
            dis = sd_elipsoid(Xq, Y, Z, (1600.0, 908.0, 1098.0), (105.0, 102.0, 66.0))    # dış lob (geniş, derin)
            ic = sd_elipsoid(Xq, Y, Z, (1612.0, 830.0, 1090.0), (72.0, 62.0, 46.0))       # iç lob (sığ, alçak) -> damla
            return smin(dis, ic, 28.0)
        g = smax(kapak(X), X - 1612.0, 5.0)                                            # arka cam yüzü düz
        g = smax(g, 790.0 - Y, 8.0)
        ayak = sd_kutu3(X, Y, Z, (1588.0, 782.0, 1052.0), (30.0, 46.0, 13.0), 10.0)    # ince yatay kol
        taban = sd_kutu3(X, Y, Z, (1548.0, 770.0, 1036.0), (68.0, 28.0, 26.0), 18.0)   # kapı köşesi tabanı
        g = smin(g, smin(ayak, taban, 14.0), 16.0)
        cam = np.maximum(kapak(np.full_like(X, 1606.0)) + p(0.45), 1598.0 - X)       # cam, kapak kenarından 0.45 mm içeride
        g = oyuk(g, g.copy(), cam, p(0.45))
        sin = np.maximum(sd_cizgi(X, Z, [(1515, 1058), (1600, 1046)]) - p(0.2), 965 - Y)
        g = oyuk(g, g.copy(), sin, p(0.25))                                            # sinyal çizgisi
        return smin(dd, g, p(0.4))
    uygula((1420, 1680, 660, 1060, 980, 1200), ayna)
    return d


def asimetrik(d, x, y, z, mod):
    """Yalnız bir tarafta olan detaylar (tam hacim üzerinde, gerçek mm koordinatları)."""
    def kes(x0, x1, y0, y1, z0, z1):
        ix = slice(*np.searchsorted(x, [x0, x1]))
        iy = slice(*np.searchsorted(y, [y0, y1]))
        iz = slice(*np.searchsorted(z, [z0, z1]))
        if ix.start >= ix.stop or iy.start >= iy.stop or iz.start >= iz.stop:
            return None
        return (ix, iy, iz), x[ix][:, None, None], y[iy][None, :, None], z[iz][None, None, :]

    # yakıt kapağı (sağ arka, +Y)
    g = kes(3200, 3460, 800, 960, 780, 1030)
    if g:
        sl, X, Y, Z = g
        s = sd_kutu2(X, Z, 3330.0, 905.0, 85.0, 85.0, 30.0)
        cur = d[sl]
        d[sl] = np.maximum(cur, -np.maximum(-(cur + p(0.45)), np.abs(s) - p(0.2)))
    # silecekler (sol direksiyon, park konumu)
    g = kes(1370, 1520, -760, 760, 1000, 1120)
    if g:
        sl, X, Y, Z = g
        sil = np.minimum(sd_cizgi(X, Y, [(1452, 700), (1418, 40)]),
                         sd_cizgi(X, Y, [(1445, -20), (1412, -690)])) - p(0.3)
        kol = np.minimum(sd_cizgi(X, Y, [(1478, 640), (1440, 330), (1430, 300)]),
                         sd_cizgi(X, Y, [(1470, -80), (1440, -400), (1432, -430)])) - p(0.35)
        cur = d[sl]
        e = np.maximum(cur - p(0.75), np.minimum(sil, kol))
        d[sl] = np.minimum(cur, e)
    if mod == "kit":
        for xa in (AKS_ON, AKS_ARKA):
            g = kes(xa - 100, xa + 100, -700, 700, AKS_Z - 100, AKS_Z + 100)
            if g:
                sl, X, Y, Z = g
                delik = np.sqrt((X - xa) ** 2 + (Z - AKS_Z) ** 2) - p(AKS_DELIK_R)
                d[sl] = np.maximum(d[sl], -np.broadcast_to(delik, d[sl].shape))


def arac_sdf(b, mod):
    X, Y, Z = b.x[:, None, None], b.y[None, :, None], b.z[None, None, :]
    d0 = govde_temel(X, Y, Z)
    d = detaylar(d0.copy(), d0, b)
    for xa in (AKS_ON, AKS_ARKA):
        g = b.bolge(xa - 520, xa + 520, KUYU_Y - 30, 1100, -30, 830)
        if g is None:
            continue
        sl, Xs, Ys, Zs = g
        d[sl] = smax(d[sl], -camurluk_kuyusu(Xs, Ys, Zs, xa), p(0.4))
        r = np.sqrt((Xs - xa) ** 2 + (Zs - AKS_Z) ** 2)
        if mod == "tek":
            stub = np.maximum(r - p(2.0), np.maximum(615.0 - Ys, Ys - 700.0))
            d[sl] = np.minimum(d[sl], stub)
            d[sl] = np.minimum(d[sl], tekerlek_sdf(Xs - xa, Zs - AKS_Z, Ys - TEKER_Y))
        else:
            boss = np.maximum(r - p(2.3), np.maximum(KUYU_Y - 12.0 - Ys, Ys - (TEKER_Y - TEKER_YG - 5.0)))
            d[sl] = np.minimum(d[sl], boss)
    if mod == "tek":
        d = np.maximum(d, 6.0 - Z)  # lastik tabanında 0.25 mm düzlük (tablaya oturma)
    return d


# ============================================================ meshleme (parça parça)
def _ters(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0


def govde_mesh(mod, h_baski, n, parca=96, dilim=8):
    from skimage.measure import marching_cubes
    h = h_baski * n  # gerçek mm cinsinden ızgara
    x = np.arange(-40.0, 4520.0, h)
    yh = np.arange(0.0, 1060.0, h)
    z = np.arange(-20.0, 1540.0, h)
    y = np.concatenate([-yh[:0:-1], yh])
    nx = len(x)
    V, F, ofs = [], [], 0
    t0 = time.time()
    for c0 in range(0, nx - 1, parca):
        c1 = min(nx - 1, c0 + parca)
        xs = x[c0:c1 + 1]
        yarim = np.empty((len(xs), len(yh), len(z)), np.float32)
        for i0 in range(0, len(xs), dilim):
            i1 = min(len(xs), i0 + dilim)
            yarim[i0:i1] = arac_sdf(Blok(xs[i0:i1], yh, z), mod)
        tam = np.concatenate([yarim[:, :0:-1, :], yarim], axis=1)
        del yarim
        asimetrik(tam, xs, y, z, mod)
        if c0 == 0:
            tam[0] = 1e3
        if c1 == nx - 1:
            tam[-1] = 1e3
        tam[:, 0] = tam[:, -1] = 1e3
        tam[:, :, 0] = tam[:, :, -1] = 1e3
        if tam.min() < 0:
            v, f, _, _ = marching_cubes(tam, level=0.0137, allow_degenerate=False)
            v = v.astype(np.float64)
            v[:, 0] += c0
            V.append(v)
            F.append(f + ofs)
            ofs += len(v)
        print(f"    {mod}: %{100 * c1 / (nx - 1):.0f} ({time.time() - t0:.0f} sn)", flush=True)
    v = np.concatenate(V)
    f = np.concatenate(F)
    # parça sınırlarındaki ortak köşeleri birleştir (indeks uzayında birebir aynıdır)
    anahtar = np.round(v * 1e4).astype(np.int64)
    _, ilk, ters = np.unique(anahtar, axis=0, return_index=True, return_inverse=True)
    v = v[ilk]
    f = ters.reshape(-1)[f]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    v = v * h + np.array([x[0], y[0], z[0]])
    if _ters(v, f):
        f = f[:, ::-1]
    return v / n, f  # baskı mm


def manifold_mu(f):
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, c = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(c == 2))


def sadelestir(v, f, tolerans):
    """Su geçirmezliği koruyan sadeleştirme: yüzey en fazla 'tolerans' mm sapar."""
    from manifold3d import Manifold, Mesh
    m = Manifold(Mesh(vert_properties=np.asarray(v, np.float32), tri_verts=np.asarray(f, np.uint32)))
    if m.status().name != "NoError":
        print(f"    uyarı: manifold kontrolü {m.status()}, sadeleştirme atlandı")
        return v, f
    m = m.simplify(tolerans)
    g = m.to_mesh()
    return np.asarray(g.vert_properties)[:, :3].astype(np.float64), np.asarray(g.tri_verts).astype(np.int64)


def kaydet(v, f, yol):
    import trimesh
    tm = trimesh.Trimesh(v, f, process=False)
    par = tm.split(only_watertight=False)
    if len(par) > 1:
        tm = max(par, key=lambda m: len(m.faces))
    tm.vertices[:, 2] -= tm.vertices[:, 2].min()
    tm.export(yol)
    e = tm.extents
    print(f"  -> {yol}: {e[0]:.1f} x {e[1]:.1f} x {e[2]:.1f} mm, {len(tm.faces)} üçgen, "
          f"su geçirmez={tm.is_watertight}, hacim={tm.volume / 1000:.1f} cm3")
    return tm


def teker_mesh(h_baski, n, tolerans, yol):
    from skimage.measure import marching_cubes
    h = h_baski * n
    a = np.arange(-360.0, 360.0 + h, h)
    t = np.arange(-140.0, 140.0 + h, h)
    DX, DZ, T = a[:, None, None], a[None, :, None], t[None, None, :]
    vol = tekerlek_sdf(DX, DZ, T)
    r = np.sqrt(DX ** 2 + DZ ** 2)
    delik = np.maximum(r - p(TEKER_DELIK_R), T - 40.0)
    vol = np.maximum(vol, -delik).astype(np.float32)
    vol[0] = vol[-1] = 1e3
    vol[:, 0] = vol[:, -1] = 1e3
    vol[:, :, 0] = vol[:, :, -1] = 1e3
    v, f, _, _ = marching_cubes(vol, level=0.0, spacing=(h, h, h), allow_degenerate=False)
    v += np.array([a[0], a[0], t[0]])
    if _ters(v, f):
        f = f[:, ::-1]
    v, f = sadelestir(v / n, f, tolerans)
    return kaydet(v, f, yol)


def main():
    ap = argparse.ArgumentParser(description="Opel Astra J GTC 3D baskı maketi")
    ap.add_argument("--olcek", type=float, default=24.0, help="ölçek paydası (24 -> 1:24)")
    ap.add_argument("--parca", choices=["tek", "kit", "hepsi"], default="hepsi")
    ap.add_argument("--h", type=float, default=0.15, help="baskı ölçeğinde ızgara aralığı (mm)")
    ap.add_argument("--tolerans", type=float, default=0.005, help="sadeleştirme toleransı (mm)")
    ap.add_argument("--onek", default="astra_gtc")
    a = ap.parse_args()
    n = a.olcek
    if a.parca in ("tek", "hepsi"):
        print("[tek parça]")
        v, f = govde_mesh("tek", a.h, n)
        print(f"  {len(f)} üçgen -> sadeleştiriliyor (tolerans {a.tolerans} mm)")
        kaydet(*sadelestir(v, f, a.tolerans), f"{a.onek}_tek_parca.stl")
    if a.parca in ("kit", "hepsi"):
        print("[kit gövde]")
        v, f = govde_mesh("kit", a.h, n)
        print(f"  {len(f)} üçgen -> sadeleştiriliyor (tolerans {a.tolerans} mm)")
        kaydet(*sadelestir(v, f, a.tolerans), f"{a.onek}_govde.stl")
        print("[teker]")
        teker_mesh(min(a.h, 0.07), n, a.tolerans, f"{a.onek}_teker.stl")


if __name__ == "__main__":
    main()
