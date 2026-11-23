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
                  (1000, 945), (1250, 985), (1350, 1000), (1600, 1015), (2000, 1040), (2600, 1075),
                  (3000, 1100), (3400, 1125), (3700, 1135), (3950, 1110), (4130, 1065), (4250, 1030),
                  (4350, 975), (4420, 890), (4455, 770), (4466, 650)])
KAPUT_BOMBE = Egri([(0, 25), (200, 45), (1300, 45), (1450, 10), (4000, 10), (4300, 30)])
Z_ALT = Egri([(0, 300), (30, 190), (80, 140), (200, 118), (400, 112), (4100, 112), (4250, 150),
              (4380, 230), (4440, 300), (4466, 380)])
HW_PLAN = Egri([(0, 900), (935, 920), (1500, 903), (2300, 895), (3000, 905), (3630, 922), (4466, 905)])
F_DIKEY = Egri([(100, 0.89), (150, 0.915), (220, 0.95), (340, 0.985), (480, 0.998), (620, 1.0),
                (740, 0.995), (860, 0.975), (960, 0.95), (1100, 0.90)])
ON_PLAN = Egri([(0, 0), (200, 12), (400, 45), (550, 95), (680, 170), (770, 250), (840, 345), (885, 420),
                (905, 480), (918, 540), (926, 600)])
ON_YAN = Egri([(100, 90), (150, 45), (200, 20), (300, 5), (400, 0), (500, 0), (560, 12), (640, 45),
               (720, 100), (800, 180)])
ARKA_PLAN = Egri([(0, 0), (200, 5), (400, 20), (550, 50), (680, 100), (780, 175), (850, 270), (895, 380),
                  (915, 480), (925, 560)])
ARKA_YAN = Egri([(100, 70), (150, 40), (200, 22), (300, 8), (450, 0), (650, 0), (800, 15), (900, 45),
                 (1000, 95), (1100, 170)])
Z_KABIN = Egri([(900, 600), (1150, 880), (1300, 975), (1400, 1030), (1600, 1145), (1800, 1255),
                (2000, 1350), (2150, 1410), (2300, 1450), (2500, 1475), (2800, 1472), (3100, 1448),
                (3400, 1410), (3650, 1372), (3800, 1350), (3870, 1335), (3920, 1300), (4000, 1225),
                (4080, 1150), (4160, 1075), (4220, 1020), (4320, 880), (4420, 500)])
HW_KABIN = Egri([(1100, 700), (1350, 790), (1700, 830), (2500, 845), (3100, 835), (3500, 805),
                 (3900, 750), (4250, 680)])
EGIM = 0.5   # yan camların içe eğimi


def hw_kabin(x, z):
    return HW_KABIN(x) - (z - 1000.0) * EGIM


def omuz_z(x):
    return 830.0 + 0.045 * (x - 600.0)


# ============================================================ detay şekilleri (gerçek mm)
ON_CAM = YPoligon([(1440, 1030), (2840, 1080), (2840, 1405), (2500, 1428), (2200, 1402)], 22)
ARKA_YAN_CAM = YPoligon([(2900, 1083), (3150, 1110), (3470, 1175), (3150, 1372), (2900, 1397)], 22)
FAR_ON = YPoligon([(360, 622), (470, 642), (620, 660), (760, 676), (850, 690), (908, 712), (908, 808),
                   (840, 816), (690, 806), (560, 792), (460, 738), (395, 662)], 10)
FAR_YAN = YPoligon([(40, 700), (300, 718), (520, 765), (720, 815), (650, 835), (420, 818), (200, 808),
                    (40, 800)], 10)
UST_IZGARA = YPoligon([(-385, 655), (-300, 690), (0, 705), (300, 690), (385, 655), (330, 600),
                       (150, 572), (0, 566), (-150, 572), (-330, 600)], 14)
ALT_IZGARA = YPoligon([(-560, 445), (560, 445), (645, 285), (600, 205), (-600, 205), (-645, 285)], 30)
KOSE_HAVA = YPoligon([(650, 300), (760, 255), (825, 300), (815, 525), (745, 545), (695, 455)], 14)
STOP_ARKA = YPoligon([(260, 960), (500, 940), (760, 930), (880, 960), (915, 1010), (860, 1042),
                      (600, 1036), (330, 1012)], 10)
STOP_YAN = YPoligon([(4120, 1000), (4300, 958), (4500, 948), (4500, 1052), (4330, 1052),
                     (4170, 1032)], 10)
DIFUZOR = YPoligon([(-720, 150), (720, 150), (680, 330), (-680, 330)], 25)
EGZOZ = YPoligon([(85, 195), (255, 195), (240, 285), (100, 285)], 12)
KAPI_CIZGILERI = [
    [(1340, 300), (1350, 700), (1390, 1000), (1400, 1012), (2180, 1442), (2500, 1462), (2875, 1452)],
    [(2875, 1452), (2880, 1090), (2900, 700), (2900, 300)],
    [(1340, 300), (2900, 300)],
    [(330, 705), (690, 640)],                 # ön tampon - çamurluk
    [(4150, 950), (3900, 640)],               # arka tampon - çamurluk
]


def kaput_poligon():
    ys = np.linspace(-640, 640, 33)
    on = [(170 + 0.00025 * y * y, y) for y in ys]
    return np.array(on + [(1330, 705), (1330, -705)])


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
    zb = 380.0 + (X - 1450.0) * 0.178
    ust = Z - zb
    oy = p(0.6) * sstep(0, p(0.15), ust) * (1 - sstep(p(1.0), p(6.0), ust))
    hw = hw - oy * sstep(1380, 1600, X) * (1 - sstep(3050, 3300, X))
    # OPC marşpiyel
    hw = hw + p(0.35) * sstep(120, 150, Z) * (1 - sstep(250, 280, Z)) * sstep(1300, 1420, X) * (1 - sstep(3150, 3280, X))
    # çamurluk kabarıklıkları (arka daha belirgin)
    for xa, a in ((AKS_ON, 0.6), (AKS_ARKA, 0.85)):
        r = np.sqrt((X - xa) ** 2 + (Z - AKS_Z) ** 2)
        hw = hw + p(a) * sstep(CAMUR_R, CAMUR_R + p(0.4), r) * (1 - sstep(CAMUR_R + p(1.2), CAMUR_R + p(3.8), r)) * sstep(150, 250, Z)
    d_yan = (Y - hw) / np.sqrt(1 + (dhwx * fz) ** 2 + (hwx * dfz) ** 2)

    bombe = KAPUT_BOMBE(X)
    zu = Z_ALT_UST(X) - bombe * (Y / 900.0) ** 2
    kaput = sstep(150, 300, X) * (1 - sstep(1250, 1350, X))
    zu = zu + kaput * (p(0.5) * np.maximum(0, 1 - Y / p(4.0)) + p(0.25) * np.maximum(0, 1 - np.abs(Y - 700) / p(3.0)))
    d_ust = (Z - zu) / np.sqrt(1 + Z_ALT_UST.t(X) ** 2 + (2 * bombe * Y / 900.0 ** 2) ** 2)
    d_alt = (Z_ALT(X) - Z) / np.sqrt(1 + Z_ALT.t(X) ** 2)
    d_on = (ON_PLAN(Y) + ON_YAN(Z) - X) / np.sqrt(1 + ON_PLAN.t(Y) ** 2 + ON_YAN.t(Z) ** 2)
    d_arka = (X - (L - ARKA_PLAN(Y) - ARKA_YAN(Z))) / np.sqrt(1 + ARKA_PLAN.t(Y) ** 2 + ARKA_YAN.t(Z) ** 2)

    alt = smax(d_yan, d_ust, p(4.5))
    alt = smax(alt, d_on, p(4.0))
    alt = smax(alt, d_arka, p(3.5))
    alt = smax(alt, d_alt, p(1.2))

    zk = Z_KABIN(X) - 24.0 * (Y / 700.0) ** 2
    d_kust = (Z - zk) / np.sqrt(1 + Z_KABIN.t(X) ** 2 + (48.0 * Y / 700.0 ** 2) ** 2)
    d_kyan = (Y - hw_kabin(X, Z)) / np.sqrt(1 + EGIM ** 2 + HW_KABIN.t(X) ** 2)
    kabin = smax(d_kyan, d_kust, p(2.6))
    kabin = smax(kabin, 850.0 - Z, p(0.5))
    kabin = np.maximum(kabin, X - 4380.0)
    kabin = np.maximum(kabin, 950.0 - X)

    d = smin(alt, kabin, p(0.9))
    spoiler = sd_kutu3(X, Y, Z, (3868.0, 0.0, 1324.0), (78.0, 470.0 - 0.6 * np.maximum(X - 3868.0, 0), 13.0), 11.0)
    return smin(d, spoiler, p(0.8))


def camurluk_kuyusu(X, Y, Z, xa):
    dx, dz = X - xa, Z - AKS_Z
    daire = np.sqrt(dx * dx + dz * dz) - CAMUR_R
    damla = np.maximum((dz + np.abs(dx)) / math.sqrt(2) - CAMUR_R, dz - 470.0)
    damla = np.maximum(damla, -dz)
    ic = np.minimum(daire, damla)
    return np.maximum(np.where(Y < 860.0, ic, daire), KUYU_Y - Y)


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
    delta = 0.035 + 0.00105 * r
    kol_ang = r * np.abs(np.abs(a) - delta) - 13.0
    yuz = 108.0 - 0.11 * (r - 70.0)
    kol = np.maximum(np.maximum(kol_ang, r - 243.0), 60.0 - r)
    kol = np.maximum(kol, np.maximum(t - yuz, (yuz - 38.0) - t))
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
        m = smax(smax(3875 - X, X - 4165, p(1.2)), Y - (hw_kabin(X, Z) - p(3.0)), p(1.2))
        return oyuk(dd, z0, np.maximum(m, 1062 - Z), CAM)
    uygula((3850, 4200, 0, 900, 1040, 1400), arka_cam)

    def yan_cam(dd, z0, X, Y, Z):
        m = np.minimum(ON_CAM(X, Z), ARKA_YAN_CAM(X, Z))
        return oyuk(dd, z0, np.maximum(m, (hw_kabin(X, Z) - p(1.2)) - Y), CAM)
    uygula((1420, 3500, 450, 900, 1010, 1440), yan_cam)

    # ---------- kaput çizgisi (üstten)
    def kaput(dd, z0, X, Y, Z):
        s = sd_poligon(X, Y, KAPUT)
        m = np.maximum(np.maximum(np.abs(s) - p(0.22), 690 - Z), X - 1320)
        return oyuk(dd, z0, m, p(0.5))
    uygula((150, 1340, 0, 760, 680, 1020), kaput)

    # ---------- ön: üst ızgara, krom kanat çıtası, rozet halkası
    def ust_izgara(dd, z0, X, Y, Z):
        m = UST_IZGARA(Y, Z)
        dd = oyuk(dd, z0, m, p(0.9))
        for zc in (595.0, 675.0):
            dd = doldur(dd, z0, np.maximum(np.abs(Z - zc) - p(0.25), m), -p(0.5))
        kanat = sd_cizgi(Y, Z, [(0, 628), (150, 632), (280, 648), (390, 668)]) - p(0.55)
        dd = doldur(dd, z0, np.maximum(kanat, Y - 400), p(0.12))
        rr = np.sqrt(Y ** 2 + (Z - 630) ** 2)
        dd = doldur(dd, z0, np.abs(rr - 72) - p(0.3), p(0.3))
        dd = doldur(dd, z0, rr - 61, p(0.1))
        return dd
    uygula((-40, 300, 0, 420, 540, 730), ust_izgara)

    # ---------- farlar (bumerang)
    def farlar_on(dd, z0, X, Y, Z):
        m = np.maximum(FAR_ON(Y, Z), X - 400)
        dd = oyuk(dd, z0, m, p(0.7))
        drl = sd_cizgi(Y, Z, [(440, 712), (560, 735), (700, 752), (850, 770), (900, 790)]) - p(0.22)
        dd = doldur(dd, z0, np.maximum(drl, m), -p(0.2))
        alt = sd_cizgi(Y, Z, [(410, 672), (520, 700), (700, 718), (880, 738)]) - p(0.15)
        dd = doldur(dd, z0, np.maximum(alt, m), -p(0.35))
        for yc, zc in ((585.0, 765.0), (760.0, 778.0)):
            mer = np.sqrt((Y - yc) ** 2 + (Z - zc) ** 2)
            dd = doldur(dd, z0, np.maximum(mer - 42.0, m), -p(0.3))
            dd = oyuk(dd, z0, np.maximum(mer - 22.0, m), p(0.9))
        return dd
    uygula((-20, 420, 360, 940, 620, 830), farlar_on)

    def farlar_yan(dd, z0, X, Y, Z):
        m = np.maximum(FAR_YAN(X, Z), 760 - Y)
        dd = oyuk(dd, z0, m, p(0.7))
        drl = sd_cizgi(X, Z, [(60, 770), (300, 782), (540, 800), (690, 818)]) - p(0.22)
        return doldur(dd, z0, np.maximum(drl, m), -p(0.2))
    uygula((20, 760, 740, 940, 680, 850), farlar_yan)

    # ---------- ön tampon: plaka, petek ızgara, köşe hava girişleri, splitter
    def on_tampon(dd, z0, X, Y, Z):
        plaka = sd_kutu2(Y, Z, 0, 525, 262, 57, 12)
        dd = doldur(dd, z0, plaka, p(0.18))
        m = ALT_IZGARA(Y, Z)
        dd = oyuk(dd, z0, m, p(1.3))
        dd = doldur(dd, z0, np.maximum(petek(Y, Z, p(0.9)) - p(0.17), m), -p(0.4))
        k = KOSE_HAVA(Y, Z)
        dd = oyuk(dd, z0, k, p(1.1))
        dd = doldur(dd, z0, np.maximum(petek(Y, Z, p(0.8)) - p(0.16), k), -p(0.45))
        bicak = sd_cizgi(Y, Z, [(630, 275), (770, 232), (850, 285), (860, 380)]) - p(0.35)
        dd = doldur(dd, z0, bicak, p(0.25))
        # tampon üzerinde gülümseme çizgisi
        gul = sd_cizgi(Y, Z, [(0, 462), (300, 462), (560, 470), (760, 545), (880, 620)]) - p(0.2)
        dd = oyuk(dd, z0, gul, p(0.35))
        dudak = np.maximum(np.abs(Z - 160) - p(0.45), Y - 690)
        dd = doldur(dd, z0, dudak, p(0.35))
        return dd
    uygula((-40, 600, 0, 940, 120, 640), on_tampon)

    # ---------- yan: kapı aralıkları, kol
    def kapilar(dd, z0, X, Y, Z):
        m = None
        for c in KAPI_CIZGILERI:
            s = sd_cizgi(X, Z, c)
            m = s if m is None else np.minimum(m, s)
        return oyuk(dd, z0, np.maximum(m - p(0.22), 480 - Y), p(0.5))
    uygula((300, 4200, 480, 960, 250, 1480), kapilar)

    def kol(dd, z0, X, Y, Z):
        xc, zc = 2650.0, omuz_z(2650.0) - 45
        cep = np.sqrt(((X - xc) / 85) ** 2 + ((Z - zc) / 28) ** 2) - 1.0
        dd = oyuk(dd, z0, np.maximum(cep * 28, 700 - Y), p(0.55))
        k = sd_kutu2(X, Z, xc, zc + 4, 70, 12, 11)
        return doldur(dd, z0, np.maximum(k, 700 - Y), p(0.3))
    uygula((2540, 2760, 700, 960, 820, 960), kol)

    # ---------- arka
    def arka(dd, z0, X, Y, Z):
        bagaj = sd_kutu2(Y, Z, 0, 1060, 640, 370, 60)
        dd = oyuk(dd, z0, np.maximum(np.abs(bagaj) - p(0.22), 4280 - X), p(0.5))
        st = np.maximum(STOP_ARKA(Y, Z), 4180 - X)
        dd = oyuk(dd, z0, st, p(0.5))
        g = sd_cizgi(Y, Z, [(300, 990), (520, 975), (760, 972), (890, 998)]) - p(0.2)
        dd = doldur(dd, z0, np.maximum(g, st), -p(0.15))
        g2 = sd_cizgi(Y, Z, [(380, 962), (600, 952), (800, 958)]) - p(0.15)
        dd = doldur(dd, z0, np.maximum(g2, st), -p(0.3))
        trim = np.maximum(sd_cizgi(Y, Z, [(0, 1000), (285, 996)]) - p(0.35), 4300 - X)
        dd = doldur(dd, z0, trim, p(0.12))
        rr = np.sqrt(Y ** 2 + (Z - 1000) ** 2)
        dd = doldur(dd, z0, np.maximum(np.abs(rr - 80) - p(0.28), 4300 - X), p(0.25))
        dd = doldur(dd, z0, np.maximum(rr - 68, 4300 - X), p(0.08))
        plaka = np.maximum(sd_kutu2(Y, Z, 0, 760, 262, 57, 12), 4300 - X)
        dd = oyuk(dd, z0, plaka, p(0.35))
        dif = np.maximum(DIFUZOR(Y, Z), 4300 - X)
        dd = oyuk(dd, z0, dif, p(0.8))
        for yc in (350.0, 480.0, 600.0):
            dd = doldur(dd, z0, np.maximum(np.abs(Y - yc) - p(0.25), dif), -p(0.15))
        eg = np.maximum(EGZOZ(Y, Z), 4300 - X)
        dd = doldur(dd, z0, np.maximum(np.abs(eg + p(0.25)) - p(0.25), 4300 - X), p(0.35))
        dd = oyuk(dd, z0, eg + p(0.5), p(3.0))
        ref = np.maximum(sd_kutu2(Y, Z, 850, 450, 18, 70, 12), 4250 - X)
        dd = oyuk(dd, z0, ref, p(0.4))
        return dd
    uygula((4100, 4520, 0, 960, 130, 1150), arka)

    def stop_yan(dd, z0, X, Y, Z):
        st = np.maximum(STOP_YAN(X, Z), 800 - Y)
        dd = oyuk(dd, z0, st, p(0.5))
        g = sd_cizgi(X, Z, [(4150, 1010), (4320, 985), (4480, 990)]) - p(0.2)
        return doldur(dd, z0, np.maximum(g, st), -p(0.15))
    uygula((4100, 4520, 780, 960, 930, 1070), stop_yan)

    # ---------- köpekbalığı anten
    def anten(dd, z0, X, Y, Z):
        return smin(dd, sd_elipsoid(X, Y, Z, (3560.0, 0.0, 1395.0), (p(3.6), p(0.85), p(2.5))), p(0.5))
    uygula((3450, 3680, 0, 60, 1320, 1470), anten)

    # ---------- aynalar
    def ayna(dd, z0, X, Y, Z):
        kutu = sd_kutu3(X, Y, Z, (1565.0, 905.0, 1112.0), (72.0, 105.0, 66.0), 52.0)
        on = sd_elipsoid(X, Y, Z, (1535.0, 905.0, 1112.0), (92.0, 106.0, 68.0))
        g = smin(kutu, on, 25.0)
        ayak = sd_kutu3(X, Y, Z, (1555.0, 780.0, 1062.0), (52.0, 95.0, 26.0), 18.0)
        g = smin(g, ayak, 20.0)
        cam = np.maximum(sd_kutu2(Y, Z, 905.0, 1114.0, 92.0, 54.0, 32.0), 1600 - X)
        g = oyuk(g, g.copy(), cam, p(0.35))
        sin = sd_cizgi(X, Z, [(1480, 1085), (1560, 1080)]) - p(0.18)
        g = oyuk(g, g.copy(), np.maximum(sin, 940 - Y), p(0.25))  # sinyal çizgisi
        return smin(dd, g, p(0.4))
    uygula((1420, 1680, 660, 1040, 990, 1200), ayna)
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
                         sd_cizgi(X, Y, [(1445, -20), (1412, -690)])) - p(0.28)
        kol = np.minimum(sd_cizgi(X, Y, [(1478, 640), (1440, 330), (1430, 300)]),
                         sd_cizgi(X, Y, [(1470, -80), (1440, -400), (1432, -430)])) - p(0.3)
        cur = d[sl]
        e = np.maximum(cur - p(0.45), np.minimum(sil, kol))
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
