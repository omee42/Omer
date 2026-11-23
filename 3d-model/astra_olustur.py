#!/usr/bin/env python3
"""
Opel Astra (L, 2022+) 5 kapı hatchback — 3D baskı maketi üretici.

Model, işaretli mesafe alanı (SDF) ile tanımlanır ve marching cubes ile yüzeye
çevrilir. Bu yüzden gövde köşeli değil, gerçek bir araba gibi kavislidir.

Gerçek ölçüler (kamuya açık teknik veriler):
    boy 4374 mm, genişlik 1860 mm, yükseklik 1470 mm, aks mesafesi 2675 mm,
    lastik 225/40 R18. Tasarım 1:24 ölçekte yapılır (boy ≈ 182 mm).

Çıktılar (--parca):
    tek  -> astra_tek_parca.stl      tekerlekler gövdeye bağlı, tek parça
    kit  -> astra_govde.stl          tekerleksiz gövde (aks delikli, alt yüzeyi düz)
            astra_teker.stl          tek tekerlek (4 adet basılır, 1.75 mm filament aks)
    hepsi (varsayılan)

Kullanım:
    pip install numpy scipy scikit-image trimesh manifold3d pyfqmr
    python3 astra_olustur.py                       # 1:24, hepsi
    python3 astra_olustur.py --olcek 18            # 1:18 (≈ 243 mm)
    python3 astra_olustur.py --parca tek --h 0.15  # daha ince ızgara (daha yavaş)

Not: Marka logosu (Opel şimşeği) ve yazılar eklenmemiştir; ön ve arka rozet yerleri
boş halka olarak bırakılmıştır (çıkartma yapıştırmak için).
"""
import argparse
import math
import time

import numpy as np
from scipy.interpolate import PchipInterpolator
from manifold3d import CrossSection, JoinType

# ============================================================ genel ölçüler (1:24, mm)
L = 182.25                 # boy
HW = 38.75                 # yarı genişlik (aynasız)
AKS_ON, AKS_ARKA = 36.0, 147.5
TEKER_R = 13.25            # 225/40 R18 dış yarıçap
AKS_Z = TEKER_R
TEKER_Y = 33.3             # tekerlek orta düzlemi (iz genişliği / 2)
TEKER_YG = 4.7             # lastik yarı genişliği
JANT_R = 9.525             # 18 inç
CAMUR_R = 16.0             # çamurluk açıklığı
KUYU_Y = 26.4              # çamurluk kuyusunun iç duvarı
AKS_DELIK_R = 1.05         # kit: gövdedeki aks deliği (1.75 mm filament serbest döner)
TEKER_DELIK_R = 0.90       # kit: tekerlekteki kör delik (sıkı geçme)


# ============================================================ yardımcılar
class Egri:
    """Monoton kübik (PCHIP) eğri; uçlarda sabit kalır."""

    def __init__(self, pts):
        p = np.asarray(pts, float)
        self.lo, self.hi = p[0, 0], p[-1, 0]
        self.f = PchipInterpolator(p[:, 0], p[:, 1])
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


def yuvarlak_poligon(pts, r):
    """Köşeleri r ile yuvarlatılmış poligonun SDF'i için (içe ötelenmiş köşeler, r)."""
    p = np.asarray(pts, float)
    alan = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    if alan < 0:
        p = p[::-1]
    ic = CrossSection([p.tolist()]).offset(-r, JoinType.Miter).to_polygons()[0]
    return np.asarray(ic), r


def sd_ypoligon(px, py, yp):
    ic, r = yp
    return sd_poligon(px, py, ic) - r


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


def oyuk(d, d0, maske, derinlik):
    """Yüzeyden 'derinlik' kadar içeri, maske bölgesinde oyuk aç."""
    s = np.maximum(-(d0 + derinlik), maske)
    return np.maximum(d, -s)


def doldur(d, d0, maske, seviye):
    """Maske bölgesini yüzeye göre 'seviye' konumuna kadar doldur (+ dışarı taşar, - içeride kalır)."""
    return np.minimum(d, np.maximum(d0 - seviye, maske))


# ============================================================ profiller
# Kaput + bel çizgisi + bagaj kapağı (alt gövde üst yüzeyi, orta hat)
Z_ALT_UST = Egri([(0, 20.0), (0.6, 24.0), (1.6, 27.4), (3.5, 29.9), (8, 31.7), (20, 34.1), (35, 36.4),
                  (50, 38.4), (54, 39.0), (62, 40.3), (80, 41.3), (100, 42.3), (125, 43.6), (140, 44.6),
                  (155, 45.2), (166, 45.1), (170, 44.6), (175, 43.4), (178.5, 41.5), (181, 37.5),
                  (182, 32.0), (182.25, 27.0)])
KAPUT_BOMBE = Egri([(0, 1.0), (10, 1.6), (50, 1.6), (60, 0.35), (160, 0.35), (175, 1.0)])
Z_ALT = Egri([(0, 13.0), (1.5, 10.2), (5, 7.5), (12, 6.1), (20, 5.8), (165, 5.8), (174, 6.6),
              (179, 8.6), (181.5, 11.0), (182.25, 14.0)])
HW_PLAN = Egri([(0, 38.0), (36, 38.75), (60, 38.2), (100, 37.6), (128, 38.1), (147.5, 38.75), (182.25, 38.3)])
F_DIKEY = Egri([(4, 0.89), (6, 0.915), (9, 0.95), (14, 0.985), (20, 0.998), (26, 1.0), (31, 0.995),
                (36, 0.975), (40, 0.95), (46, 0.90)])
ON_PLAN = Egri([(0, 0.0), (10, 0.15), (20, 0.8), (27, 2.0), (32, 4.0), (35.5, 6.8), (37.6, 10.5),
                (38.6, 14.5), (39.5, 20.0)])
ON_YAN = Egri([(5, 3.5), (8, 1.6), (12, 0.6), (17, 0.0), (22, 0.0), (26, 0.6), (29, 1.8), (32, 4.0)])
ARKA_PLAN = Egri([(0, 0.0), (10, 0.2), (20, 0.8), (27, 1.9), (31.5, 3.6), (35, 6.2), (37.5, 10.0),
                  (38.9, 15.0), (39.8, 22.0)])
ARKA_YAN = Egri([(5, 3.4), (8, 1.8), (12, 0.8), (18, 0.3), (24, 0.0), (30, 0.2), (35, 0.9), (39, 2.0),
                 (42, 3.4), (45, 5.2)])
# Kabin (camlı bölüm)
Z_KABIN = Egri([(40, 25.0), (50, 37.4), (54, 40.6), (60, 43.8), (70, 49.2), (80, 54.2), (88, 58.1),
                (95, 60.3), (105, 61.25), (120, 61.0), (135, 59.9), (145, 58.7), (151, 57.8),
                (153, 56.6), (156, 54.6), (162, 50.3), (166.5, 46.3), (169, 44.2), (175, 36.0)])
HW_KABIN = Egri([(48, 31.0), (56, 33.4), (70, 35.0), (90, 35.6), (120, 35.4), (140, 34.5),
                 (155, 32.6), (166, 30.5), (175, 28.0)])
EGIM = 0.42  # yan camların içe eğimi (tan ~23°)


def omuz_z(x):
    return 35.6 + 0.016 * (x - 10.0)


def hw_kabin(x, z):
    return HW_KABIN(x) - (z - 42.0) * EGIM


# ============================================================ cam ve çizgi şekilleri
ON_CAM_YAN = yuvarlak_poligon([(63.6, 41.5), (97.0, 42.6), (97.0, 57.3), (92.0, 57.6), (87.0, 56.4)], 0.9)
ARKA_CAM_YAN = yuvarlak_poligon([(99.6, 42.7), (131.0, 44.3), (136.4, 47.6), (128.5, 56.2),
                                 (118.0, 57.4), (99.6, 57.7)], 0.9)
FAR = yuvarlak_poligon([(18.5, 26.2), (33.8, 25.0), (35.4, 27.0), (34.6, 29.0), (18.8, 28.6)], 0.5)
IZGARA = yuvarlak_poligon([(-23.5, 9.2), (23.5, 9.2), (21.5, 19.0), (-21.5, 19.0)], 1.6)
YAN_HAVA = yuvarlak_poligon([(26.5, 9.6), (32.5, 10.6), (33.5, 17.0), (28.0, 17.5)], 0.6)
STOP_ARKA = yuvarlak_poligon([(9.8, 40.3), (30.5, 39.8), (36.8, 40.4), (38.6, 42.4), (37.0, 43.0),
                              (10.6, 42.5)], 0.4)
STOP_YAN = yuvarlak_poligon([(171.0, 41.6), (175.5, 40.3), (184.0, 40.2), (184.0, 43.0), (176.5, 43.1),
                             (172.0, 42.6)], 0.4)
DIFUZOR = yuvarlak_poligon([(-27.0, 6.0), (27.0, 6.0), (25.5, 10.8), (-25.5, 10.8)], 0.8)
KAPI_CIZGILERI = [
    [(56.8, 8.8), (57.6, 24.0), (58.8, 39.8), (88.8, 58.8), (98.3, 59.6)],      # ön kapı ön kenar + üst
    [(98.3, 59.6), (98.3, 8.8)],                                                 # B direği
    [(98.3, 59.6), (118.0, 59.4), (127.5, 57.9), (137.8, 48.2), (134.8, 38.0),
     (133.8, 28.5), (133.5, 8.8)],                                               # arka kapı
    [(56.8, 8.8), (133.5, 8.8)],                                                 # kapı alt kenarı
    [(10.2, 26.0), (23.3, 23.9)],                                                # tampon-çamurluk birleşimi
]


# ============================================================ gövde
def govde_temel(X, Y, Z):
    """Kaporta (alt gövde + kabin + spoiler + çamurluk kuyuları). Y >= 0 yarısı."""
    # --- alt gövde yan yüzeyi
    hwx, dhwx = HW_PLAN(X), HW_PLAN.t(X)
    fz, dfz = F_DIKEY(Z), F_DIKEY.t(Z)
    hw = hwx * fz
    # omuz çizgisi (keskin sırt)
    zc = omuz_z(X)
    hw = hw + 0.35 * np.maximum(0, 1 - np.abs(Z - zc) / 2.2) * sstep(6, 14, X) * (1 - sstep(170, 178, X))
    # alt kapı oyuğu (içbükey, üst kenarı keskin, arkaya doğru yükselir)
    zt = 18.5 + 0.035 * (X - 60.0)
    bant = sstep(11.5, 12.5, Z) * (1 - sstep(zt - 0.3, zt + 0.3, Z))
    hw = hw - 0.5 * bant * sstep(54, 60, X) * (1 - sstep(127, 133, X))
    # çamurluk kabarıklıkları
    for xa in (AKS_ON, AKS_ARKA):
        r = np.sqrt((X - xa) ** 2 + (Z - AKS_Z) ** 2)
        hw = hw + 0.55 * sstep(CAMUR_R, CAMUR_R + 0.4, r) * (1 - sstep(17.8, 20.2, r)) * sstep(5, 8, Z)
    d_yan = (Y - hw) / np.sqrt(1 + (dhwx * fz) ** 2 + (hwx * dfz) ** 2)

    # --- üst yüzey (kaput bombesi + kaput çizgileri)
    bombe = KAPUT_BOMBE(X)
    ust = Z_ALT_UST(X) - bombe * (Y / 38.0) ** 2
    kaput = sstep(3, 8, X) * (1 - sstep(52, 57, X))
    ust = ust + kaput * (0.4 * np.maximum(0, 1 - Y / 3.5) + 0.25 * np.maximum(0, 1 - np.abs(Y - 21.0) / 2.5))
    d_ust = (Z - ust) / np.sqrt(1 + Z_ALT_UST.t(X) ** 2 + (2 * bombe * Y / 38.0 ** 2) ** 2)
    d_alt = (Z_ALT(X) - Z) / np.sqrt(1 + Z_ALT.t(X) ** 2)

    # --- ön ve arka yüz
    d_on = (ON_PLAN(Y) + ON_YAN(Z) - X) / np.sqrt(1 + ON_PLAN.t(Y) ** 2 + ON_YAN.t(Z) ** 2)
    d_arka = (X - (L - ARKA_PLAN(Y) - ARKA_YAN(Z))) / np.sqrt(1 + ARKA_PLAN.t(Y) ** 2 + ARKA_YAN.t(Z) ** 2)

    alt = smax(d_yan, d_ust, 4.5)
    alt = smax(alt, d_on, 4.0)
    alt = smax(alt, d_arka, 3.5)
    alt = smax(alt, d_alt, 1.2)

    # --- kabin
    zk = Z_KABIN(X) - 1.0 * (Y / 30.0) ** 2
    d_kust = (Z - zk) / np.sqrt(1 + Z_KABIN.t(X) ** 2 + (2 * Y / 900.0) ** 2)
    d_kyan = (Y - hw_kabin(X, Z)) / np.sqrt(1 + EGIM ** 2 + HW_KABIN.t(X) ** 2)
    kabin = smax(d_kyan, d_kust, 2.6)
    kabin = smax(kabin, 36.0 - Z, 0.5)

    d = smin(alt, kabin, 0.9)

    # --- tavan spoyleri
    spoiler = sd_kutu3(X, Y, Z, (150.6, 0.0, 57.25), (2.9, 23.0, 0.5), 0.45)
    d = smin(d, spoiler, 0.8)
    return d


def camurluk_kuyusu(X, Y, Z, xa):
    """Çamurluk kuyusu: dışta yuvarlak açıklık, içte destek gerektirmeyen damla (teardrop) tavan."""
    dx, dz = X - xa, Z - AKS_Z
    daire = np.sqrt(dx * dx + dz * dz) - CAMUR_R
    damla = np.maximum((dz + np.abs(dx)) / math.sqrt(2) - CAMUR_R, dz - 19.5)
    damla = np.maximum(damla, -dz)
    ic = np.minimum(daire, damla)
    d2 = np.where(Y < 36.2, ic, daire)
    return np.maximum(d2, KUYU_Y - Y)


# ============================================================ tekerlek (yerel koordinat)
def tekerlek_sdf(dx, dz, t):
    """dx, dz: aks merkezine göre (mm); t: dış yüze doğru eksen (0 = lastik ortası)."""
    r = np.sqrt(dx * dx + dz * dz)
    th = np.arctan2(dz, dx)

    # lastik: yuvarlatılmış omuzlu kesit
    lastik = sd_kutu2(r, t, 11.35, 0.0, 1.9, TEKER_YG, 1.7)
    # boyuna diş kanalları
    for tc in (-2.6, -0.9, 0.9, 2.6):
        kanal = np.maximum(np.abs(t - tc) - 0.22, (TEKER_R - 0.55) - r)
        lastik = np.maximum(lastik, -kanal)
    # omuzlarda enine diş blokları
    s = np.mod(r * th, 1.6) - 0.8
    enine = np.maximum(np.maximum(np.abs(s) - 0.18, (TEKER_R - 0.5) - r), 2.9 - np.abs(t))
    lastik = np.maximum(lastik, -enine)
    # yan duvar yazı bandı yerine ince bir halka çizgisi
    halka = np.maximum(np.abs(r - 11.6) - 0.12, (TEKER_YG - 0.25) - t)
    lastik = np.maximum(lastik, -halka)

    # jant parçaları
    arka_dolgu = np.maximum(np.maximum(r - 9.6, -TEKER_YG - t), t + 0.6)
    dudak = sd_kutu2(r, t, 9.27, 3.42, 0.67, 0.83, 0.25)
    gobek_kovan = sd_kutu2(r, t, 9.0, 1.2, 0.6, 1.8, 0.1)
    disk = np.maximum(r - 7.4, np.abs(t + 0.2) - 0.4)
    disk_sapka = np.maximum(r - 3.2, np.abs(t - 0.6) - 1.2)
    # fren kaliperi (üst-ön)
    dth = np.angle(np.exp(1j * (th - 2.4)))
    kaliper = np.maximum(np.maximum(np.abs(r - 6.95) - 1.35, np.abs(dth) * r - 2.6), np.abs(t - 0.55) - 1.05)
    kaliper = kaliper - 0.0
    # 5 çift kollu jant
    a = np.mod(th + math.pi, 2 * math.pi / 5) - math.pi / 5
    delta = 0.05 + 0.028 * r
    kol_ang = r * np.abs(np.abs(a) - delta) - 0.5
    yuz = 3.9 - 0.12 * (r - 2.5)
    kol = np.maximum(np.maximum(kol_ang, r - 9.1), 2.2 - r)
    kol = np.maximum(kol, np.maximum(t - yuz, (yuz - 1.4) - t))
    # göbek + bijon delikleri + merkez kapak
    gobek = np.maximum(r - 2.9, np.maximum(0.8 - t, t - 3.95))
    kapak = np.maximum(r - 1.5, np.abs(t - 3.95) - 0.12)
    gobek = smin(gobek, kapak, 0.1)
    for k in range(5):
        ang = k * 2 * math.pi / 5 + math.pi / 5 - math.pi
        bx, bz = 2.05 * math.cos(ang), 2.05 * math.sin(ang)
        bijon = np.maximum(np.sqrt((dx - bx) ** 2 + (dz - bz) ** 2) - 0.32, 3.6 - t)
        gobek = np.maximum(gobek, -bijon)

    jant = np.minimum(arka_dolgu, dudak)
    jant = np.minimum(jant, gobek_kovan)
    jant = np.minimum(jant, disk)
    jant = np.minimum(jant, disk_sapka)
    jant = np.minimum(jant, kaliper)
    jant = smin(jant, kol, 0.25)
    jant = smin(jant, gobek, 0.3)
    return np.minimum(lastik, jant)


# ============================================================ ızgara üzerinde değerlendirme
class Blok:
    """Bir x dilimi üzerinde, sınırlayıcı kutuya göre alt-bölge seçimi."""

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def bolge(self, x0, x1, y0, y1, z0, z1):
        def aralik(a, lo, hi):
            i0 = np.searchsorted(a, lo, "left")
            i1 = np.searchsorted(a, hi, "right")
            return i0, i1
        ix = aralik(self.x, x0, x1)
        iy = aralik(self.y, y0, y1)
        iz = aralik(self.z, z0, z1)
        if ix[1] <= ix[0] or iy[1] <= iy[0] or iz[1] <= iz[0]:
            return None
        sl = (slice(*ix), slice(*iy), slice(*iz))
        X = self.x[sl[0]][:, None, None]
        Y = self.y[sl[1]][None, :, None]
        Z = self.z[sl[2]][None, None, :]
        return sl, X, Y, Z


def detaylar(d, d0, b: Blok, mod):
    """Tüm yüzey detaylarını uygular (yarım model, Y >= 0)."""

    def uygula(bb, fn):
        g = b.bolge(*bb)
        if g is None:
            return
        sl, X, Y, Z = g
        d[sl] = fn(d[sl], d0[sl], X, Y, Z)

    # ---------- camlar (0.45 mm girinti)
    def on_cam(dd, dz0, X, Y, Z):
        m = smax(smax(57.6 - X, X - 88.2, 1.5), Y - (hw_kabin(X, Z) - 2.4), 1.5)
        m = np.maximum(m, 41.2 - Z)
        return oyuk(dd, dz0, m, 0.45)
    uygula((56, 90, 0, 40, 39, 63), on_cam)

    def arka_cam(dd, dz0, X, Y, Z):
        m = smax(smax(152.8 - X, X - 166.0, 1.2), Y - (hw_kabin(X, Z) - 3.4), 1.2)
        m = np.maximum(m, 46.6 - Z)
        return oyuk(dd, dz0, m, 0.45)
    uygula((151, 168, 0, 40, 44, 60), arka_cam)

    def yan_camlar(dd, dz0, X, Y, Z):
        m = np.minimum(sd_ypoligon(X, Z, ON_CAM_YAN), sd_ypoligon(X, Z, ARKA_CAM_YAN))
        m = np.maximum(m, (hw_kabin(X, Z) - 1.2) - Y)
        return oyuk(dd, dz0, m, 0.45)
    uygula((62, 138, 20, 40, 40, 59), yan_camlar)

    # ---------- kaput kapağı çizgisi (üstten görünüş)
    def kaput_cizgi(dd, dz0, X, Y, Z):
        s = sd_kutu2(X, Y, 31.5, 0.0, 27.2, 31.0, 7.0)
        m = np.maximum(np.abs(s) - 0.22, 28.0 - Z)
        m = np.maximum(m, X - 57.0)
        return oyuk(dd, dz0, m, 0.5)
    uygula((2, 58, 0, 34, 27, 42), kaput_cizgi)

    # ---------- ön: vizör paneli, farlar, rozet
    def vizor(dd, dz0, X, Y, Z):
        panel = sd_kutu2(Y, Z, 0.0, 26.8, 33.8, 2.6, 1.2)
        dd = oyuk(dd, dz0, panel, 0.25)
        far = sd_ypoligon(Y, Z, FAR)
        dd = oyuk(dd, dz0, far, 0.6)
        # gündüz farı bıçakları (çift çizgi imza)
        b1 = sd_cizgi(Y, Z, [(19.5, 27.9), (33.6, 27.6), (34.6, 28.4)]) - 0.22
        b2 = sd_cizgi(Y, Z, [(24.5, 26.6), (33.2, 26.1)]) - 0.22
        dd = doldur(dd, dz0, np.maximum(np.minimum(b1, b2), far), -0.25)
        # projektör mercekleri
        for yc in (22.0, 25.0, 28.0):
            mer = np.sqrt((Y - yc) ** 2 + (Z - 26.9) ** 2) - 0.55
            dd = doldur(dd, dz0, np.maximum(mer, far), -0.4)
        # rozet halkası (logo yok)
        rr = np.sqrt(Y ** 2 + (Z - 26.9) ** 2)
        halka = np.abs(rr - 2.7) - 0.28
        dd = doldur(dd, dz0, halka, 0.15)
        dd = doldur(dd, dz0, rr - 2.3, 0.0)
        return dd
    uygula((-1, 14, 0, 40, 22, 31), vizor)

    # ---------- ön tampon: ızgara, plaka, yan hava girişleri, alt çizgi
    def on_tampon(dd, dz0, X, Y, Z):
        izg = sd_ypoligon(Y, Z, IZGARA)
        dd = oyuk(dd, dz0, izg, 1.2)
        for zc in (10.6, 12.6, 14.6, 16.6, 18.3):
            cubuk = np.maximum(np.abs(Z - zc) - 0.28, izg)
            dd = doldur(dd, dz0, cubuk, -0.45)
        for yc in (-16.0, -8.0, 0.0, 8.0, 16.0):
            dik = np.maximum(np.abs(Y - yc) - 0.25, izg)
            dd = doldur(dd, dz0, dik, -0.6)
        plaka = sd_kutu2(Y, Z, 0.0, 14.3, 10.8, 2.3, 0.4)
        dd = doldur(dd, dz0, plaka, -0.15)
        yh = sd_ypoligon(Y, Z, YAN_HAVA)
        dd = oyuk(dd, dz0, yh, 0.9)
        dd = doldur(dd, dz0, np.maximum(np.abs(Z - 13.3) - 0.25, yh), -0.5)
        lip = np.maximum(np.abs(Z - 8.2) - 0.2, Y - 30.0)
        dd = oyuk(dd, dz0, lip, 0.35)
        return dd
    uygula((-1, 12, 0, 36, 5, 21), on_tampon)

    # ---------- yan: kapı aralıkları, kollar, yakıt kapağı
    def kapilar(dd, dz0, X, Y, Z):
        m = None
        for c in KAPI_CIZGILERI:
            s = sd_cizgi(X, Z, c)
            m = s if m is None else np.minimum(m, s)
        m = np.maximum(m - 0.22, 20.0 - Y)
        return oyuk(dd, dz0, m, 0.5)
    uygula((9, 139, 20, 40, 7, 61), kapilar)

    def kollar(dd, dz0, X, Y, Z):
        for xc in (86.5, 124.5):
            zc = omuz_z(xc) - 2.0
            cep = np.sqrt(((X - xc) / 3.4) ** 2 + ((Z - zc) / 1.15) ** 2) - 1.0
            cep = np.maximum(cep * 1.15, 30.0 - Y)
            dd = oyuk(dd, dz0, cep, 0.55)
            kol = sd_kutu2(X, Z, xc, zc + 0.15, 2.9, 0.5, 0.45)
            dd = doldur(dd, dz0, np.maximum(kol, 30.0 - Y), 0.3)
        return dd
    uygula((82, 129, 30, 41, 32, 40), kollar)

    # ---------- arka: bagaj kapağı, stoplar, plaka, rozet, difüzör, reflektörler
    def arka(dd, dz0, X, Y, Z):
        bagaj = sd_kutu2(Y, Z, 0.0, 49.5, 30.8, 20.5, 2.0)
        dd = oyuk(dd, dz0, np.maximum(np.abs(bagaj) - 0.22, 169.0 - X), 0.5)
        st = np.maximum(sd_ypoligon(Y, Z, STOP_ARKA), 172.0 - X)
        dd = oyuk(dd, dz0, st, 0.45)
        bar = np.maximum(sd_cizgi(Y, Z, [(11.5, 41.45), (36.6, 41.45)]) - 0.18, st)
        dd = doldur(dd, dz0, bar, -0.15)
        plaka = np.maximum(sd_kutu2(Y, Z, 0.0, 32.9, 10.8, 2.3, 0.5), 170.0 - X)
        dd = oyuk(dd, dz0, plaka, 0.35)
        rr = np.sqrt(Y ** 2 + (Z - 41.4) ** 2)
        dd = doldur(dd, dz0, np.maximum(np.abs(rr - 2.6) - 0.26, 175.0 - X), 0.15)
        dif = np.maximum(sd_ypoligon(Y, Z, DIFUZOR), 175.0 - X)
        dd = oyuk(dd, dz0, dif, 0.7)
        for yc in (6.0, 12.0, 18.0):
            fin = np.maximum(np.abs(np.abs(Y) - yc) - 0.25, dif)
            dd = doldur(dd, dz0, fin, -0.15)
        ref = np.maximum(sd_kutu2(Y, Z, 32.0, 14.1, 3.0, 0.5, 0.4), 172.0 - X)
        dd = oyuk(dd, dz0, ref, 0.35)
        # tampon üst çizgisi ve alt karakter çizgisi
        for zc in (27.6, 17.2):
            cz = np.maximum(np.abs(Z - zc) - 0.2, 172.0 - X)
            dd = oyuk(dd, dz0, cz, 0.4)
        return dd
    uygula((164, 184, 0, 40, 5, 52), arka)

    def stop_yan(dd, dz0, X, Y, Z):
        st = np.maximum(sd_ypoligon(X, Z, STOP_YAN), 34.0 - Y)
        dd = oyuk(dd, dz0, st, 0.45)
        bar = np.maximum(sd_cizgi(X, Z, [(172.5, 41.9), (183.5, 41.6)]) - 0.18, st)
        return doldur(dd, dz0, bar, -0.15)
    uygula((170, 184, 33, 40, 39, 44), stop_yan)

    # ---------- köpekbalığı yüzgeci anten
    def anten(dd, dz0, X, Y, Z):
        fin = sd_elipsoid(X, Y, Z, (141.0, 0.0, 58.9), (3.6, 0.85, 2.5))
        return smin(dd, fin, 0.5)
    uygula((136, 146, 0, 3, 55, 63), anten)

    # ---------- dikiz aynaları (kapıya monte)
    def ayna(dd, dz0, X, Y, Z):
        kutu = sd_kutu3(X, Y, Z, (67.0, 40.8, 45.4), (2.2, 3.3, 2.0), 1.5)
        on = sd_elipsoid(X, Y, Z, (66.3, 40.8, 45.4), (2.5, 3.3, 2.0))
        govde = smin(kutu, on, 0.6)
        ayak = sd_kutu3(X, Y, Z, (67.8, 36.4, 42.8), (1.8, 2.6, 1.2), 0.6)
        govde = smin(govde, ayak, 0.6)
        cam = np.maximum(sd_kutu2(Y, Z, 40.6, 45.4, 2.5, 1.35, 0.8), 67.5 - X)
        govde = oyuk(govde, govde.copy(), cam, 0.35)
        return smin(dd, govde, 0.4)
    uygula((62, 72, 32, 46, 40, 49), ayna)
    return d


def yakit_kapagi(d, x, y, z, olc):
    """Sağ arka çamurlukta yakıt kapağı (yalnızca +Y tarafı)."""
    ix = slice(*np.searchsorted(x, [155.0 * olc, 160.0 * olc]))
    iy = slice(*np.searchsorted(y, [34.0 * olc, 40.0 * olc]))
    iz = slice(*np.searchsorted(z, [30.5 * olc, 37.5 * olc]))
    X = x[ix][:, None, None] / olc
    Y = y[iy][None, :, None] / olc
    Z = z[iz][None, None, :] / olc
    s = sd_kutu2(X, Z, 157.5, 34.0, 2.3, 2.3, 1.0)
    m = np.abs(s) - 0.2
    # yerel yüzeye göre oyuk: mevcut d (ölçekli) kullanılarak
    sub = d[ix, iy, iz] / olc
    s3 = np.maximum(-(sub + 0.45), m)
    d[ix, iy, iz] = np.maximum(sub, -s3) * olc


def arac_sdf(b: Blok, mod):
    X = b.x[:, None, None]
    Y = b.y[None, :, None]
    Z = b.z[None, None, :]
    d0 = govde_temel(X, Y, Z)
    d = d0.copy()
    d = detaylar(d, d0, b, mod)
    for xa in (AKS_ON, AKS_ARKA):
        g = b.bolge(xa - 21, xa + 21, KUYU_Y - 1, 45, -1, 34)
        if g is None:
            continue
        sl, Xs, Ys, Zs = g
        d[sl] = smax(d[sl], -camurluk_kuyusu(Xs, Ys, Zs, xa), 0.4)
        if mod == "tek":
            stub = np.maximum(np.sqrt((Xs - xa) ** 2 + (Zs - AKS_Z) ** 2) - 2.0,
                              np.maximum(25.5 - Ys, Ys - 29.6))
            d[sl] = np.minimum(d[sl], stub)
            tk = tekerlek_sdf(Xs - xa, Zs - AKS_Z, Ys - TEKER_Y)
            d[sl] = np.minimum(d[sl], tk)
        else:
            dr = np.sqrt((Xs - xa) ** 2 + (Zs - AKS_Z) ** 2)
            boss = np.maximum(dr - 2.3, np.maximum(KUYU_Y - 0.5 - Ys, Ys - 28.2))
            d[sl] = np.minimum(d[sl], boss)
    return d


def kit_delikler(d, x, y, z, olc):
    """Gövde boyunca aks delikleri (baskı ölçüsünde, ölçekten bağımsız)."""
    for xa in (AKS_ON, AKS_ARKA):
        ix = slice(*np.searchsorted(x, [(xa - 3) * olc, (xa + 3) * olc]))
        iz = slice(*np.searchsorted(z, [(AKS_Z - 3) * olc, (AKS_Z + 3) * olc]))
        iy = slice(0, np.searchsorted(y, 29.0 * olc))
        X = x[ix][:, None, None]
        Z = z[iz][None, None, :]
        delik = np.sqrt((X - xa * olc) ** 2 + (Z - AKS_Z * olc) ** 2) - AKS_DELIK_R
        delik = np.broadcast_to(delik, d[ix, iy, iz].shape)
        d[ix, iy, iz] = np.maximum(d[ix, iy, iz], -delik)


# ============================================================ mesh işlemleri
def hacim_uret(mod, h, olc, dilim=12):
    """Yarım (Y>=0) SDF hacmini baskı ölçüsünde hesaplar, aynalayıp tam hacmi döndürür."""
    x = np.arange(-1.5 * olc, 185.0 * olc, h)
    y = np.arange(0.0, 46.0 * olc, h)
    z = np.arange(-1.0 * olc, 65.0 * olc, h)
    yarim = np.empty((len(x), len(y), len(z)), np.float32)
    t0 = time.time()
    for i0 in range(0, len(x), dilim):
        i1 = min(len(x), i0 + dilim)
        b = Blok(x[i0:i1] / olc, y / olc, z / olc)
        yarim[i0:i1] = arac_sdf(b, mod) * olc
        if (i0 // dilim) % 15 == 0:
            print(f"    {mod}: %{100 * i1 / len(x):.0f}  ({time.time() - t0:.0f} sn)", flush=True)
    if mod == "kit":
        kit_delikler(yarim, x, y, z, olc)
    tam = np.concatenate([yarim[:, :0:-1, :], yarim], axis=1)
    yt = np.concatenate([-y[:0:-1], y])
    yakit_kapagi(tam, x, yt, z, olc)
    return tam, x, yt, z


def _ters(v, f):
    """Hacim negatifse yüzey yönü terstir."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0


def yuzey_cikar(vol, x, y, z, h):
    from skimage.measure import marching_cubes
    vol[0, :, :] = vol[-1, :, :] = 1.0
    vol[:, 0, :] = vol[:, -1, :] = 1.0
    vol[:, :, 0] = vol[:, :, -1] = 1.0
    v, f, _, _ = marching_cubes(vol, level=0.0, spacing=(h, h, h), allow_degenerate=False)
    v += np.array([x[0], y[0], z[0]])
    if _ters(v, f):
        f = f[:, ::-1].copy()
    return v, f


def sadelestir(v, f, hedef):
    import pyfqmr
    if len(f) <= hedef:
        return v, f
    s = pyfqmr.Simplify()
    s.setMesh(v, f)
    s.simplify_mesh(target_count=hedef, aggressiveness=5, preserve_border=True, verbose=False)
    v2, f2, _ = s.getMesh()
    return v2, f2


def kaydet(v, f, yol):
    import trimesh
    tm = trimesh.Trimesh(v, f, process=False)
    # sayısal kırıntıları at, yalnızca ana parçayı tut
    parcalar = tm.split(only_watertight=False)
    if len(parcalar) > 1:
        tm = max(parcalar, key=lambda m: len(m.faces))
    tm.export(yol)
    e = tm.extents
    print(f"  -> {yol}: {e[0]:.1f} x {e[1]:.1f} x {e[2]:.1f} mm, {len(tm.faces)} üçgen, "
          f"su geçirmez={tm.is_watertight}, parça={tm.body_count}, hacim={tm.volume / 1000:.1f} cm3")
    return tm


def uret_govde(mod, h, olc, hedef, yol):
    print(f"[{mod}] SDF hesaplanıyor (ızgara {h} mm)...")
    vol, x, y, z = hacim_uret(mod, h, olc)
    print(f"[{mod}] yüzey çıkarılıyor ({vol.shape})...")
    v, f = yuzey_cikar(vol, x, y, z, h)
    del vol
    print(f"[{mod}] {len(f)} üçgen -> sadeleştiriliyor ({hedef})...")
    v, f = sadelestir(v, f, hedef)
    if mod == "tek":
        # lastiklerin altı 0.25 mm düzleştirilmiş: modeli tablaya oturt
        v[:, 2] -= v[:, 2].min()
    else:
        v[:, 2] -= v[:, 2].min()
    return kaydet(v, f, yol)


def uret_teker(h, olc, hedef, yol):
    print("[teker] SDF hesaplanıyor...")
    hh = h
    x = np.arange(-14.5 * olc, 14.5 * olc, hh)
    z = np.arange(-14.5 * olc, 14.5 * olc, hh)
    t = np.arange(-5.5 * olc, 5.5 * olc, hh)
    DX = x[:, None, None] / olc
    DZ = z[None, :, None] / olc
    T = t[None, None, :] / olc
    vol = (tekerlek_sdf(DX, DZ, T) * olc).astype(np.float32)
    # arka yüz tamamen düz olsun diye arka dolgu zaten t=-4.7'de; kör aks deliği
    r = np.sqrt((x[:, None, None]) ** 2 + (z[None, :, None]) ** 2)
    delik = np.maximum(r - TEKER_DELIK_R, t[None, None, :] - 1.5 * olc)
    vol = np.maximum(vol, -delik)
    v, f = yuzey_cikar(vol, x, z, t, hh)
    v, f = sadelestir(v, f, hedef)
    v[:, 2] -= v[:, 2].min()  # arka yüz tablada, jant yüzü yukarıda
    return kaydet(v, f, yol)


def main():
    ap = argparse.ArgumentParser(description="Opel Astra L hatchback 3D baskı maketi")
    ap.add_argument("--olcek", type=float, default=24.0, help="ölçek paydası: 24 -> 1:24 (varsayılan)")
    ap.add_argument("--parca", choices=["tek", "kit", "hepsi"], default="hepsi")
    ap.add_argument("--h", type=float, default=0.2, help="gövde SDF ızgara aralığı (mm)")
    ap.add_argument("--hedef", type=int, default=900_000, help="gövde için hedef üçgen sayısı")
    a = ap.parse_args()
    olc = 24.0 / a.olcek
    if a.parca in ("tek", "hepsi"):
        uret_govde("tek", a.h, olc, a.hedef, "astra_tek_parca.stl")
    if a.parca in ("kit", "hepsi"):
        uret_govde("kit", a.h, olc, a.hedef, "astra_govde.stl")
        uret_teker(min(a.h, 0.08), olc, 200_000, "astra_teker.stl")


if __name__ == "__main__":
    main()
