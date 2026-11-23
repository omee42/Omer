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

Ön ve arka rozetlerde Opel şimşeği kabartma olarak bulunur; yazılar eklenmez.
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
JANT_R = 254.0           # 20 inç (oturma yarıçapı)
JANT_DIS = 282.0         # görünen jant dudağı dış yarıçapı: fotoğraflarda jant/lastik(yüklü) oranı ~0.865
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
# ön: tampon alt hattı (ön eşleme); eşik: görünen etek alt çizgisi ön kapı altında ~185, arka teker önünde ~150 (yan eşleme)
Z_ALT = Egri([(0, 330), (30, 236), (70, 214), (150, 186), (250, 150), (400, 116), (520, 112), (1100, 116), (1350, 140),
              (1550, 150), (2000, 146), (2500, 136), (2900, 128), (3250, 120), (3500, 114), (4100, 112), (4250, 150),
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
                 (3600, 796), (3900, 745), (4250, 640)])
EGIM_X = Egri([(1500, 0.47), (2400, 0.50), (2900, 0.54), (3300, 0.64), (3650, 0.74), (4000, 0.80)])


def hw_kabin(x, z):
    """Kabin (cam) yan yüzeyi: içe eğim arkaya doğru artar (C direği içe yatar)."""
    return HW_KABIN(x) - (z - 1000.0) * EGIM_X(x)


# omuz çizgisi (fotoğraf eşlemesi): kapı boyunca ~860'ta yatay, kapı arkasından arka basene doğru yükselir
OMUZ = Egri([(400, 848), (1400, 860), (2300, 860), (2450, 870), (2650, 912), (2850, 952), (3050, 982),
             (3300, 1004), (3550, 1020), (3850, 1032), (4200, 1018), (4420, 1000)])


def omuz_z(x):
    return OMUZ(x)


# GTC "bıçak": üst keskin kenar kapının ön üst köşesinden aşağı kıvrılır, alt kapı boyunca ~600'de
# neredeyse yatay gider ve arka çamurluk kemerine doğru alçalarak söner; altında ~230 mm'lik içbükey bant.
BICAK_UST = Egri([(1380, 705), (1460, 680), (1560, 648), (1680, 622), (1850, 610), (2200, 607), (2600, 600),
                  (2850, 582), (3020, 562), (3170, 525), (3320, 470)])
BICAK_ON = Egri([(340, 1830), (400, 1700), (455, 1626), (560, 1522), (690, 1415), (740, 1385)])  # z -> ön sınır x


# ============================================================ detay şekilleri (gerçek mm)
ON_CAM = YPoligon([(1470, 1006), (1800, 1020), (2100, 1036), (2500, 1055), (2778, 1072), (2785, 1366),
                   (2600, 1372), (2380, 1352), (2200, 1308), (2000, 1250), (1800, 1162), (1640, 1086), (1540, 1040)], 20)
ARKA_YAN_CAM = YPoligon([(2810, 1082), (3010, 1098), (3240, 1134), (3410, 1174), (3520, 1199), (3578, 1208),
                         (3540, 1224), (3460, 1254), (3310, 1292), (3140, 1324), (2950, 1340), (2815, 1358)], 14)
FAR_ON = YPoligon([(384, 661), (430, 648), (522, 642), (652, 651), (739, 669), (790, 700), (815, 745),
                   (850, 800), (960, 846), (960, 856), (785, 845), (712, 830), (618, 806), (496, 762),
                   (417, 717)], 6)
FAR_YAN = YPoligon([(-60, 600), (200, 612), (330, 640), (420, 690), (490, 760), (535, 830), (552, 862),
                    (-60, 872)], 8)
_IZ_UST = [(0, 731), (150, 731), (260, 729), (330, 724), (362, 717)]
_IZ_ALT = [(338, 704), (290, 684), (230, 664), (150, 646), (70, 634), (0, 630)]
UST_IZGARA = YPoligon(_IZ_UST + _IZ_ALT[:-1] + [(-y, z) for y, z in reversed(_IZ_ALT[:-1])]
                      + [(-y, z) for y, z in reversed(_IZ_UST[1:])], 5)
# krom kanat çıtası ızgaranın üst kenarını oluşturur: rozette kalın (~50), uçlara doğru sivrilir
_KN_UST = [(0, 727), (150, 727), (260, 725), (330, 720), (356, 716)]
_KN_ALT = [(300, 705), (230, 694), (150, 684), (60, 679), (0, 678)]
KANAT = YPoligon(_KN_UST + _KN_ALT[:-1] + [(-y, z) for y, z in reversed(_KN_ALT[:-1])]
                 + [(-y, z) for y, z in reversed(_KN_UST[1:])], 3)
# OPC petek girişi: üst kenar siyah çıtanın alt kenarı ("gülümseme"), alta doğru daralan yamuk
ALT_IZGARA = YPoligon([(-520, 462), (-330, 448), (0, 432), (330, 448), (520, 462), (470, 385), (410, 314),
                       (200, 302), (0, 300), (-200, 302), (-410, 314), (-470, 385)], 14)
# plakanın oturduğu siyah tampon çıtası: köşe girişlerinin üst iç köşesine kadar uzanır
SIYAH_CITA = YPoligon([(-622, 510), (-533, 530), (-330, 507), (0, 500), (330, 507), (533, 530), (622, 510),
                       (572, 472), (520, 462), (330, 448), (0, 432), (-330, 448), (-520, 462), (-572, 472)], 5)
KROM_DLO = [(1478, 1022), (1560, 1060), (1640, 1102), (1800, 1178), (2000, 1268), (2200, 1324), (2380, 1366),
            (2600, 1385), (2800, 1381), (2950, 1352), (3140, 1338), (3310, 1305), (3460, 1267), (3545, 1236), (3596, 1212)]
YPoligon_AYNA = YPoligon([(1585, 990), (1718, 998), (1718, 1040)], 6)
# OPC köşe hava girişi: üçgen ağız + iç ve alt kenarını saran gümüş "bumerang" bıçak
KOSE_HAVA = YPoligon([(748, 471), (668, 356), (783, 360)], 9)
KOSE_BICAK = YPoligon([(744, 492), (630, 402), (576, 318), (814, 292), (786, 372), (758, 472)], 10)
# Arka stop (uzun bumerang): arkadan (y,z) görünüş ∩ yandan (x,z) görünüş. Ölçüler fotoğraf eşlemesinden
# (beyaz OPC tam arka + siyah OPC arka-üst çapraz): iç sivri uç y 325 z 838, üst kenar dışa doğru 950'ye
# yükselir, alt kenar ~800; dış uç köşeyi sarar, yanda yükselerek öne uzanır ve arka tekerin arka
# kenarı hizasında (x≈3950, z≈1000) C direği kırımının ucunda sivri biter.
STOP_ARKA = YPoligon([(318, 838), (380, 878), (450, 902), (530, 921), (620, 936), (690, 946), (735, 1020),
                      (1000, 1030), (1000, 778), (780, 780), (690, 785), (600, 795), (470, 806), (370, 820)], 9)
STOP_YAN = YPoligon([(3948, 1003), (4050, 993), (4200, 972), (4350, 958), (4600, 956), (4600, 772),
                     (4360, 778), (4280, 800), (4180, 850), (4080, 912), (3995, 975)], 8)
# OPC arka: siyah alt difüzör (tampon alt kenarı z≈380), uçlarında tek trapez egzoz (y 436..645, z 250..330)
DIFUZOR = YPoligon([(-720, 150), (720, 150), (712, 300), (690, 372), (-690, 372), (-712, 300)], 22)
EGZOZ = YPoligon([(436, 332), (645, 332), (648, 250), (458, 248)], 16)
REFLEKTOR = YPoligon([(362, 556), (575, 552), (572, 580), (380, 574)], 6)
KAPI_CIZGILERI = [
    [(1368, 250), (1372, 700), (1400, 960), (1450, 1004), (1640, 1112), (1800, 1194), (2000, 1287),
     (2200, 1340), (2400, 1386), (2600, 1398), (2800, 1394)],
    [(2800, 1394), (2797, 1080), (2792, 700), (2780, 480), (2750, 400), (2712, 330), (2640, 282)],
    [(2640, 282), (2440, 268), (2120, 256), (1800, 250), (1368, 250)],
    [(540, 818), (620, 735), (728, 652)],     # ön tampon - çamurluk (far arka alt ucundan teker kemerine)
    [(4300, 786), (4150, 676), (3948, 562)],  # arka tampon - çamurluk (stop alt köşesinden kemere)
]


# kaput ön kenarının ÖNDEN görünüşteki yüksekliği (fotoğraf ölçümü): ızgaranın ~100 mm üstünden geçer,
# y≈610'da farın üst kenarına iner, oradan far üst kenarını izleyerek dış üst köşeye (y≈785) çıkar
KAPUT_ON_Z = Egri([(0, 838), (200, 833), (366, 822), (530, 812), (610, 804), (680, 818), (740, 832), (785, 843)])


def kaput_poligon():
    """Kaput çizgisinin plan poligonu: her y için gövde yüzeyinin KAPUT_ON_Z(y) yüksekliğine ulaştığı x."""
    ys = np.array([0, 100, 200, 300, 366, 450, 530, 610, 650, 700, 740, 770, 785])
    xs = np.arange(0.0, 900.0, 1.0)
    on = []
    for yv in ys:
        d = govde_temel(xs[:, None, None], np.array([[[float(yv)]]]), np.array([[[float(KAPUT_ON_Z(yv))]]]))[:, 0, 0]
        i = int(np.argmax(d < 0))
        on.append((xs[i], float(yv)))
    sag = on + [(560.0, 792.0), (900.0, 760.0), (1330.0, 705.0)]
    return np.array([(x, y) for x, y in sag] + [(x, -y) for x, y in reversed(sag) if y > 0])


# arka daralma tablosu R(x, z) [mm, + = daha dar]: satırlar KD_X, sütunlar KD_Z; bikübik spline ile
# pürüzsüz yüzey. Değerler arkadan (beyaz OPC) ve arka-üst çaprazdan (siyah OPC) siluet eşlemesinden.
KD_X = [3000, 3300, 3630, 3900, 4100, 4300, 4500]
KD_Z = [150, 300, 450, 600, 700, 800, 900, 1000, 1060, 1150, 1300]
KD = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10],
    [0, 0, 0, -12, -10, 22, 45, 72, 75, 60, 60],
    [0, 0, 0, 5, 55, 115, 160, 170, 160, 130, 130],
    [0, 0, 8, 35, 95, 165, 215, 215, 200, 170, 170],
    [0, 0, 10, 30, 70, 160, 225, 215, 190, 160, 160],
    [0, 0, 10, 30, 70, 160, 225, 215, 190, 160, 160]], float)
_KD_SPL = None


def kuyruk_daralma(X, Z):
    global _KD_SPL
    if _KD_SPL is None:
        from scipy.interpolate import RectBivariateSpline
        _KD_SPL = RectBivariateSpline(KD_X, KD_Z, KD, kx=3, ky=3, s=0)
    Xb, Zb = np.broadcast_arrays(np.clip(X, KD_X[0], KD_X[-1]), np.clip(Z, KD_Z[0], KD_Z[-1]))
    return _KD_SPL.ev(Xb.ravel(), Zb.ravel()).reshape(Xb.shape) * sstep(3200, 3400, Xb)   # x<3200 dokunulmaz


# ============================================================ gövde
def govde_temel(X, Y, Z):
    hwx, dhwx = HW_PLAN(X), HW_PLAN.t(X)
    fz, dfz = F_DIKEY(Z), F_DIKEY.t(Z)
    hw = hwx * fz
    # omuz çizgisi
    zc = omuz_z(X)
    hw = hw + p(0.35) * np.maximum(0, 1 - np.abs(Z - zc) / p(1.7)) * sstep(500, 900, X) * (1 - sstep(4250, 4400, X))
    # GTC "bıçak": keskin üst kenar BICAK_UST, altında aşağı doğru sönen içbükey bant; ön sınırı çapraz kıvrım
    alt_b = BICAK_UST(X) - Z
    oy = 20.0 * sstep(0, 3.0, alt_b) * (1 - sstep(40.0, 235.0, alt_b))
    hw = hw - oy * sstep(-35.0, 45.0, X - BICAK_ON(Z)) * (1 - sstep(3080, 3330, X))
    # OPC marşpiyel
    hw = hw + p(0.35) * sstep(Z_ALT(X) + 8.0, Z_ALT(X) + 38.0, Z) * (1 - sstep(245, 275, Z)) * sstep(1300, 1420, X) * (1 - sstep(3150, 3280, X))
    # çamurluk kabarıklıkları (arka daha belirgin)
    for xa, a in ((AKS_ON, 0.6), (AKS_ARKA, 0.85)):
        r = np.sqrt((X - xa) ** 2 + (Z - AKS_Z) ** 2)
        hw = hw + p(a) * sstep(CAMUR_R, CAMUR_R + p(0.4), r) * (1 - sstep(CAMUR_R + p(1.2), CAMUR_R + p(3.8), r)) * sstep(150, 250, Z)
    # GTC arka basen (haunch): arka teker üstünde omuz bölgesi dışa şişer
    hw = hw + 32.0 * sstep(2950, 3550, X) * (1 - sstep(4000, 4400, X)) * sstep(700, 900, Z) * (1 - sstep(zc - 30.0, zc + 80.0, Z))
    # arka daralma (fotoğraf siluetlerinden): basenin üstü içe döner, teker arkasında gövde hızla
    # stop köşesine kaçar; stobun üstünde yüzey D direği tabanına dikleşerek akar
    hw = hw - kuyruk_daralma(X, Z)
    d_yan = (Y - hw) / np.sqrt(1 + (dhwx * fz) ** 2 + (hwx * dfz) ** 2)

    bombe = KAPUT_BOMBE(X)
    zu = Z_ALT_UST(X) - bombe * (Y / 900.0) ** 2
    kaput = sstep(120, 260, X) * (1 - sstep(1250, 1350, X))
    zu = zu + kaput * (p(1.1) * np.maximum(0, 1 - (Y / 560.0) ** 2) ** 2 + p(0.3) * np.maximum(0, 1 - np.abs(Y - 690) / 110.0))
    d_ust = (Z - zu) / np.sqrt(1 + Z_ALT_UST.t(X) ** 2 + (2 * bombe * Y / 900.0 ** 2) ** 2)
    # ön tampon alt kenarı köşelerde yükselir (köşe bıçaklarının altı z≈295)
    z_alt = Z_ALT(X) + 60.0 * sstep(470.0, 820.0, Y) * (1 - sstep(300.0, 540.0, X))
    d_alt = (z_alt - Z) / np.sqrt(1 + Z_ALT.t(X) ** 2)
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

    d = smin(alt, kabin, p(0.9) + 40.0 * sstep(3700, 4100, X))   # arkada D direği - basen geçişi yumuşak
    # OPC tavan spoyleri: tavanın devamı, arka kenarda ~55 mm yüksek yüz (3. stop burada), dışbükey arka
    # kenar, iki yanda D direği boyunca aşağı inen uç plakaları (fotoğraf: alt dudak z≈1275, uçlar |y|≈570)
    yy = np.minimum((Y / 560.0) ** 2, 1.0)
    sp_ust = 1356.0 - 0.12 * np.maximum(X - 3800.0, 0) - 26.0 * yy
    x_ak = 3992.0 + 26.0 * (1 - yy)
    kal = 30.0 + 26.0 * sstep(3890.0, 4000.0, X)
    sp = np.maximum(X - x_ak, 3790.0 - X)
    sp = smax(sp, Y - (hw_kabin(X, sp_ust) - 4.0), 15.0)
    sp = smax(sp, np.maximum(Z - sp_ust, (sp_ust - kal) - Z), 8.0)
    # uç plakaları: kabin yan yüzeyine oturan 35 mm'lik kabuk, arkaya doğru aşağı iner
    hk = hw_kabin(X, Z)
    z_alt = 1300.0 - 105.0 * sstep(3880.0, 3995.0, X)
    uc = smax(np.maximum(Y - (hk + 10.0), (hk - 26.0) - Y), np.maximum(z_alt - Z, Z - sp_ust), 6.0)
    uc = smax(uc, np.maximum(3860.0 - X, X - (x_ak - 6.0)), 8.0)
    sp = smin(sp, uc, 10.0)
    return smin(d, sp, p(0.5))


KAPUT = kaput_poligon()


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
    lastik = sd_kutu2(r, t, (JANT_DIS - 4 + TEKER_R) / 2, 0.0, (TEKER_R - JANT_DIS + 4) / 2, TEKER_YG, p(1.0))
    for tc in (-80.0, -28.0, 28.0, 80.0):
        kanal = np.maximum(np.abs(t - tc) - p(0.22), (TEKER_R - p(0.55)) - r)
        lastik = np.maximum(lastik, -kanal)
    s = np.mod(r * th, p(1.7)) - p(0.85)
    eg = s + 0.25 * (np.abs(t) - 95.0)
    enine = np.maximum(np.maximum(np.abs(eg) - p(0.18), (TEKER_R - p(0.5)) - r), 100.0 - np.abs(t))
    lastik = np.maximum(lastik, -enine)
    halka = np.maximum(np.abs(r - 312.0) - p(0.12), (TEKER_YG - p(0.25)) - t)
    lastik = np.maximum(lastik, -halka)

    # jant
    arka = np.maximum(np.maximum(r - (JANT_DIS - 4), -TEKER_YG - t), t + 15.0)
    dudak = sd_kutu2(r, t, JANT_DIS - 16.0, 92.0, 16.0, 22.0, 6.0)
    kovan = sd_kutu2(r, t, JANT_DIS - 24.0, 35.0, 18.0, 52.0, 3.0)
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
    kol = np.maximum(np.maximum(kol_ang, r - (JANT_DIS - 22.0)), 60.0 - r)
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
        # alt kenar ortada z 1030, köşelere doğru yükselir (|y| 560'ta ~1120)
        m = smax(smax(3950 - X, X - 4400, p(1.2)), Y - (hw_kabin(X, Z) - p(1.0)), p(0.9))
        alt = 1030.0 + 95.0 * sstep(280.0, 620.0, Y)
        return oyuk(dd, z0, smax(m, alt - Z, p(0.8)), CAM)
    uygula((3900, 4420, 0, 900, 1000, 1400), arka_cam)

    def yan_cam(dd, z0, X, Y, Z):
        m = np.minimum(ON_CAM(X, Z), ARKA_YAN_CAM(X, Z))
        return oyuk(dd, z0, np.maximum(m, (hw_kabin(X, Z) - p(1.2)) - Y), CAM)
    uygula((1440, 3620, 450, 900, 985, 1440), yan_cam)

    # GTC'nin imzası olan krom DLO kavisi: cam üst kenarı boyunca A direğinden arka cam ucuna ince kabarık şerit
    def krom(dd, z0, X, Y, Z):
        c = sd_cizgi(X, Z, KROM_DLO) - 6.5
        m = np.maximum(c, (hw_kabin(X, Z) - p(1.5)) - Y)
        return doldur(dd, z0, m, p(0.2))
    uygula((1450, 3640, 450, 900, 1000, 1440), krom)

    # ---------- kaput çizgisi (üstten)
    def kaput(dd, z0, X, Y, Z):
        s = sd_poligon(X, Y, KAPUT)
        m = np.maximum(np.maximum(np.abs(s) - p(0.22), 690 - Z), X - 1320)
        return oyuk(dd, z0, m, p(0.5))
    uygula((90, 1340, 0, 790, 680, 1020), kaput)

    # ---------- ön: üst ızgara (alt kenarı aşağı sarkan mercek/tekne şekli), krom kanat, rozet
    def ust_izgara(dd, z0, X, Y, Z):
        m = np.maximum(UST_IZGARA(Y, Z), X - 300)
        dd = oyuk(dd, z0, m, p(1.0))
        # kanat altındaki siyah ızgara: iki yatay çıta + alt kenarı izleyen ince krom şerit
        for zc in (652.0, 664.0):
            dd = doldur(dd, z0, np.maximum(np.abs(Z - (zc + 0.08 * np.abs(Y))) - p(0.14), m), -p(0.55))
        alt_krom = sd_cizgi(np.abs(Y), Z, [(0, 638), (150, 652), (240, 668), (320, 696)]) - p(0.16)
        dd = doldur(dd, z0, np.maximum(alt_krom, m), -p(0.3))
        # krom kanat çıtası: ızgaranın üst kenarı boyunca, yüzeye kadar dolu, ön yüzü hafif bombeli
        kn = np.maximum(KANAT(Y, Z), X - 300)
        dd = doldur(dd, z0, kn, -p(0.08) - p(0.12) * sstep(0, p(1.2), -KANAT(Y, Z)))
        rr = np.sqrt(Y ** 2 + (Z - 686) ** 2)
        dd = doldur(dd, z0, np.maximum(rr - 52, X - 300), -p(0.05))                     # rozet tabanı Ø104
        dd = doldur(dd, z0, np.maximum(np.abs(rr - 48) - p(0.2), X - 300), p(0.3))      # rozet halkası
        dd = oyuk(dd, z0, np.maximum(rr - (48 - p(0.2)), X - 300), p(0.1))              # halka içi (şimşek zemini)
        return dd
    uygula((-40, 300, 0, 500, 610, 760), ust_izgara)

    # ---------- farlar (bumerang): önden FAR_ON ∩ yandan FAR_YAN
    def farlar(dd, z0, X, Y, Z):
        fo = FAR_ON(Y, Z)
        m = np.maximum(fo, FAR_YAN(X, Z))
        dd = oyuk(dd, z0, m, p(0.45) + p(0.45) * sstep(660.0, 840.0, Z))
        # far camı kenar çerçevesi: dış hatta ince koyu çerçeve (gölge oluğu)
        dd = oyuk(dd, z0, np.maximum(-m - p(0.35), m), p(1.0))
        # iç (alt) uzun reflektör bölmesi
        ref = sd_cizgi(Y, Z, [(450, 670), (560, 668), (610, 676)]) - p(1.15)
        dd = oyuk(dd, z0, np.maximum(ref, m), p(1.0))
        # dış küçük bölme (sinyal/viraj)
        dis = sd_cizgi(Y, Z, [(752, 742), (790, 770)]) - p(0.9)
        dd = oyuk(dd, z0, np.maximum(dis, m), p(0.9))
        # üst kenar boyunca kanat DRL ışık kılavuzu (yükseltilmiş krom kaş)
        drl = sd_cizgi(Y, Z, [(410, 682), (500, 736), (618, 780), (712, 806), (780, 822), (850, 834)]) - p(0.28)
        dd = doldur(dd, z0, np.maximum(drl, m), -p(0.25))
        # projektör: krom halka + bombeli mercek
        mer = np.sqrt((Y - 672.0) ** 2 + (Z - 711.0) ** 2)
        dd = oyuk(dd, z0, np.maximum(mer - 46.0, m), p(1.3))
        dd = doldur(dd, z0, np.maximum(np.abs(mer - 42.0) - p(0.18), m), -p(0.3))
        dd = doldur(dd, z0, np.maximum(mer - 34.0, m), -p(0.45) - p(0.35) * (1 - (mer / 34.0) ** 2))
        return dd
    uygula((-20, 600, 360, 960, 600, 880), farlar)

    # ---------- ön tampon: siyah çıta + plaka, petek giriş, köşe girişleri (bıçaklı), dudak
    def on_tampon(dd, z0, X, Y, Z):
        on = X - 520
        # siyah çıta (ayrı parça): hafif içeride, kenarı keskin oluk
        ct = np.maximum(SIYAH_CITA(Y, Z), on)
        dd = oyuk(dd, z0, np.maximum(np.abs(SIYAH_CITA(Y, Z)) - p(0.12), on), p(0.8))
        dd = oyuk(dd, z0, ct, p(0.4) + p(0.25) * sstep(470, 520, Z))     # üst kenarda gölge veren basamak
        # plaka (520x110), çıtanın üstüne oturur
        plaka = sd_kutu2(Y, Z, 0, 507, 262, 56, 10)
        dd = oyuk(dd, z0, np.maximum(np.abs(plaka + p(0.15)) - p(0.15), on), p(0.35))
        dd = doldur(dd, z0, np.maximum(plaka + p(0.3), on), p(0.15))
        # petek giriş: derin yamuk, dış çerçevesinde ince dudak
        m = np.maximum(ALT_IZGARA(Y, Z), on)
        dd = oyuk(dd, z0, m, p(1.6))
        dd = doldur(dd, z0, np.maximum(petek(Y / 2.1, Z, p(0.75)) - p(0.13), m), -p(0.6))   # yatay uzun altıgenler
        # köşe girişleri: çevresi içbükey çukur (heykelsi köşe), gümüş bıçak (kabarık) + üçgen ağız + kanatçık
        kbd = KOSE_BICAK(Y, Z)
        dd = dd + p(0.45) * (1 - sstep(0.0, 85.0, kbd)) * (1 - sstep(470.0, 540.0, X))
        kb = np.maximum(kbd, on)
        dd = doldur(dd, z0, kb, p(0.3) + p(0.25) * sstep(0.0, p(1.0), -kbd))   # pahlı kenar, ortası daha kabarık
        dd = oyuk(dd, z0, np.maximum(np.abs(KOSE_BICAK(Y, Z) + p(0.1)) - p(0.1), on), p(0.2))
        k = np.maximum(KOSE_HAVA(Y, Z), on)
        dd = oyuk(dd, z0, k, p(1.6))
        fin = np.maximum(np.abs(Y - 728.0 + 0.3 * (Z - 400.0)) - p(0.2), k)
        dd = doldur(dd, z0, fin, -p(0.5))
        # far yıkama kapağı ve çeki kancası kapağı (ince panel çizgileri)
        def elips(u, v, a, b):
            k0 = np.sqrt((u / a) ** 2 + (v / b) ** 2)
            k1 = np.sqrt((u / a ** 2) ** 2 + (v / b ** 2) ** 2)
            return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)
        ca, sa = math.cos(0.23), math.sin(0.23)
        u, v = (Y - 693.0) * ca + (Z - 612.0) * sa, -(Y - 693.0) * sa + (Z - 612.0) * ca
        dd = oyuk(dd, z0, np.maximum(np.abs(elips(u, v, 62.0, 22.0)) - p(0.11), on), p(0.25))
        u, v = (Y - 470.0) * math.cos(0.3) + (Z - 535.0) * math.sin(0.3), -(Y - 470.0) * math.sin(0.3) + (Z - 535.0) * math.cos(0.3)
        ck = sd_kutu2(u, v, 0.0, 0.0, 44.0, 13.0, 9.0)
        dd = oyuk(dd, z0, np.maximum(np.abs(ck) - p(0.11), on), p(0.25))
        # splitter: gövde renkli dudağın altında öne taşan keskin plaka, uçları köşe bıçaklarına kıvrılır
        dudak = np.maximum(sd_cizgi(np.abs(Y), Z, [(0, 226), (420, 228), (600, 240), (760, 262), (830, 282)]) - p(0.4), X - 420)
        dd = doldur(dd, z0, dudak, p(0.7))
        return dd
    uygula((-60, 560, 0, 960, 150, 620), on_tampon)

    # ---------- yan: kapı aralıkları, kol
    def kapilar(dd, z0, X, Y, Z):
        m = None
        for c in KAPI_CIZGILERI:
            s = sd_cizgi(X, Z, c)
            m = s if m is None else np.minimum(m, s)
        return oyuk(dd, z0, np.maximum(m - p(0.22), 480 - Y), p(0.5))
    uygula((300, 4200, 480, 960, 250, 1480), kapilar)

    def kol(dd, z0, X, Y, Z):
        xc, zc = 2560.0, 838.0
        cep = np.sqrt(((X - xc) / 85) ** 2 + ((Z - zc) / 28) ** 2) - 1.0
        dd = oyuk(dd, z0, np.maximum(cep * 28, 700 - Y), p(0.55))
        k = sd_kutu2(X, Z, xc, zc + 4, 70, 12, 11)
        return doldur(dd, z0, np.maximum(k, 700 - Y), p(0.3))
    uygula((2450, 2670, 700, 960, 780, 920), kol)

    # ---------- arka
    def arka(dd, z0, X, Y, Z):
        ay = np.abs(Y)
        # bagaj kapağı aralığı: plaka altında "gülümseme" yayı (orta z 712), stobun alt kenarına çıkar,
        # stobun içinden (iç/dış parça ayrımı) yukarı geçer, cam köşesine ve D direği boyunca spoylere çıkar
        bagaj = sd_cizgi(Y, Z, [(0, 712), (120, 716), (240, 727), (330, 739), (400, 750), (480, 771),
                                (548, 796), (540, 860), (530, 921)])
        dd = oyuk(dd, z0, np.maximum(bagaj - p(0.22), 4200 - X), p(0.6))
        # rozet: halka Ø~120, merkez z 876 (stoplar arasında, gülümsemenin üstünde)
        rr = np.sqrt(Y ** 2 + (Z - 876) ** 2)
        dd = doldur(dd, z0, np.maximum(np.abs(rr - 55) - p(0.3), 4300 - X), p(0.3))
        dd = doldur(dd, z0, np.maximum(rr - 46, 4300 - X), p(0.12))
        # plaka (520x110) tamponda, z 480..590; arkasında hafif yuva
        yuva = np.maximum(sd_kutu2(Y, Z, 0, 555, 170, 50, 14), 4300 - X)
        dd = oyuk(dd, z0, yuva, p(0.25))
        plaka = np.maximum(sd_kutu2(Y, Z, 0, 535, 260, 55, 8), 4300 - X)
        dd = doldur(dd, z0, plaka, p(0.25))
        dd = oyuk(dd, z0, np.maximum(np.abs(plaka + p(0.3)) - p(0.12), 4300 - X), p(0.15))   # plaka kenar çizgisi
        # reflektörler (tampon, plaka hizası)
        ref = np.maximum(REFLEKTOR(ay, Z), 4250 - X)
        dd = oyuk(dd, z0, ref, p(0.3))
        dd = doldur(dd, z0, np.maximum(ref + p(0.12), 4250 - X), -p(0.12))
        # park sensörleri
        for yc in (255.0, 610.0):
            sn = np.sqrt((ay - yc) ** 2 + (Z - 472) ** 2) - 11.0
            dd = oyuk(dd, z0, np.maximum(sn, 4250 - X), p(0.15))
        # OPC difüzör: tampon alt kenarında basamak, içinde dikey kanatlar; uçlarda trapez egzozlar
        dif = np.maximum(DIFUZOR(Y, Z), 4250 - X)
        dd = oyuk(dd, z0, dif, p(0.7))
        for yc in (0.0, 150.0, 300.0):
            dd = doldur(dd, z0, np.maximum(np.abs(ay - yc) - p(0.3), np.maximum(dif, Z - 345)), -p(0.1))   # 0.6 mm kanat
        eg = np.maximum(EGZOZ(ay, Z), 4250 - X)
        dd = doldur(dd, z0, np.maximum(np.abs(eg + p(0.28)) - p(0.28), 4250 - X), p(0.55))  # krom çerçeve
        dd = oyuk(dd, z0, eg + p(0.55), p(3.0))
        return dd
    uygula((4100, 4520, 0, 960, 130, 1150), arka)

    def stoplar(dd, z0, X, Y, Z):
        # arka (y,z) ∩ yan (x,z) -> köşeyi saran bumerang; ince çerçeve dudağı + çukur
        st = np.maximum(STOP_ARKA(Y, Z), STOP_YAN(X, Z))
        dd = doldur(dd, z0, np.abs(st + p(0.2)) - p(0.2), p(0.15))
        dd = oyuk(dd, z0, st, p(0.55))
        # LED ışık kılavuzları: içte yatay çubuk, dışta "Γ" (üst yatay + dış dikey kol), yanda üst şerit
        g = np.minimum(sd_cizgi(Y, Z, [(392, 884), (505, 902)]),
                       sd_cizgi(Y, Z, [(540, 910), (672, 921), (660, 846)])) - p(0.22)
        dd = doldur(dd, z0, np.maximum(np.maximum(g, st), 4250 - X), -p(0.2))
        g2 = sd_cizgi(X, Z, [(4010, 982), (4200, 950), (4400, 935)]) - p(0.22)
        dd = doldur(dd, z0, np.maximum(np.maximum(g2, st), 640 - Y), -p(0.2))
        return dd
    uygula((3920, 4520, 300, 960, 740, 1060), stoplar)

    # ---------- 3. stop lambası: spoyler arka yüzünde ince yatay şerit
    def ucuncu_stop(dd, z0, X, Y, Z):
        m = np.maximum(sd_kutu2(Y, Z, 0, 1300, 150, 7, 5), 3990 - X)
        return oyuk(dd, z0, m, p(0.25))
    uygula((3960, 4060, 0, 200, 1270, 1340), ucuncu_stop)

    # ---------- köpekbalığı anten
    def anten(dd, z0, X, Y, Z):
        return smin(dd, sd_elipsoid(X, Y, Z, (3560.0, 0.0, 1395.0), (p(3.6), p(0.85), p(2.5))), p(0.5))
    uygula((3450, 3680, 0, 60, 1320, 1470), anten)

    # ---------- aynalar: kapıya bağlı ince ayak + damla planlı kapak (önü bombeli, arkası düz cam yüzü)
    def ayna(dd, z0, X, Y, Z):
        # kapak: yuvarlatılmış kutu (üstü düz, altı içe doğru yükselir) + dış uçta bombeli burun -> damla
        def kapak(Xq):
            ana = sd_kutu3(Xq, Y, Z, (1662.0, 922.0, 1060.0), (58.0, 88.0, 68.0), 46.0)    # dış: derin, yüksek (~136 mm)
            ic = sd_kutu3(Xq, Y, Z, (1682.0, 846.0, 1054.0), (36.0, 42.0, 46.0), 28.0)     # iç: sığ, alçak
            return smin(ana, ic, 26.0)
        g = smax(kapak(X), X - 1708.0, 5.0)                                            # arka cam yüzü düz
        g = smax(g, 800.0 - Y, 6.0)
        kol = sd_kutu3(X, Y, Z, (1688.0, 800.0, 1012.0), (24.0, 52.0, 11.0), 8.0)       # ince yatay kol (bıçak)
        g = smin(g, kol, 10.0)
        cam = np.maximum(kapak(np.full_like(X, 1702.0)) + p(0.45), 1694.0 - X)
        g = oyuk(g, g.copy(), cam, p(0.45))
        sin = np.maximum(sd_cizgi(X, Z, [(1618, 1040), (1690, 1032)]) - p(0.2), 975 - Y)
        g = oyuk(g, g.copy(), sin, p(0.25))                                            # sinyal çizgisi
        dd = smin(dd, g, p(0.3))
        # kapı köşesindeki üçgen ayna tabanı: gövdeye yapışık, 0.3 mm kabarık
        yelken = np.maximum(YPoligon_AYNA(X, Z), 720.0 - Y)
        return doldur(dd, z0, yelken, p(0.3))
    uygula((1505, 1770, 660, 1060, 920, 1170), ayna)
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

    # Opel şimşeği (rozet halkalarının içinde). Şekil ayna simetrik değil, nokta simetriktir;
    # bu yüzden yarım hacimde değil burada işlenir. u: bakana göre sağa, v: yukarı (R birimi).
    simsek = np.array([(-1.15, 0.02), (0.02, 0.30), (0.20, 0.06), (1.15, -0.02), (-0.02, -0.30), (-0.20, -0.06)])

    def logo(sl, U, V, R):
        m = sd_poligon(U / R, V / R, simsek) * R - p(0.05)
        cur = d[sl]
        d[sl] = np.minimum(cur, np.maximum(cur - p(0.3), m))

    g = kes(-40, 300, -80, 80, 640, 740)      # ön rozet: merkez (y 0, z 688), R 50; önden bakınca sağ = -Y
    if g:
        sl, X, Y, Z = g
        logo(sl, -Y + 0 * X, Z - 688.0, 50.0)
    g = kes(4300, 4520, -80, 80, 800, 950)     # arka rozet: merkez (y 0, z 876), R 48; arkadan bakınca sağ = +Y
    if g:
        sl, X, Y, Z = g
        logo(sl, Y + 0 * X, Z - 876.0, 48.0)
    # arka silecek: göbek camın alt kenarında ortanın hemen sağında, süpürge sola (-Y) yatık
    g = kes(4200, 4440, -420, 120, 1010, 1110)
    if g:
        sl, X, Y, Z = g
        sil = sd_cizgi(Y, Z, [(-345, 1052), (30, 1060)]) - p(0.3)
        gob = np.sqrt((Y - 45.0) ** 2 + (Z - 1056.0) ** 2) - p(0.75)
        cur = d[sl]
        d[sl] = np.minimum(cur, np.maximum(cur - p(0.45), np.minimum(sil, gob)))

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
        sil = np.minimum(sd_cizgi(X, Y, [(1450, 630), (1418, 40)]),
                         sd_cizgi(X, Y, [(1445, -20), (1414, -620)])) - p(0.3)
        kol = np.minimum(sd_cizgi(X, Y, [(1474, 590), (1440, 330), (1430, 300)]),
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
