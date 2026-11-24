#!/usr/bin/env python3
"""
Opel Astra J GTC (2011–2018, 3 kapı, OPC donanım) — 3D baskı maketi üretici.

Tüm geometri GERÇEK aracın milimetre ölçüleriyle tanımlanır, baskı ölçeğine en sonda küçültülür.
    boy 4466, genişlik 1840 (aynalarla 2020), yükseklik 1482, aks mesafesi 2695,
    iz genişliği ön 1584 / arka 1588, yerden yükseklik ~112, lastik 245/35 R20 (OPC).
Gövde işaretli mesafe alanı (SDF) ile modellenir ve marching cubes ile pürüzsüz yüzeye çevrilir.

Çıktılar:
    astra_gtc_tek_parca.stl   tekerlekler gövdeye bağlı, tek parça
    astra_gtc_govde.stl       tekerleksiz gövde (alt yüzeyi düz, aks delikli)  ┐
    astra_gtc_teker_on.stl    ön tekerlek, Brembo fren (2 adet basılır)        │ kit
    astra_gtc_teker_arka.stl  arka tekerlek, küçük disk/kaliper (2 adet)       ┘

Kullanım:
    pip install numpy scipy scikit-image trimesh manifold3d
    python3 astra_gtc_olustur.py                    # 1:24, ızgara 0.15 mm, hepsi
    python3 astra_gtc_olustur.py --olcek 18         # 1:18 (≈ 248 mm)
    python3 astra_gtc_olustur.py --parca tek --h 0.12
    python3 astra_gtc_olustur.py --plakasiz         # plakasız: yuvadaki kabartma "GTC" görünür

Ön ve arka rozetlerde Opel şimşeği kabartma olarak bulunur. Arkada fotoğraflardaki yerinde kabartma "opc"
rozeti; plakasız seçenekte plaka yuvasında kabartma "GTC" yazısı (başka marka/model yazısı eklenmez).
"""
import argparse
import gc
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
KAPUT_BOMBE = Egri([(0, 0), (250, 8), (600, 22), (1000, 35), (1300, 40), (1450, 10), (3950, 10), (4120, 45), (4280, 95), (4420, 110)])
# arkada bagaj kapağı üst yüzü iki yana doğru alçalır: D direği kırımı cam köşesinden stobun dış üst köşesine iner
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
# ön tampon splitter dudağının ALT (keskin) kenarı z(|y|): kalibre ön çapraz fotoğrafta (15922355098) tamponun en
# alttaki öne bakan parlak kenarı, her y için yüzeye geri izdüşümle okundu (ortada ~230-238, ızgara köşesi altında
# ~225'e hafifçe iner, köşeye doğru ~260'a yükselir). Gövdenin alt yüzü de (govde_temel z_alt) bu kenarın hemen
# altına çekilir: dudak en alttaki öne bakan kenardır, altında ikinci bir "çene" kalmaz. Köşeyi dönüp yanda söner.
ON_DUDAK_ALT = Egri([(0, 229), (200, 231), (320, 231), (430, 226), (500, 225), (600, 230), (690, 238), (780, 248),
                     (830, 257), (870, 262), (920, 262)])
ARKA_PLAN = Egri([(0, 0), (200, 5), (400, 20), (550, 50), (680, 100), (780, 175), (850, 270), (895, 380),
                  (915, 480), (925, 560)])
ARKA_YAN = Egri([(100, 95), (150, 62), (200, 35), (250, 18), (300, 8), (400, 2), (450, 0), (700, 0),
                 (800, 4), (880, 13), (960, 30), (1015, 55), (1050, 90), (1100, 170)])
Z_KABIN = Egri([(900, 600), (1150, 880), (1300, 975), (1400, 1030), (1600, 1145), (1800, 1255),
                (2000, 1350), (2150, 1410), (2300, 1447), (2450, 1466), (2600, 1472), (2800, 1466),
                (3100, 1444), (3400, 1412), (3650, 1377), (3800, 1352), (3880, 1335), (3940, 1310),
                (4020, 1262), (4120, 1196), (4220, 1128), (4300, 1075), (4360, 1030), (4420, 900)])
# arkada D direği tabanı daralır, üstü (spoyler hizası z≈1350) aynı kalır: beyaz OPC tam arka ve siyah OPC
# çapraz siluetlerinde stobun üstünde D direğine geçen belirgin bir omuz/raf var (direk tabanı |y|≈560-640)
HW_KABIN = Egri([(1100, 665), (1350, 752), (1700, 788), (2300, 794), (2800, 795), (3200, 797),
                 (3600, 796), (3800, 735), (3950, 690), (4100, 604), (4250, 548)])
EGIM_X = Egri([(1500, 0.47), (2400, 0.50), (2900, 0.54), (3300, 0.64), (3650, 0.74), (3800, 0.70), (3950, 0.67),
               (4100, 0.55), (4250, 0.54)])


SP_TASMA = 40.0          # OPC tavan spoyleri uçlarının kabin yan yüzeyinden dışa taşması (mm)


def hw_kabin(x, z):
    """Kabin (cam) yan yüzeyi: içe eğim arkaya doğru artar (C direği içe yatar)."""
    return HW_KABIN(x) - (z - 1000.0) * EGIM_X(x) - ust_duzeltme(x, z)


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
# OPC marşpiyel: ~10 mm dışa basamaklı, üst kenarı keskin kırım. Kapı boyunca kırım kapının alt kenarıdır
# (siyah OPC 34848925543 yan fotoğrafında parlak çizgi kapı alt kenarını izler, arka alt köşede kapı arka
# kenarına kıvrılır); kapının arkasında arka çamurlukta daha yüksekten başlayıp arka teker kemerine yükselir:
# z = 228 + 0.077 (x - 1290) (aynı fotoğraf + gri 54188905081). İki kısım kapı arka kenarında (aralıkta) birleşir.
ESIK_KAPI = Egri([(1250, 248), (1368, 250), (1800, 250), (2120, 256), (2440, 268), (2640, 282), (2720, 288)])
KAPI_W = p(0.30)   # kapı aralığı oluğunun yarı genişliği (kenarı yüzeye teğet iner)
# KAPI_ARKA_X (z -> kapı arka kenarı x) kapı çizgileriyle aynı eğriden türetilir: KAPI_CIZGILERI'nin altında


def esik_basamak(X, Z):
    """Marşpiyel basamağının (10 mm dışa) yükseklik çarpanı: kapı altında kırım ESIK_KAPI, arka çamurlukta daha yüksek
    'arka' kırım. Kapı arka aralığında iki kırımın basamak çarpanları x'te karıştırılır (kırım yüksekliği değil): arka
    çamurluğun basamak ön yüzü oluğun arka kenarından ~18 mm'de eğik iner. Eskiden kırım yüksekliği karıştırıldığından
    ön yüz x'te dikey bir duvardı; oluk izdüşümü bu duvarın köşesinde kopuk kıymıklar ve tırtıklı kenar bırakıyordu
    (x 2661 z 283, x 2729 z 337)."""
    arka = 228.0 + 0.077 * (np.clip(X, 1290.0, 3240.0) - 1290.0)
    zk = ESIK_KAPI(X)
    t = sstep(1.0 * KAPI_W, 3.5 * KAPI_W, X - KAPI_ARKA_X(Z))
    return (1.0 - t) * (1.0 - sstep(zk - 8.0, zk, Z)) + t * (1.0 - sstep(arka - 8.0, arka, Z))


# ============================================================ detay şekilleri (gerçek mm)
# kapı camının arka kenarı eğik: altta 2692, üstte 2745 (gri ve siyah OPC yan fotoğrafları); arkasında kapıya bağlı
# siyah B direği kaplaması (gövde yüzeyinde, camdan 0.45 mm kabarık), 2805-2820'de arka yan cam önünde krom çıta
# (arka yan cam ön kenarı 2818-2821: cam pahı çıtanın arka yamacından başlar, dik duvar kalmaz)
# Camların üst kenarı ayrı poligonla değil doğrudan krom DLO şeridinin alt kenarıyla tanımlanır (KROM_YOL): eski
# poligonların üst kenarı kromun 4-16 mm altından geçiyordu (krom ile cam arasında gövde renkli, paralel olmayan
# şerit), köşe yuvarlatması (r 20/14) ise ön üçgen camın ucunu kromun başlangıcından 70 mm, arka yan camın ucunu
# krom ucundan 56 mm geride bitiriyordu. Burada yalnız alt kenarlar (x -> z) ve kapı camının eğik arka kenarı var;
# ön üçgen camın sivri ucu kromun uçtaki kıvrımının içinde biter.
# Bel (cam alt kenarı) önde alçak: siyah OPC 34848925543 ve gri 54188905081 kalibre kameralarında ön kapı önünde
# z ≈ 905-945, B direğinde ≈ 1045-1075 (eski hat önde 1008-1019'du: 60-90 mm yüksek). Ön üçgen camın ucu A direği
# kromunun düz uzantısında x ≈ 1330, z ≈ 930'a iner; ön kapı çizgisi cam alt kenarına üçgen camın ortasının altında
# değer (eski uç 1488'de, kapı çizgisi ucun hemen arkasındaydı). Gövde değişmez: cam önde A direği dibinin yuvarlak
# yüzeyine oturur (gövde üstünü bele kadar indirmek kapı omzunda dalga bırakıyordu).
ON_CAM_ALT = Egri([(1320, 926.0), (1385, 926.0), (1430, 927), (1500, 932), (1700, 958), (2000, 993),
                   (2300, 1023), (2500, 1043), (2692, 1062), (2780, 1071)])
ON_CAM_ARKA = ((2692.0, 1068.0), (2745.0, 1366.0))      # kapı camı arka kenarı (B direği kaplamasının önü)
ARKA_CAM_ALT = Egri([(2800, 1080), (2818, 1082), (3010, 1098), (3240, 1134), (3410, 1174), (3520, 1199),
                     (3578, 1208), (3620, 1214)])
ARKA_CAM_ON_X = 2821.0                                    # arka yan cam ön kenarı (B direği krom çıtasının arkası)
FAR_ON = YPoligon([(384, 661), (430, 648), (522, 642), (652, 651), (739, 669), (790, 700), (815, 745),
                   (850, 800), (960, 846), (960, 856), (785, 845), (712, 830), (618, 806), (496, 762),
                   (417, 717)], 6)
FAR_YAN = YPoligon([(-60, 600), (200, 612), (330, 640), (420, 690), (490, 760), (535, 830), (552, 862),
                    (-60, 872)], 8)
# Far iç detayları "far haritası" (u, z) düzleminde tanımlıdır: u = y·cos32° + x·sin32°, yani far, önden
# 32° dışa dönük bakıştan (15922355098 fotoğrafının kalibre kamerası) dik izdüşümle görülür; bu bakışta
# önden köşeye saran farın tamamı tek parça, bozulmadan görünür. Bütün ölçüler fotoğraf o haritaya
# düzleştirilerek (kalibre kamera + gövde yüzeyi) okundu. Maskeler bu yönde uzanan prizmalardır.
FAR_TH = math.radians(32.0)


def far_u(X, Y):
    return Y * math.cos(FAR_TH) + X * math.sin(FAR_TH)


# cam (dış kapak) hattı: iç sivri uç u 382, alt kenar projektör altında aşağı "kıvrılır" (z 641),
# dış uç kaput-çamurluk aralığına değen yuvarlak sivri uç (u 932, z 842)
FAR_UZ = YPoligon([(382, 665), (392, 661), (440, 660), (505, 660), (570, 661), (603, 658), (623, 654),
                   (638, 648), (655, 643), (680, 641), (705, 640), (750, 645), (788, 655), (815, 665),
                   (842, 684), (862, 705), (881, 731), (895, 757), (908, 787), (921, 813), (933, 842),
                   (900, 842), (850, 837), (800, 830), (750, 822), (700, 812), (670, 804), (637, 795),
                   (603, 785), (570, 777), (537, 767), (503, 757), (470, 745), (437, 732), (410, 713),
                   (392, 688)], 4)
# üst yansıtıcı: projektör bölmesinin üstünde sivrilen bant + dış uçta aşağı inen üçgen (sinyal/park)
FAR_UST = YPoligon([(512, 756), (570, 769), (670, 787), (750, 798), (850, 812), (884, 816), (888, 810),
                    (878, 785), (865, 757), (851, 736), (838, 720), (827, 713), (820, 716), (819, 750),
                    (816, 784), (755, 780), (705, 772), (655, 764), (600, 757), (560, 754)], 3)
# derin iç hazne (camın arkası): kaşın altı, alt çerçevenin üstü; projektör altında aşağı sarkar
FAR_HAZNE = YPoligon([(398, 676), (420, 686), (600, 687), (626, 678), (646, 665), (680, 657), (740, 656),
                      (785, 664), (806, 678), (815, 700), (816, 745), (813, 781), (755, 776), (705, 768),
                      (655, 759), (600, 752), (550, 747), (500, 740), (452, 729), (418, 708)], 4)
# projektör altındaki noktalı alt yansıtıcı
FAR_ALT_REF = YPoligon([(628, 676), (648, 664), (690, 657), (745, 657), (790, 667), (800, 680), (760, 672),
                        (700, 671), (660, 676), (640, 684)], 3)
# far yıkama kapağı hattı (far haritası u, z): fotoğraftan geri izdüşüm, iç üst köşe pahlanıp yuvarlatıldı
FAR_YIKAMA = YPoligon([(795.0, 637.5), (696.5, 625.2), (660.0, 606.0), (623.5, 564.5), (750.0, 577.8),
                       (767.7, 589.5)], 9)
FAR_PROJ = (716.0, 718.0)   # projektör merkezi (u, z)
FAR_UZUN = (630.0, 722.0)   # uzun far çanağı merkezi
_IZ_UST = [(0, 731), (150, 731), (260, 729), (330, 724), (362, 717)]
# alt kenar (fotoğraf eşlemesi): uçta kanadın ucuna dik kıvrılır, sonra dolgun "tekne" gibi yatay iner
_IZ_ALT = [(352, 707), (342, 692), (322, 675), (292, 663), (230, 650), (150, 640), (70, 633), (0, 630)]
UST_IZGARA = YPoligon(_IZ_UST + _IZ_ALT[:-1] + [(-y, z) for y, z in reversed(_IZ_ALT[:-1])]
                      + [(-y, z) for y, z in reversed(_IZ_UST[1:])], 5)
# krom kanat çıtası ızgaranın üst kenarını oluşturur: rozette kalın (~50), uçlara doğru sivrilir
_KN_UST = [(0, 724), (150, 724), (260, 722), (330, 717), (354, 714)]   # üstte 0.3 mm siyah aralık kalır
_KN_ALT = [(300, 705), (230, 694), (150, 684), (60, 679), (0, 678)]
KANAT = YPoligon(_KN_UST + _KN_ALT[:-1] + [(-y, z) for y, z in reversed(_KN_ALT[:-1])]
                 + [(-y, z) for y, z in reversed(_KN_UST[1:])], 3)
# kanadın ön yüzündeki keskin sırt (üst yüz açık, alt yüz gölgede kalan kırım; fotoğraftaki parlak bant sınırı)
KANAT_SIRT = [(0, 701), (150, 705), (260, 709), (330, 713), (350, 714)]
# kanadın altındaki siyah yarığın altında ince krom şerit: rozetten dışa, uçta yukarı kıvrılıp kanadın ucuna bağlanır
IZ_ALT_KROM = [(0, 661), (100, 662), (200, 666), (260, 672), (300, 681), (330, 694), (346, 708)]
# OPC petek girişi: üst kenar siyah çıtanın alt kenarı ("gülümseme"), alta doğru daralan yamuk
ALT_IZGARA = YPoligon([(-520, 462), (-330, 448), (0, 432), (330, 448), (520, 462), (470, 385), (410, 314),
                       (200, 302), (0, 300), (-200, 302), (-410, 314), (-470, 385)], 14)
# plakanın oturduğu siyah tampon çıtası: köşe girişlerinin üst iç köşesine kadar uzanır
# (dış ucu iki kalibre fotoğraftan: üst kenar düz z≈505, alt kenar gövde rengi sırtın kenarı; köşe girişinin
# tepe noktasında sivri kama olarak biter)
_CITA_UST = [(0, 500), (330, 507), (480, 507), (600, 504), (690, 499), (724, 491)]
_CITA_ALT = [(718, 487), (672, 485), (620, 480), (568, 468), (520, 462), (330, 448), (0, 432)]
SIYAH_CITA = YPoligon([(-y, z) for y, z in reversed(_CITA_UST[1:])] + _CITA_UST + _CITA_ALT
                      + [(-y, z) for y, z in reversed(_CITA_ALT[:-1])], 3)


def yumusat(pts, adim=15.0):
    """Kontrol noktalarından geçen Catmull-Rom eğrisini ~'adim' mm aralıklı sık çoklu çizgiye çevirir. Kırık
    çizgi köşeleri oluk/şerit kenarlarında görünür dirsek bırakıyordu."""
    P = np.asarray(pts, float)
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = [P[0]]
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        n = max(1, int(math.ceil(np.hypot(*(p2 - p1)) / adim)))
        for t in np.arange(1, n + 1) / n:
            out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    return np.array(out)


# GTC krom DLO kavisi (merkez çizgisi). Ön uç = ön üçgen camın ucu (A direği dibi), arka uç = arka yan camın ucu;
# ön uç fotoğraflardaki gibi A direği kromunun düz (uca doğru hafif dikleşen) uzantısında x ≈ 1333, z ≈ 932'ye iner
# (eski uç 1488/1026'da kütle bitiyordu); yuvarlak ucu cam alt kenarına (z 926) değer, camın sivri ucu ucun hemen
# arkasında biter. krom_isaretli() x'e göre artan kısmı (en ön noktadan en arka noktaya) kullanır.
# kontrol noktaları eğim sırası düzgün olacak şekilde yumuşatıldı (eski 1640/2950 noktaları kavisi dalgalandırıyordu)
# Alt uç (x 1333-1640) yandan değil 3B'de düzgün: krom gövde yüzeyinde yürür ve A direği dibinde direğin yuvarlak
# köşesinden gövde omzuna geçerken yüzeyin y'si sıçrar (eski yolda x 1560->1488 y 727->722 geri, sonra 1488->1408
# 722->777 ani dışa): çapraz bakışta (önden 30-60°) alt uç S yapıyordu. Noktalar, yüzeydeki 3B eğrinin eğriliği
# düzgün değişecek şekilde (3. farkların en küçüklenmesi) z'de en çok 15 mm (x≈1483) aşağı kaydırıldı: y artık
# 723 -> 829 tek yönde, ucuna doğru giderek dışa döner. Uç noktası (1333, 932) ve x ≥ 1640 değişmedi. Kalibre ön
# çapraz kamera (15922355098) ve siyah OPC yan (34848925543) eski yolun ortasını uca göre 4-17 mm yüksek gösteriyordu.
KROM_DLO = [(1333, 932), (1358, 950.7), (1383, 966.3), (1408, 978.9), (1433, 988.8), (1458, 997.7), (1483, 1008.3),
            (1508, 1022.1), (1533, 1037.6), (1558, 1052.9), (1583, 1067.9), (1608, 1082.1),
            (1640, 1099), (1800, 1178), (2000, 1268),
            (2200, 1324), (2380, 1366),
            (2600, 1385), (2800, 1381), (2950, 1358), (3140, 1334), (3310, 1303), (3460, 1266), (3545, 1236),
            (3578, 1220), (3589.5, 1214.5), (3592, 1209), (3590, 1204), (3585, 1201), (3570, 1198.5), (3558, 1196)]
# arka uç: krom çıta arka yan camın sivri ucunu (x≈3584, z≈1209) camdan 8 mm uzaklıkta sarar ve camın alt kenarı
# boyunca ~30 mm gidip biter (mavi OPC 26608068581). Eskiden camın ucunun ~12 mm ötesinde (3596, 1212) yuvarlak
# küt uçla C direği üstünde bitiyordu. krom_isaretli() x'e göre artan kısmı (ucun en arka noktasına kadar) kullanır;
# geri dönen kuyruk cam alt kenarını da kromla sınırlar.
KROM_YOL = yumusat(KROM_DLO, 40.0)
# köpekbalığı anten yan profili (x, tavandan yükseklik)
ANTEN_PROFIL = YPoligon([(3576, -10), (3577, 3), (3585, 10), (3600, 15), (3625, 18), (3648, 21), (3668, 33),
                         (3688, 46), (3706, 58), (3716, 65), (3723, 66), (3727, 60), (3728, 40), (3731, 15),
                         (3734, -10)], 4)
# ayna ayağının oturduğu siyah taban plakası: ön üçgen camı bölen dikey çıtanın (x 1700-1715) dibinde, kapı
# camının ön alt köşesinde, bel çizgisi üzerinde (fotoğraflar: 26608054501 mavi OPC, 54188905081 gri)
YPoligon_AYNA = YPoligon([(1536, 906), (1697, 906), (1684, 990), (1553, 990)], 5)
# sabit ön üçgen cam ayracı (x aralığı): kemer çizgisinden (taban plakası) A direği kromuna düz siyah çıta; ön
# kenarı fotoğraflarda kapı camı kanalının ~25-30 mm önünde (eski 15 mm'lik çıta ince çubuk gibi okunuyordu)
# Siyah OPC (önden) ayracın ön kenarını, gri (arkadan) arka kenarını gösterir: ayraç aynanın dibinde geniş (x 1533-1700),
# A direği kromuna doğru daralan (1605-1660) düz siyah yelkendir; eski 1688-1715 çıta yalnız arka kenara denk geliyordu
AYNA_CITA = YPoligon([(1530, 900), (1702, 900), (1659, 1150), (1608, 1150)], 4)
# yan sinyal (ön çamurluk, kapı çizgisinin ~15 mm önünde): yuvarlatılmış yamuk 75x35, üst kenar yatay, ön alt
# köşesi geriye çekik (siyah OPC 34848925543 ve mavi OPC 15922355098 kalibre kameralarından geri izdüşüm:
# üst kenar 1259..1342, alt 1277..1357, kapı çizgisine göre düzeltilmiş merkez x≈1320, z 690-737 -> 712)
YAN_SINYAL = YPoligon([(1282.5, 729.5), (1357.5, 729.5), (1357.5, 694.5), (1300.5, 694.5)], 7)
# OPC köşe hava girişi (fren soğutma). Ölçüler iki kalibre fotoğraftan (mavi OPC ön çapraz + lacivert OPC)
# görünen kenarların gövde yüzeyine / derin parçalar için ofset yüzeye ışın izdüşümüyle, iki kamera ortalaması:
#   KOSE_AGIZ   yuvarlak üçgen ağız (tepe y 722 z 484, iç alt köşe 580/315, dış alt köşe 824/312)
#   KOSE_PLAKA  saten gümüş plaka: ağzın iç ve alt yanını saran "bumerang"; ağız kenarından basamakla
#               başlar, çan ağzı gibi içe dönerek kanala iner (derinlik alanı _kose_t, Laplace çözümü)
#   KOSE_KANAL  siyah kanal çerçevesinin dış hattı (yuvarlak üçgen, plakadan çıkıntılı); içinde petek ağ
#   KOSE_KIRIM  ağzın tepesinden çamurluk/far alt köşesine yükselen keskin tampon kırımı; altı içe oyulur
KOSE_AGIZ = YPoligon([(722, 484), (690, 472), (662, 458), (640, 432), (615, 395), (594, 362), (580, 334),
                      (582, 314), (600, 302), (630, 298), (680, 295), (730, 296), (775, 298), (808, 302),
                      (824, 316), (818, 342), (810, 370), (796, 405), (780, 440), (762, 465), (742, 480)], 6)
KOSE_KANAL = YPoligon([(694, 452), (734, 451), (782, 334), (670, 338)], 13)
KOSE_PLAKA = YPoligon([(540, 530), (735, 530), (733, 492), (707, 452), (679, 343), (765, 341), (790, 331),
                       (815, 323), (860, 321), (860, 240), (540, 240)], 3)
KOSE_KIRIM = Egri([(700, 478), (725, 486), (761, 498), (794, 511), (828, 529), (862, 543), (884, 554), (910, 562)])


def _kose_plaka_alani(h=2.0):
    """Plaka derinlik alanı t(y, z): ağız kenarında 0, kanal/boşluk sınırında 1. Laplace denkleminin çözümü
    olduğundan köşelerde kırım (medial eksen izi) bırakmaz: plaka tek parça düzgün içbükey yüzey olur."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    ys = np.arange(540.0, 870.0, h)
    zs = np.arange(266.0, 516.0, h)
    Yg, Zg = np.meshgrid(ys, zs, indexing="ij")
    a = KOSE_AGIZ(Yg, Zg)
    pl = np.maximum(KOSE_PLAKA(Yg, Zg), -KOSE_KANAL(Yg, Zg))
    ic = (a < 0) & (pl < 0)
    sinir = np.where(a >= 0, 0.0, 1.0)
    idx = -np.ones(Yg.shape, int)
    n = int(ic.sum())
    idx[ic] = np.arange(n)
    I, J = np.nonzero(ic)
    k = idx[I, J]
    sat, sut, deg = [k], [k], [np.full(n, 4.0)]
    b = np.zeros(n)
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        I2, J2 = I + di, J + dj
        kom = idx[I2, J2]
        m = kom >= 0
        sat.append(k[m]); sut.append(kom[m]); deg.append(np.full(int(m.sum()), -1.0))
        np.add.at(b, k[~m], sinir[I2[~m], J2[~m]])
    A = coo_matrix((np.concatenate(deg), (np.concatenate(sat), np.concatenate(sut))), shape=(n, n)).tocsr()
    t = sinir.copy()
    t[ic] = spsolve(A, b)
    return ys[0], zs[0], h, t


_KOSE_T = _kose_plaka_alani()


def kose_t(Y, Z):
    from scipy.ndimage import map_coordinates
    y0, z0, h, t = _KOSE_T
    Yb, Zb = np.broadcast_arrays(Y, Z)
    return map_coordinates(t, [(Yb - y0) / h, (Zb - z0) / h], order=1, mode="nearest")
# Arka stop (uzun bumerang): köşeyi saran lamba "açılmış" U koordinatında tanımlanır:
#   U = |y| + max(0, 4400 - x)   (arka yüzde U≈|y|, köşeden sonra yan yüzde öne doğru artar)
# Kenarlar beyaz OPC tam arka (8366545858; kalibre kamera tekerlek + tavan noktalarıyla kurulu, lambadan ve
# amblemden bağımsız) ve gri GTC arka çapraz (54188905081; kamera amblem merkezine göre yerel ötelendi)
# fotoğraflarından yüzeye ışın izdüşümü; iki ölçüm 5-15 mm içinde örtüşür. Lamba eskisinden daha yatık: üst kenar
# içten dışa yalnız ~45 mm yükselir (U 476 z 893 -> U 1031 z 940), alt kenar U 380-800 arasında neredeyse yatay
# (~775-795), iç uç yuvarlak küt burun (U 335, z ~806). Eski tablo (siyah OPC kamerasıyla ortalama) dış üst kenarı
# ~25 mm yüksek, iç alt kenarı ~30 mm yüksek ve dış alt kenarı ~40 mm sarkık çiziyordu: lamba dışa doğru dik
# kalkıyor, dış ucu küt kalıyordu. Dış uç (U 1165, z 941; yüzeyde x≈3970) ~33°'lik sivri uçtur: alt kenar mavi OPC
# (26608068581) ve gri GTC'de aynı eğimle (U 1004 z ~842, U 1089 z ~896) üst kenara kavuşur; bagaj kapağı D direği
# aralığı (BAGAJ_YAN) tam bu uca iner.
STOP_UST = Egri([(335, 806), (345, 826), (362, 848), (385, 866), (410, 878), (476, 893), (546, 909), (630, 918),
                 (738, 928), (880, 937), (1031, 940), (1165, 941)])
STOP_ALT = Egri([(335, 806), (350, 799), (380, 794), (430, 792), (480, 791), (526, 787), (570, 781), (620, 777),
                 (680, 773), (740, 773), (800, 779), (860, 788), (920, 800), (960, 815), (1004, 840), (1050, 867),
                 (1090, 893), (1125, 915), (1165, 941)])
_su = [(u, float(STOP_UST(u))) for u in (335, 337, 340, 344, 349, 355, 362, 372, 385, 400, 430, 476, 546, 630, 738,
                                         880, 1031, 1100, 1140, 1165)]
_sa = [(u, float(STOP_ALT(u))) for u in (1155, 1145, 1135, 1125, 1110, 1090, 1070, 1050, 1027, 1004, 982, 960, 940,
                                         920, 890, 860, 830, 800, 770, 740, 710, 680, 650, 620, 590, 560, 526, 500,
                                         480, 455, 430, 405, 380, 365, 355, 348, 342, 338)]
# köşe yuvarlatması 2 mm (eskiden 5 mm: 33°'lik uçta poligon ~20 mm geri çekilip küt bir burun bırakıyordu)
STOP = YPoligon(_su + _sa, 2)


def stop_u(X, Y):
    return np.abs(Y) + np.maximum(0.0, 4400.0 - X)


def stop_uv(pts, adim=20.0):
    """(U, v) çizgisi -> (U, z); v: lambanın yerel alt (0) - üst (1) kenarı arasında. Sık örneklenir."""
    out = []
    for (u0, v0), (u1, v1) in zip(pts[:-1], pts[1:]):
        n = max(1, int(abs(u1 - u0) / adim))
        for k in range(n):
            t = k / n
            u, v = u0 + (u1 - u0) * t, v0 + (v1 - v0) * t
            out.append((u, float(STOP_ALT(u) + v * (STOP_UST(u) - STOP_ALT(u)))))
    u, v = pts[-1]
    out.append((u, float(STOP_ALT(u) + v * (STOP_UST(u) - STOP_ALT(u)))))
    return out


# lamba içi (U, v): iç/dış birim ayrımı (bagaj kapağı aralığı) eğik geçer; krom bıçak v≈0.36'da tüm boyu
# geçip dış birimde "¬" LED bacağıyla birleşir (kanat grafiği); altında geri vites / sinyal odaları.
STOP_ARALIK = [(526.0, 0.0), (612.0, 1.0)]   # beyaz OPC: alt ucu U 526 (gülümseme burada biter), üst ucu U ~605
STOP_KROM = [(372, 0.42), (420, 0.38), (520, 0.36), (640, 0.36), (720, 0.37), (762, 0.40)]
STOP_LED_IC = [(445, 0.60), (560, 0.62)]
STOP_LED_DIS = [(628, 0.72), (760, 0.72), (835, 0.71), (812, 0.58), (786, 0.49), (762, 0.40)]
STOP_KANAT = ([(900, 0.80), (1000, 0.80), (1100, 0.76), (1160, 0.70)], [(890, 0.46), (1000, 0.48), (1100, 0.52), (1150, 0.58)])
STOP_CERCEVE = [(918, 1.05), (905, 0.80), (885, 0.45), (850, 0.22), (800, -0.05)]
STOP_ODA_IC = [(370, -0.1), (505, -0.1), (545, 0.36), (480, 0.32), (420, 0.34), (370, 0.40)]
STOP_ODA_DIS = [(540, -0.1), (700, -0.1), (700, 0.34), (566, 0.33)]
STOP_LED_ALT = [(716, 0.16), (762, 0.12)]
# OPC arka: siyah alt difüzör (tampon alt kenarı z≈380), uçlarında tek trapez egzoz (y 436..645, z 250..330)
# OPC arka tampon (beyaz OPC tam arka + siyah OPC çapraz fotoğraflarından yüzeye ışın izdüşümü, iki taraf simetrik):
# siyah difüzör üst kenarı z≈372 (uçlarda |y|≈650'den aşağı kıvrılır), içinde y 0, ±230'da dikey kanatlar;
# trapez egzoz uçları: üst kenar |y| 444..618 z 338, alt kenar 478..680 z 262 (dış kenar aşağıda dışa açılır);
# reflektörler |y| 362..590 z 570..600 (dış ucu eğik); park sensörleri |y| 222, 625 z 432; plaka 520x110.
DIFUZOR = YPoligon([(-724, 120), (724, 120), (722, 250), (712, 300), (690, 345), (650, 364), (560, 370), (300, 373),
                    (-300, 373), (-560, 370), (-650, 364), (-690, 345), (-712, 300), (-722, 250)], 18)
EGZOZ = YPoligon([(444, 338), (618, 338), (682, 262), (478, 262)], 13)
REFLEKTOR = YPoligon([(362, 600), (592, 585), (572, 569), (366, 575)], 5)
# tampon kırımı (beyaz OPC 8366545858, mavi OPC 26608068581): reflektörlerin üst kenarı hizasında tüm genişlikte
# keskin yatay basamak; altındaki yüz 0.3 mm geride. Kırım yüksekliği |y|'ye göre: reflektör üst kenarını izler,
# eğik dış ucundan sonra köşe boyunca ~45° aşağı iner (beyaz OPC'de sol köşeden geri izdüşüm: |y| 605/640/673/708 ->
# z 547/533/519/501) ve |y| 700-780'de söner; basamak 7 mm'lik yumuşak geçişle iner. Plaka lambası yarıkları basamakta.
KIRIM_Z = Egri([(0, 600), (362, 600), (560, 587), (592, 584), (606, 568), (625, 548), (640, 536), (673, 519),
                (708, 502), (800, 470)])
KIRIM_D = p(0.3)
# plaka yuvası (plakasız fotoğraf 8366545858, kalibre kamerayla yüzeye ışın izdüşümü): yamuk, üst kenar 300 mm
# (z 568), alt kenar 352 mm (z 497) — yan duvarlar aşağı doğru dışa açılır; tabanında iki koyu tapa (Ø20, |y| 130)
PLAKA_YUVA = YPoligon([(-150, 568), (150, 568), (176, 497), (-176, 497)], 9)
PLAKA_TAPA = ((-130.0, 505.0), (130.0, 505.0))
# bagaj kapağı gülümsemesinin amblem altındaki çentiği: aralık ortada ~|y| 90 boyunca aşağı genişler, taban z≈695
GULUMSEME_CENTIK = YPoligon([(-90, 714), (-78, 706), (-64, 698), (-50, 695), (50, 695), (64, 698), (78, 706),
                             (90, 714)], 3)
# Arka yazılar (beyaz OPC tam arka + mavi/siyah OPC çapraz fotoğraflarından kalibre kamerayla yüzeye ışın izdüşümü):
# "opc" kabartma rozeti sağ (+Y) stobun iç ucunun altında, bagaj kapağı aralığının ~28 mm üstünde (merkez y 370,
# z 766, ~74 x 24 mm); plaka yuvasının tabanında kabartma italik "GTC" (merkez y 0, z 528, ~160 x 28 mm) —
# plaka takılıyken görünmez, ARKA_PLAKA = False (--plakasiz) ile plakasız yuvada görünür. Başka yazı yok.
ARKA_PLAKA = True
OPC_ROZET = (370.0, 766.0)          # x yüksekliği merkezi (y, z); beyaz OPC: gülümsemenin ~28 mm üstü
GTC_YAZI = (0.0, 528.0)             # yazı merkezi (y, z)
DIF_KANAT = (0.0, 230.0)
DIF_KANAT_ALT = 198.0     # kanatların alt kenarı (gerçek mm)
SENSOR = ((222.0, 432.0), (625.0, 432.0))
# kapı aralıkları (yan görünüş x, z). 0: ön (çamurluk-kapı), bel hizasında ön üçgen camın alt kenarına, camın
# sivri ucunun hemen arkasında biter (fotoğraflarda çizgi kemer çıtasına ucun arkasından değer; cam çerçevesinin
# A direği boyunca giden aralığı yandan görünmez). 1: arka, bel üstünde B direği kaplaması ile krom çıta arasından
# DLO kromunun üst kenarına çıkar; alt köşe 2 ile tek eğri. 2: alt, marşpiyel kırımının (ESIK_KAPI) tam üstünde:
# oluğun alt duvarı kırımın kendisidir (eskiden oluk kırımın üstüne biniyor, kırım boncuk boncuk görünüyordu).
_ka = lambda x: (float(x), float(ESIK_KAPI(x)) + KAPI_W)
KAPI_CIZGILERI = [
    [_ka(1368), (1370, 500), (1372, 700), (1388, 820), (1410, 895), (1426, 926)],
    [(2800, 1388), (2797, 1080), (2792, 700), (2780, 480), (2752, 400), (2722, 345), (2685, 309), _ka(2640)],
    [_ka(2640), _ka(2440), _ka(2120), _ka(1800), _ka(1368)],
    # ön tampon - çamurluk: farın dış ALT köşesinden (cam hattının u≈884, z≈735 noktası) aşağı-geriye, ön teker
    # kemerinin dudağına. Kalibre ön çapraz (15922355098: (383,803,736) (411,841,711) (459,871,685) (533,891,652)
    # (601,903,623)) ve siyah OPC yan (34848925543: (457,672) (493,654) (530,637) (567,619) (603,603) (632,590))
    # kameralarından yüzeye geri izdüşümün ortalaması; iki kamera 8-15 mm içinde uyuşur. Eski hat (540,818)->
    # (728,652) farın arka ÜST ucundan çıkıyordu: fotoğraflardaki dikişin 90-120 mm üstünde/arkasında. Yanda yüzey
    # normaline göre işlenir (detaylar(): on_dikis); ilk nokta far camının içinde, cam kenarı oluğunda kesilir.
    [(362, 748), (383, 731), (411, 707), (458, 678), (532, 644), (602, 613), (657, 588), (684, 576)],
    # arka tampon - çamurluk: stobun alt ön köşesinden (U≈1000, z≈858) arka teker kemerinin dudağına; mavi OPC
    # 26608068581 kalibre kamerasından geri izdüşüm (4132,858) (4104,813) (4060,760) (4011,711) (3953,667)
    # (3933,645). Eski hat (4300,786)->(3948,562) ~170 mm aşağıdaydı, stoba değmiyor ve kemerin önünde bitiyordu.
    # Bu hat yanda yüzey normaline göre işlenir (detaylar(): arka_dikisler), kapilar() döngüsünde değil.
    # Stobun dolgun alt kenarıyla (STOP_ALT) hat, lambanın alt dış köşesinde (4135, 835) kenar oluğundan çıkar ve
    # 45° düz iner; ilk nokta lens içinde, oluk ortasında kesilir.
    [(4152, 852), (4135, 835), (4098, 799), (4060, 762), (4011, 712), (3962, 678), (3930, 642), (3888, 596)],
]
KAPI_ON = yumusat(KAPI_CIZGILERI[0], 12.0)
KAPI_ARKA = yumusat(KAPI_CIZGILERI[1] + KAPI_CIZGILERI[2][1:], 12.0)      # arka + alt: köşe teğet sürekli
_kk = KAPI_ARKA[(KAPI_ARKA[:, 1] > KAPI_CIZGILERI[1][-1][1] + 4.0) & (KAPI_ARKA[:, 1] < 1100.0)][::-1]
KAPI_ARKA_X = Egri([(200.0, _kk[0, 0] - 0.9 * (_kk[0, 1] - 200.0))] + [(z, x) for x, z in _kk])  # z -> arka kenar x


# kaput ön kenarının ÖNDEN görünüşteki yüksekliği (fotoğraf ölçümü): ızgaranın ~100 mm üstünden geçer,
# y≈610'da farın üst kenarına iner, oradan far üst kenarını izleyerek dış üst köşeye (y≈785) çıkar
KAPUT_ON_Z = Egri([(0, 838), (200, 833), (366, 822), (530, 812), (610, 804), (680, 818), (740, 832), (785, 843)])


def kaput_arka_x(y):
    """Kaputun arka kenarı (plan): ortada x 1372, A direği dibinde (|y| 700) ~1411. Fotoğraflarda silecekler bu
    kenarın hemen arkasındaki siyah kanalda, camın tabanında yatar."""
    return 1372.0 + 34.0 * (y / 650.0) ** 2


def cam_taban_x(y):
    """Ön camın görünen alt kenarı (torpido/kalorifer ızgarası paneli ile cam ek yeri), plan."""
    return 1405.0 + 55.0 * (y / 650.0) ** 2


# kaput-çamurluk aralığı (üstten, x, y): farın dış sivri ucundan (u 932 z 842 -> x 534 y 766) başlar, arkada
# camın tabanına doğru içe kıvrılır (çamurluk A direği dibinde camın altına kadar uzanır). Kalibre ön çapraz
# fotoğraftan (15922355098) yüzeye geri izdüşümle; eski hat (560,792)-(1425,694) farın ucunu ~25 mm dışından
# geçiyor ve arkada A direği dibine düz gidiyordu (fotoğrafta y≈616'da cama varır; 6146990208'de de görülür).
# Çamurluk üstünde x 900-1100 arası fotoğraftaki koyu aralık çizgisinin piksel izi (15922355098) yeniden geri
# izdüşürüldü: (921,762) (953,761) (985,761) (1027,755) (1067,744) -> hat burada eskisinden 8-14 mm dışarıda;
# önde (x<900) ve arkada (x>1150) eski hat 3-6 mm içinde doğruydu, farın ucu değişmez.
KAPUT_YAN = [(534.0, 766.0), (629.0, 764.0), (764.0, 762.0), (900.0, 759.0), (1000.0, 756.0), (1060.0, 747.0),
             (1103.0, 736.0), (1196.0, 698.0), (1290.0, 656.0), (1377.0, 618.0), (1450.0, 588.0)]
KAPUT_YAN_Y = Egri(KAPUT_YAN)


def kaput_poligon():
    """Kaput çizgisinin plan poligonu: her y için gövde yüzeyinin KAPUT_ON_Z(y) yüksekliğine ulaştığı x.
    y ~10 mm aralıkla, x 0.1 mm adımla örneklenir ve Catmull-Rom ile yumuşatılır: eski 13 örnek (1 mm adım)
    önden bakışta y ±200/366/530'da kırık/dalgalı bir kaput çizgisi bırakıyordu."""
    ys = np.arange(0.0, 761.0, 10.0)
    xs = np.arange(0.0, 900.0, 0.1)
    on = []
    for yv in ys:
        d = govde_temel(xs[:, None, None], np.array([[[float(yv)]]]), np.array([[[float(KAPUT_ON_Z(yv))]]]))[:, 0, 0]
        i = int(np.argmax(d < 0))
        # alt-ızgara doğrusal ara değer: yüzeyin tam kesiştiği x
        x0 = xs[i] - 0.1 * d[i] / (d[i] - d[i - 1]) if i > 0 and d[i - 1] > 0 else xs[i]
        on.append((float(x0), float(yv)))
    on = [tuple(q) for q in yumusat(np.array(on), 10.0)]
    sag = on + [tuple(q) for q in yumusat(np.array(KAPUT_YAN), 10.0)]
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


# arka gövde (x>3450): hedef yarı genişlik tablosu HW_ARKA(x, z) [mm], doğrudan bikübik yüzey (eski formülle
# 3450..3700 arasında harmanlanır). Kesitler dışbükey: basen stop bandına kadar dolgun kalır, stobun üstünde
# gövde D direğine doğru yuvarlanarak içe döner. Değerler beyaz OPC tam arka ve siyah OPC arka çapraz
# siluetlerinden geri izdüşümle (x≈4100'de z 960→|y| 700, 1010→630, 1040→570); 3450 satırı eski formülden.
HA_X = [3450, 3630, 3800, 3950, 4100, 4250, 4400, 4500]
HA_Z = [150, 300, 450, 600, 700, 800, 875, 950, 1000, 1050, 1100, 1175, 1300]
HW_ARKA = {
    3630: [824, 879, 897, 912, 906, 880, 860, 830, 800, 755, 722, 737, 732],
    3800: [823, 879, 898, 906, 876, 836, 806, 774, 745, 705, 662, 640, 620],
    3950: [822, 877, 894, 887, 845, 800, 768, 735, 705, 648, 588, 555, 530],
    4100: [820, 875, 885, 862, 828, 780, 745, 708, 660, 585, 528, 500, 480],
    4250: [818, 873, 881, 862, 822, 760, 718, 676, 612, 528, 485, 465, 455],
    4400: [814, 869, 877, 865, 830, 762, 712, 662, 575, 497, 455, 440, 430],
    4500: [814, 869, 877, 865, 830, 762, 712, 662, 575, 497, 455, 440, 430]}
_HA_SPL = None


def _eski_hw(X, Z):
    """Eski arka formülü (basen + daralma tablosu), omuz çizgisi ve çamurluk kabarıklığı hariç."""
    zc = omuz_z(X)
    return HW_PLAN(X) * F_DIKEY(Z) + 32.0 * sstep(2950, 3550, X) * (1 - sstep(4000, 4400, X)) \
        * sstep(700, 900, Z) * (1 - sstep(zc - 30.0, zc + 80.0, Z)) - kuyruk_daralma(X, Z)


# üst arka yan düzeltmesi UD(x, z) [mm, + = içe çek]: üç kalibre fotoğrafın (beyaz OPC tam arka, siyah OPC ve mavi OPC
# arka çapraz) siluet eşlemesinden. Gerçek araçta D direği camın kenarına kadar içe çekilir (bel), stobun üstünde
# yuvarlak dolgun bir omuz kalır; spoyler uçları direğin dışına taşar. Tablo bikübik, x<3550 ve z<900'de sıfır.
UD_X = [3480, 3580, 3650, 3720, 3820, 3950, 4100, 4250, 4350, 4450, 4550]
UD_Z = [900, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1250, 1300]
UD = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -3, -3, 12, 21, 22, 15, 6, 0, 0],
    [0, -5, -5, 17, 30, 32, 25, 13, 2, 0],
    [0, -6, -6, 19, 32, 34, 27, 15, 4, 0],
    [0, -6, -6, 20, 33, 35, 28, 16, 4, 0],
    [0, -3, -3, 12, 20, 22, 20, 14, 4, 0],
    [0, -3, 0, 16, 22, 22, 18, 10, 2, 0],
    [0, -4, 6, 34, 38, 28, 14, 4, 0, 0],
    [0, -8, 10, 46, 44, 28, 12, 2, 0, 0],
    [0, -8, 10, 46, 44, 28, 12, 2, 0, 0],
    [0, -8, 10, 46, 44, 28, 12, 2, 0, 0]], float)
_UD_SPL = None
UD_YUMUSAK = (50.0, 30.0)   # tablo alt kenarlarında (x, z) yumuşak kırpma yarı genişliği [mm]


def _yumusak_alt(u, u0, a):
    """max(u, u0)'ın C1 hali: u0-a altında u0, u0+a üstünde u, arada parabol."""
    t = u - u0
    return u0 + np.where(t > a, t, np.where(t < -a, 0.0, (t + a) ** 2 / (4.0 * a)))


def ust_duzeltme(X, Z):
    global _UD_SPL
    if _UD_SPL is None:
        from scipy.interpolate import RectBivariateSpline
        _UD_SPL = RectBivariateSpline(UD_X, UD_Z, UD, kx=3, ky=3, s=0)
    # tablonun alt kenarları (x=3480, z=900) yumuşak kırpılır: sert kırpmada spline eğimi orada sıçrıyor ve
    # C direği önünde dikey (x=3480), basen üstünde yatay (z=900) ince bir kırım çizgisi kalıyordu
    Xb, Zb = np.broadcast_arrays(np.clip(_yumusak_alt(X, UD_X[0], UD_YUMUSAK[0]), UD_X[0], UD_X[-1]),
                                 np.clip(_yumusak_alt(Z, UD_Z[0], UD_YUMUSAK[1]), UD_Z[0], UD_Z[-1]))
    return _UD_SPL.ev(Xb.ravel(), Zb.ravel()).reshape(Xb.shape)


def arka_hw(X, Z):
    global _HA_SPL
    if _HA_SPL is None:
        from scipy.interpolate import RectBivariateSpline
        zk = np.array(HA_Z, float)
        satir = [_eski_hw(np.full_like(zk, HA_X[0]), zk)] + [np.array(HW_ARKA[x], float) for x in HA_X[1:]]
        _HA_SPL = RectBivariateSpline(HA_X, HA_Z, np.array(satir), kx=3, ky=3, s=0)
    Xb, Zb = np.broadcast_arrays(np.clip(X, HA_X[0], HA_X[-1]), np.clip(Z, HA_Z[0], HA_Z[-1]))
    return _HA_SPL.ev(Xb.ravel(), Zb.ravel()).reshape(Xb.shape)


# ============================================================ gövde
_ZU_EGIM_SON = float(Z_ALT_UST.df(L))


def govde_temel(X, Y, Z):
    hwx, dhwx = HW_PLAN(X), HW_PLAN.t(X)
    fz, dfz = F_DIKEY(Z), F_DIKEY.t(Z)
    # F_DIKEY z=1100'de biter: üstünde sabit kalınca hem genişlik kırılıyor hem de d_yan'ın eğim düzeltmesi
    # sıçrıyordu (gövde üstü >1100 olan arka çeyrekte C direği altından geçen ince yatay çizgi). C1 doğrusal
    # uzatma; x>3450'de genişliği arka_hw tablosu belirler, orada yalnız eğim düzeltmesi sürekli kalır.
    e_son = float(F_DIKEY.df(F_DIKEY.hi))
    dfz = dfz + e_son * (Z >= F_DIKEY.hi)
    hw = hwx * (fz + e_son * np.maximum(Z - F_DIKEY.hi, 0.0) * (1 - sstep(3450, 3700, X)))
    # omuz çizgisi
    zc = omuz_z(X)
    # arkada stobun sivri ön ucunda (x≈3950) söner: stobun üst kenarı bu çizginin devamıdır
    hw = hw + p(0.35) * np.maximum(0, 1 - np.abs(Z - zc) / p(1.7)) * sstep(500, 900, X) * (1 - sstep(3820, 4120, X))
    # GTC "bıçak": keskin üst kenar BICAK_UST, altında aşağı doğru sönen içbükey bant; ön sınırı çapraz kıvrım
    # kesit: kenarda eğim kırığı (keskin sırt), altında 12 mm'de 20 mm içe kıvrılan yüzey. Eski 3 mm'lik
    # 20 mm basamak aşağı bakan bir saçak gibiydi; ızgarada kenar boyunca kesik kesik (- - -) çizgi bırakıyordu
    alt_b = BICAK_UST(X) - Z
    tb = np.clip(alt_b / 12.0, 0.0, 1.0)
    oy = 20.0 * (1.0 - (1.0 - tb) ** 2) * (1 - sstep(40.0, 235.0, alt_b))
    hw = hw - oy * sstep(-35.0, 45.0, X - BICAK_ON(Z)) * (1 - sstep(3080, 3330, X))
    # OPC marşpiyel
    hw = hw + 10.0 * sstep(Z_ALT(X) + 8.0, Z_ALT(X) + 38.0, Z) * esik_basamak(X, Z) \
        * sstep(1262.0, 1292.0, X) * (1 - sstep(3228.0, 3262.0, X))
    # çamurluk kabarıklıkları (arka daha belirgin) + OPC kıvrık dudak: kemer kenarında ~10 mm'lik yuvarlak
    # bir sırt (kuyuya kıvrılan sac), arkasında sığ bir oluk, sonra kabarık yüzey
    for xa, a in ((AKS_ON, 0.6), (AKS_ARKA, 0.85)):
        r = np.sqrt((X - xa) ** 2 + (Z - AKS_Z) ** 2)
        e = r - CAMUR_R
        kab = sstep(-12.0, 0.0, e) * (1 - sstep(p(1.4), p(4.0), e))
        dudak = 4.0 * np.exp(-((e - 7.0) / 5.5) ** 2) - 2.6 * np.exp(-((e - 18.0) / 4.5) ** 2)
        hw = hw + (p(a) * kab + dudak) * sstep(150, 250, Z)
    # GTC arka basen (haunch): arka teker üstünde omuz bölgesi dışa şişer
    hw = hw + 32.0 * sstep(2950, 3550, X) * (1 - sstep(4000, 4400, X)) * sstep(700, 900, Z) * (1 - sstep(zc - 30.0, zc + 80.0, Z))
    # arka daralma (fotoğraf siluetlerinden): basenin üstü içe döner, teker arkasında gövde hızla stop köşesine kaçar
    hw = hw - kuyruk_daralma(X, Z)
    # x>3450: dışbükey kesitli arka yüzey tablosu (basen stop bandını dolgun taşır, üstte D direğine yuvarlanır)
    w = sstep(3450, 3700, X)
    if np.any(w > 0):
        hw = hw + w * (arka_hw(X, Z) - _eski_hw(X, Z))   # omuz çizgisi ve çamurluk kabarıklığı korunur
    if np.any(X > UD_X[0] - UD_YUMUSAK[0]):
        hw = hw - ust_duzeltme(X, Z)   # D direği beli / stop üstü omuz (fotoğraf siluetleri)
    d_yan = (Y - hw) / np.sqrt(1 + (dhwx * fz) ** 2 + (hwx * dfz) ** 2)

    bombe = KAPUT_BOMBE(X)
    # Z_ALT_UST arka uçta (x = L, z 700) dik iner; Egri x > L'de sabitlenip eğimi sıfırlayınca d_ust orada
    # sıçrıyordu (-9 -> -162 mm): d_arka ile yumuşak max'ın şişkinliği x≈4466'da kaybolup alan yüzeyin ~10 mm
    # arkasında tekrar ~0-5 mm'ye iniyordu (plaka çevresinde hayalet ikinci kabuk; seviyeli dolgular orada
    # kopuk levha üretiyordu). Uç eğimiyle doğrusal uzatma: alan yüzeyin dışında tekdüze artar.
    ext = np.maximum(X - L, 0.0)
    zu = Z_ALT_UST(X) + _ZU_EGIM_SON * ext - bombe * (Y / 900.0) ** 2
    kaput = sstep(120, 260, X) * (1 - sstep(1250, 1350, X))
    # kaput kenarı boyunca yumuşak şişkinlik (çan kesitli; eski üçgen "çadır" profil keskin bir kırım çiziyordu):
    # y≈690'da, arkada kaput-çamurluk aralığı içe kıvrıldığında onun ~45 mm içinden izler
    yr = smin(690.0, KAPUT_YAN_Y(X) - 45.0, 20.0)
    zu = zu + kaput * (p(1.1) * np.maximum(0, 1 - (Y / 560.0) ** 2) ** 2 + p(0.3) * np.exp(-((Y - yr) / 65.0) ** 2))
    d_ust = (Z - zu) / np.sqrt(1 + (Z_ALT_UST.t(X) + _ZU_EGIM_SON * (X >= L)) ** 2 + (2 * bombe * Y / 900.0 ** 2) ** 2)
    # ön tampon alt kenarı köşelerde yükselir (köşe bıçaklarının altı z≈295)
    z_alt = Z_ALT(X) + 60.0 * sstep(470.0, 820.0, Y) * (1 - sstep(300.0, 540.0, X))
    d_alt = (z_alt - Z) / np.sqrt(1 + Z_ALT.t(X) ** 2)
    if np.any(X < 600.0):
        # tamponun alt yüzü dudağın altına çekilir: yüzün dönüş (alta kıvrılma) çizgisi ON_DUDAK_ALT'ın ~6 mm altında.
        # Yükseltme yalnız yüzün hemen arkasındaki bantta (X - ON_PLAN 70..260), teker kemerine ve yan yüze (|y| > 870)
        # varmadan söner; ortada (|y| < 120) alt hat (yandan görünen tampon alt çizgisi) değişmez;
        # dudaksız eski hal y 300-830'da yüzü dudağın 30-50 mm altına indirip altında ikinci bir şişkinlik bırakıyordu.
        def _z_alt(xx, yy):
            z1 = Z_ALT(xx) + 60.0 * sstep(470.0, 820.0, yy) * (1 - sstep(300.0, 540.0, xx))
            ek = np.maximum(0.0, ON_DUDAK_ALT(yy) - 6.0 - z1) * (1 - sstep(70.0, 260.0, xx - ON_PLAN(yy))) \
                * (1 - sstep(440.0, 560.0, xx)) * sstep(120.0, 280.0, yy) * (1 - sstep(870.0, 930.0, yy))
            return z1 + ek
        Xa, Ya = np.broadcast_arrays(X, Y)
        z2 = _z_alt(Xa, Ya)
        gx = (_z_alt(Xa + 0.5, Ya) - _z_alt(Xa - 0.5, Ya))
        gy = (_z_alt(Xa, Ya + 0.5) - _z_alt(Xa, Ya - 0.5))
        d_alt = np.where(X < 600.0, (z2 - Z) / np.sqrt(1 + gx ** 2 + gy ** 2), d_alt)
    d_on = (ON_PLAN(Y) + ON_YAN(Z) - X) / np.sqrt(1 + ON_PLAN.t(Y) ** 2 + ON_YAN.t(Z) ** 2)
    d_arka = (X - (L - ARKA_PLAN(Y) - ARKA_YAN(Z))) / np.sqrt(1 + ARKA_PLAN.t(Y) ** 2 + ARKA_YAN.t(Z) ** 2)

    alt = smax(d_yan, d_ust, p(4.5) - 38.0 * sstep(3900, 4150, X))   # arkada D direği kırımı daha belirgin
    alt = smax(alt, d_on, p(3.0))
    alt = smax(alt, d_arka, p(3.5))
    alt = smax(alt, d_alt, p(1.2))

    zk = Z_KABIN(X) - 40.0 * (Y / 650.0) ** 2
    # eğim (normalleştirme) tablonun arka ucunda (x 4420) sıfıra sıçramaz, 4400-4480 arasında söner: sıçrama
    # d_kust'u 2.4 kat değiştiriyor, kabinin smin şişkinliği bagaj kapağında x≈4420'de 15 mm'lik dik bir duvar
    # bırakıyordu (ortada z 955-970, yanlara inen yatay kırım; amblemin üstünden stopların üst kenarına uzanıyordu)
    zk_t = np.where(X < Z_KABIN.hi - 20.0, Z_KABIN.t(X),
                    float(Z_KABIN.df(Z_KABIN.hi - 20.0)) * (1 - sstep(Z_KABIN.hi - 20.0, Z_KABIN.hi + 60.0, X)))
    d_kust = (Z - zk) / np.sqrt(1 + zk_t ** 2 + (80.0 * Y / 650.0 ** 2) ** 2)
    eg = EGIM_X(X)
    d_kyan = (Y - hw_kabin(X, Z)) / np.sqrt(1 + eg ** 2 + (HW_KABIN.t(X) - (Z - 1000.0) * EGIM_X.t(X)) ** 2)
    kabin = smax(d_kyan, d_kust, p(2.6))
    # yan cam arkasında (C/D direği) kabin tabanı biraz iner ve alt dış kenarı geniş yuvarlanır: direk
    # omuza basamak bırakmadan akar. Geçiş x≈3560'tan önce bitmeli: orada kabin yan yüzü (UD ile) gövde
    # omzunun dışına çıkıyor; taban henüz yüksek/keskin kalırsa ikisinin arasında z≈1080'de çukur oluşuyordu
    w_d = sstep(3470, 3650, X)
    kabin = smax(kabin, (Z_ALT_UST(X) - bombe * (Y / 900.0) ** 2 - 35.0 - 60.0 * w_d) - Z, p(0.5) + 55.0 * w_d)
    # kabinin arka ucu (x 4380 düzlemi) yumuşak kesilir: sert max'ın kırığı, arkada geniş (k≈86 mm) smin
    # harmanlamasıyla gövdeye taşınıp stopların iç ucundan dış üst köşesine (y 260 z 777 -> y 605 z 868, x 4444-4406)
    # uzanan testere dişli bir kırım çizgisi bırakıyordu
    kabin = smax(kabin, X - 4380.0, 40.0)
    kabin = np.maximum(kabin, 950.0 - X)

    d = smin(alt, kabin, p(0.9) + 40.0 * sstep(3620, 3900, X) + 25.0 * sstep(3900, 4150, X))   # arkada D direği - basen geçişi yumuşak
    return smin(d, spoyler(X, Y, Z), p(0.5))


# spoyler uç plakasının yandan görünüşü (x, z): sivri alt uç, ön kenarın üstü, alt kenarın arka ucu. Uç, cam yan
# kenarının (direk yüzeyinde |y| ≈ ARKA_CAM_YAN eşyükselti çizgisi) ön ucundadır.
UC_UC = (3815.0, 1192.0)
UC_ON = (3790.0, 1345.0)
UC_ARKA = (3988.0, 1262.0)


def _uc_kenar(X, Z):
    """Uç plakasının ön kenarına (s_on, + = önde) ve alt kenarına (s_alt, + = altta) işaretli uzaklık (x-z düzlemi)."""
    ax, az = UC_ON[0] - UC_UC[0], UC_ON[1] - UC_UC[1]
    bx, bz = UC_ARKA[0] - UC_UC[0], UC_ARKA[1] - UC_UC[1]
    la, lb = math.hypot(ax, az), math.hypot(bx, bz)
    dx, dz = X - UC_UC[0], Z - UC_UC[1]
    return (dz * ax - dx * az) / la, (dx * bz - dz * bx) / lb


def spoyler(X, Y, Z):
    # OPC tavan spoyleri: tavanın devamı, arka kenarda ~55 mm yüksek yüz (3. stop burada), dışbükey arka
    # kenar, iki yanda D direği boyunca aşağı inen uç plakaları (fotoğraf: alt dudak z≈1275, uçlar |y|≈570)
    yy = np.minimum((Y / 560.0) ** 2, 1.0)
    # arka kenarda yukarı kalkar (beyaz OPC: arka yüz z≈1265..1370, fren lambası 1280-1300)
    sp_ust = 1356.0 + 0.10 * np.maximum(X - 3800.0, 0) - 26.0 * yy
    x_ak = 3992.0 + 26.0 * (1 - yy)
    kal = 30.0 + 75.0 * sstep(3880.0, 4010.0, X)
    sp = np.maximum(X - x_ak, 3790.0 - X)
    # uçlar D direğinin dışına taşar (beyaz OPC tam arka: kanat ucu direğin ~40 mm dışında)
    sp_ek = SP_TASMA * sstep(3800.0, 3900.0, X)
    # yan yüz uç plakasının üst dış yüzüyle aynı hizada (hk + 10): eskiden 14 mm içeride kalıp plakanın üst
    # kenarında basamak/çentik bırakıyordu
    sp = smax(sp, Y - (hw_kabin(X, sp_ust) + 10.0 + sp_ek), 15.0)
    sp = smax(sp, np.maximum(Z - sp_ust, (sp_ust - kal) - Z), 8.0)
    # uç plakaları (mavi OPC 26608068581 ve siyah OPC 8080939836): spoylerin yan yüzünün altında, kabin yan yüzeyine
    # oturan üçgen kanat. Ön kenarı spoylerin önünden (x≈3790) neredeyse dikey iner, sivri alt ucu (UC_UC) cam yan
    # kenarının ön ucundadır; alt kenarı oradan spoylerin arka alt dudağına yükselir (altında siyah alt yüz + cam).
    # Dış yüzün direkten taşması ucun etrafında sıfırdan büyür: kanat direkten kırımsız çıkar. Eskiden plaka arkada
    # (x≈3995) z≈1145'e kadar inip direğe kaynıyordu; orası fotoğrafta camdır ve direkte kırışık/raf bırakıyordu.
    hk = hw_kabin(X, Z)
    s_on, s_alt = _uc_kenar(X, Z)
    uc_ek = sp_ek * sstep(1180.0, 1310.0, Z)
    r_uc = np.sqrt((X - UC_UC[0]) ** 2 + (Z - UC_UC[1]) ** 2)
    tas = -6.0 + (16.0 + uc_ek) * sstep(0.0, 120.0, r_uc)
    uc = smax(np.maximum(Y - (hk + tas), (hk - 50.0) - Y), np.maximum(smax(s_on, s_alt, 6.0), Z - sp_ust), 6.0)
    uc = smax(uc, X - (x_ak - 6.0), 8.0)
    return smin(sp, uc, 10.0)


# arka cam: yan kenar beyaz OPC tam arka fotoğrafında (kalibre kamera, arkadan bakış: y/z sağlam) |y|≈578'e kadar
# uzanır, alt dış köşe y≈579 z≈1100, alt kenar y 387'de 1051, 500'de ~1075. Eski sınır (kabin yan yüzü - 24 mm)
# camı D direğinde ~120 mm dar bırakıyordu: cam köşesi (x 4235, y 420, z 1090) spoyler ucundan ve stobun sivri
# ucundan kopuktu. Yan sınır alt bölümde |y| 578'e açılır, spoyler gövdesi (uç plakaları dahil) camdan hariç tutulur.
ARKA_CAM_YAN = 578.0


def arka_cam_maske(X, Y, Z):
    # yan kenar: direk yüzeyinde |y| ≈ ARKA_CAM_YAN eşyükselti çizgisi (köşeden uç plakasının sivri ucuna, mavi OPC);
    # ön sınır uç plakasının ön kenarı: cam, plakanın altından ve arkasından spoylerin alt yüzüne kadar uzanır
    hk = hw_kabin(X, Z)
    s_on, s_alt = _uc_kenar(X, Z)
    yk = smax(hk - p(1.0), ARKA_CAM_YAN, 12.0)
    # ortada (spoylerin altı) ön sınır eskisi gibi x 3930: plakanın ön kenarına kadar uzatılınca spoyler gövdesinin
    # altında kapalı bir boşluk kalıyordu
    on = np.maximum(s_on, 3930.0 - X - 400.0 * sstep(500.0, 540.0, Y))
    m = smax(smax(on, X - 4400.0, p(1.2)), Y - yk, p(0.9))
    # plakanın yandan görünüşü de camdan hariç (ucun yakınında plaka direğin içine gömülü: cam tabanı orada plakanın
    # dış yüzüyle neredeyse çakışıp ızgarada hareli bir şerit bırakıyordu)
    m = smax(m, -smax(smax(s_on, s_alt, 6.0), (hk - 40.0) - Y, 6.0), 4.0)
    m = smax(m, (1030.0 + 72.0 * sstep(290.0, 610.0, Y)) - Z, p(0.8))
    return smax(m, 3.0 - spoyler(X, Y, Z), 4.0)


def _yuzey_nokta(x, z, y0=300.0, y1=1000.0):
    """(x, z) için detaysız gövde yüzeyinin dış y'si (yan yüzde çizgi noktalarını yüzeye oturtmak için)."""
    ys = np.arange(y0, y1, 0.25)
    dv = govde_temel(np.array([[[float(x)]]]), ys[None, :, None], np.array([[[float(z)]]]))[0, :, 0]
    return (float(x), float(ys[np.nonzero(dv < 0)[0][-1]]), float(z))


def _yuzey_egri(xz, adim=8.0):
    """(x, z) kontrol noktalarından geçen pürüzsüz eğri (kiriş uzunluğuyla parametreli kübik spline), ~adim mm'de
    örneklenip her nokta yüzeye oturtulur: kırık çizgi köşeleri olukta kıvrım bırakmaz."""
    from scipy.interpolate import CubicSpline
    a = np.asarray(xz, float)
    t = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(a, axis=0).T))])
    cs = CubicSpline(t, a, axis=0)
    tt = np.linspace(0.0, t[-1], max(3, int(t[-1] / adim) + 1))
    return np.array([_yuzey_nokta(x, z) for x, z in cs(tt)])


def cizgi3(X, Y, Z, pts):
    """3B kırık çizgiye uzaklık."""
    pts = np.asarray(pts, float)
    d = None
    for a, b in zip(pts[:-1], pts[1:]):
        e = b - a
        h = np.clip(((X - a[0]) * e[0] + (Y - a[1]) * e[1] + (Z - a[2]) * e[2]) / float(e @ e), 0.0, 1.0)
        dd = (X - a[0] - e[0] * h) ** 2 + (Y - a[1] - e[1] * h) ** 2 + (Z - a[2] - e[2] * h) ** 2
        d = dd if d is None else np.minimum(d, dd)
    return np.sqrt(d)


# bagaj kapağı yan aralığı (D direği): cam alt dış köşesinden stobun sivri ön ucuna (mavi OPC 26608068581 ve beyaz
# OPC tam arka kalibre kameralarından geri izdüşüm: (3933, 628, 1079), (3921, 668, 1046), (3974, 662, 1039),
# (3955, 714, 999) -> stop ucu U 1185 z 966). (x, z) verilir, y yüzeyden okunur.
# Alt ucu stobun sivri dış ucuna (U 1165 z 941, yüzeyde x≈3970) iner: mavi OPC'de aralık uca ~30 mm yukarıdan
# gelip geriye doğru kıvrılarak kavuşur. Son nokta lensin uç kamasının içinde: oluk lens-gövde oluğunun ucuyla birleşir
# (st_sinir oluk ortasında keser), arada çıkıntı kalmaz.
BAGAJ_YAN = [(4021.0, 1100.0), (3985.0, 1084.0), (3950.0, 1062.0), (3932.0, 1040.0), (3933.0, 1015.0),
             (3940.0, 990.0), (3948.0, 968.0), (3960.0, 951.0), (3970.0, 941.0), (3976.0, 938.5)]
_BAGAJ_YAN_3B = None


def bagaj_yan_3b():
    global _BAGAJ_YAN_3B
    if _BAGAJ_YAN_3B is None:
        _BAGAJ_YAN_3B = _yuzey_egri(BAGAJ_YAN)
    return _BAGAJ_YAN_3B


_TAMPON_DIKIS_3B = None
_ON_DIKIS_3B = None


def on_dikis_3b():
    global _ON_DIKIS_3B
    if _ON_DIKIS_3B is None:
        _ON_DIKIS_3B = _yuzey_egri(KAPI_CIZGILERI[3], 4.0)
    return _ON_DIKIS_3B


def tampon_dikis_3b():
    global _TAMPON_DIKIS_3B
    if _TAMPON_DIKIS_3B is None:
        _TAMPON_DIKIS_3B = _yuzey_egri(KAPI_CIZGILERI[4])
    return _TAMPON_DIKIS_3B


KAPUT = kaput_poligon()


def camurluk_kuyusu(X, Y, Z, xa):
    dx, dz = X - xa, Z - AKS_Z
    daire = np.sqrt(dx * dx + dz * dz) - CAMUR_R
    damla = np.maximum((dz + np.abs(dx)) / math.sqrt(2) - CAMUR_R, dz - 470.0)
    damla = np.maximum(damla, -dz)
    ic = np.minimum(daire, damla)
    return np.maximum(np.where(Y < 800.0, ic, daire), KUYU_Y - Y)


# ============================================================ tekerlek (gerçek mm, yerel)
# Astra J OPC 20" jantı (fotoğraflardan): 5 adet "kemik" kol. Her kol göbekten kısa, geniş bir gövdeyle çıkar, boğumlanır ve
# r≈160'ta yuvarlak dipli bir U ile iki ince, kavisli kola ayrılır; kollar dudakta genişleyip büyük kavislerle janta
# bağlanır. Çatal içi pencereler dudakta yuvarlatılmış üçgen, kollar arası pencereler büyük ve göbeğe kadar iner.
# Kol çiftinin işlenmiş ön yüzü dudak düzleminde, U'nun altındaki gövde göbeğe doğru dik çanakla ~34 mm içeri dalar;
# kenarlar pahlı, yan yüzler arkaya doğru genişler. Göbek: 5 bijon yuvası (kollar arasında, PCD 115), Opel şimşekli göbek kapağı.
# Ön aksta 355 mm delikli/havalandırmalı disk ve Brembo 4 pistonlu kaliper (logo plakalı); arka aksta 315 mm
# ince (havalandırmasız) delikli disk ve tek pistonlu yüzer kaliper (logosuz, fotoğraf: mavi OPC arka teker); sibop.
# Lastik 245/35 R20: jant koruma sırtı, şişkin yanak, yanak yazısı, 4 kanallı asimetrik diş deseni.
# Koordinatlar: dx, dz göbekten düzlem içi (gerçek mm), t eksenel (+ dış yüz).
JANT_FAZ = math.pi / 2                  # bir kol tam yukarı bakar
KALIPER_ACI = math.radians(170.0)       # ön kaliper aksın önünde (dx < 0), saat 9 civarı
ARKA_KALIPER_ACI = math.radians(12.0)   # arka kaliper aksın arkasında (dx > 0), hafif yukarıda
JANT_SIMSEK = np.array([(-1.18, 0.03), (0.02, 0.30), (0.21, 0.06), (1.18, -0.03), (-0.02, -0.30), (-0.21, -0.06)])
LASTIK_YAZI = (("245/35 ZR20 95Y", 90.0), ("245/35 ZR20 95Y", 270.0))   # (metin, görünüşte merkez açısı)
# yanak yazısı: harf yüksekliği, taban yarıçapı, çizgi yarı kalınlığı. Çizgi 7.5 mm (baskıda 0.31 mm), kabartma
# p(0.15) = 3.6 mm: reçine yazıcıda basılabilir; harf aralığı çizgi kalınlığına göre açıldı (harfler birbirine değmez)
YAZI_H, YAZI_R0, YAZI_IZ = 21.0, 294.5, 3.75
# yanak profili: lastik dış yanağının t konumu (r'ye göre); 280-292 jant koruma sırtı, ~310 en şişkin yer
LASTIK_YANAK = Egri([(276.0, 104.0), (280.0, 108.5), (284.0, 115.5), (288.0, 119.5), (291.0, 120.3),
                     (295.0, 118.8), (300.0, 120.3), (309.0, 121.8), (318.0, 122.0)])
# çizgi yazı tipi: genişlik 0.6, yükseklik 1 birim; her karakter bir veya birkaç kırık çizgi
_FONT = {
    "0": [[(0.15, 0), (0.45, 0), (0.6, 0.15), (0.6, 0.85), (0.45, 1), (0.15, 1), (0, 0.85), (0, 0.15), (0.15, 0)]],
    "1": [[(0.12, 0.78), (0.36, 1), (0.36, 0)]],
    "2": [[(0, 0.82), (0.16, 1), (0.44, 1), (0.6, 0.84), (0.6, 0.64), (0, 0), (0.6, 0)]],
    "3": [[(0, 1), (0.6, 1), (0.28, 0.6), (0.46, 0.58), (0.6, 0.42), (0.6, 0.15), (0.45, 0), (0.14, 0), (0, 0.14)]],
    "4": [[(0.46, 0), (0.46, 1), (0, 0.32), (0.62, 0.32)]],
    "5": [[(0.6, 1), (0.04, 1), (0, 0.56), (0.44, 0.6), (0.6, 0.44), (0.6, 0.15), (0.45, 0), (0.12, 0), (0, 0.12)]],
    "6": [[(0.56, 1), (0.2, 1), (0, 0.7), (0, 0.15), (0.15, 0), (0.45, 0), (0.6, 0.15), (0.6, 0.4), (0.45, 0.56), (0, 0.56)]],
    "7": [[(0, 1), (0.6, 1), (0.2, 0)]],
    "8": [[(0.3, 0.56), (0.08, 0.66), (0.08, 0.9), (0.18, 1), (0.42, 1), (0.52, 0.9), (0.52, 0.66), (0.3, 0.56),
           (0.6, 0.42), (0.6, 0.12), (0.46, 0), (0.14, 0), (0, 0.12), (0, 0.42), (0.3, 0.56)]],
    "9": [[(0.6, 0.44), (0.15, 0.44), (0, 0.6), (0, 0.85), (0.15, 1), (0.45, 1), (0.6, 0.85), (0.6, 0.3), (0.4, 0), (0.04, 0)]],
    "/": [[(0, 0), (0.5, 1)]],
    "Z": [[(0, 1), (0.6, 1), (0, 0), (0.6, 0)]],
    "R": [[(0, 0), (0, 1), (0.44, 1), (0.6, 0.86), (0.6, 0.64), (0.44, 0.5), (0, 0.5)], [(0.3, 0.5), (0.6, 0)]],
    "Y": [[(0, 1), (0.3, 0.5), (0.6, 1)], [(0.3, 0.5), (0.3, 0)]],
    "G": [[(0.6, 0.85), (0.45, 1), (0.15, 1), (0, 0.85), (0, 0.15), (0.15, 0), (0.45, 0), (0.6, 0.15), (0.6, 0.45), (0.34, 0.45)]],
    "T": [[(0, 1), (0.6, 1)], [(0.3, 1), (0.3, 0)]],
    "C": [[(0.6, 0.85), (0.45, 1), (0.15, 1), (0, 0.85), (0, 0.15), (0.15, 0), (0.45, 0), (0.6, 0.15)]],
}


def _elips_cizgi(cx, cy, a, b, t0, t1, n=28):
    t = np.linspace(t0, t1, n)
    return [(cx + a * math.cos(q), cy + b * math.sin(q)) for q in t]


# küçük harfler (OPC logosu): x yüksekliği 0.42, inen çizgi -0.30; harfler logodaki gibi geniş ve yuvarlak
_FONT["o"] = [_elips_cizgi(0.31, 0.21, 0.31, 0.21, 0.0, 2 * math.pi)]
_FONT["p"] = [_elips_cizgi(0.31, 0.21, 0.31, 0.21, 0.0, 2 * math.pi), [(0.0, 0.42), (0.0, -0.30)]]
_FONT["c"] = [_elips_cizgi(0.31, 0.21, 0.31, 0.21, math.radians(42), math.radians(318))]


def duz_yazi_2d(u, v, metin, h, sx=1.0, egim=0.0, bosluk=0.0):
    """Düz satır yazı: harf çizgilerine 2D uzaklık. u sağa, v yukarı (gerçek mm), (0, 0) ilk harfin taban
    çizgisinin sol ucu; h: büyük harf yüksekliği (çizgi ekseni), sx: yatay ölçek, egim: italik, bosluk: ardışık harflerin
    çizgi eksenleri arası aralık (mm; çizgi kalınlığını da karşılamalı). Döndürür: (uzaklık, toplam genişlik)."""
    d = np.full(np.broadcast(u, v).shape, 1e3)
    x0 = 0.0
    for c in metin:
        if c == " ":
            x0 += 0.5 * h * sx
            continue
        lx = u - x0 - egim * v
        gen = max(a for cz in _FONT[c] for a, _ in cz) * h * sx
        for cz in _FONT[c]:
            d = np.minimum(d, _cizgi_mesafe(lx, v, [(a * h * sx, b * h) for a, b in cz]))
        x0 += gen + bosluk
    return d, x0 - bosluk


def _cizgi_mesafe(px, py, pts):
    """Kırık çizgiye uzaklık (2D)."""
    d = None
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        ex, ey = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        h = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0, 1)
        dd = (wx - ex * h) ** 2 + (wy - ey * h) ** 2
        d = dd if d is None else np.minimum(d, dd)
    return np.sqrt(d)


def lastik_yazi_2d(xv, yv, r):
    """Yanak yazısı: harf çizgilerine 2D uzaklık. xv, yv: dışarıdan bakana göre sağ ve yukarı.
    Yazı tabanı göbeğe bakar, saat yönünde okunur (gerçek lastikteki gibi), hafif italik."""
    thv = np.arctan2(yv, xv)
    rm = YAZI_R0 + YAZI_H / 2
    y = r - YAZI_R0
    d = np.full(np.broadcast(xv, yv).shape, 1e3)
    for metin, aci in LASTIK_YAZI:
        aralik = 0.6 * YAZI_H + 2 * YAZI_IZ + 3.0                            # harf genişliği + çizgi + 3 mm boşluk
        uz = sum(0.55 * YAZI_H if c == " " else aralik for c in metin)
        s = (np.mod(math.radians(aci) - thv + math.pi, 2 * math.pi) - math.pi) * rm + uz / 2
        if np.all(np.abs(s - uz / 2) > uz / 2 + 30):
            continue
        x0 = 0.0
        for c in metin:
            if c == " ":
                x0 += 0.55 * YAZI_H
                continue
            lx = s - x0 - 0.18 * y
            for cz in _FONT[c]:
                d = np.minimum(d, _cizgi_mesafe(lx, y, [(a * YAZI_H, b * YAZI_H) for a, b in cz]))
            x0 += aralik
    return d


def lastik_yanak(r):
    return LASTIK_YANAK(r)


def teker_asimetrik_detay(xv, yv, t):
    """Ayna simetrik olmayan tekerlek detayları (yanak yazısı, göbek kapağı şimşeği) - kabartma SDF'i.
    xv, yv: dış yüze bakana göre sağ / yukarı (gerçek mm)."""
    r = np.sqrt(xv * xv + yv * yv)
    tsw = lastik_yanak(r)
    yazi = np.maximum(lastik_yazi_2d(xv, yv, r) - YAZI_IZ, np.maximum(t - (tsw + p(0.15)), (tsw - 3.0) - t))
    yazi = np.maximum(yazi, np.maximum(YAZI_R0 - 6.0 - r, r - (YAZI_R0 + YAZI_H + 6.0)))
    R = 21.0
    m = sd_poligon(xv / R, yv / R, JANT_SIMSEK) * R
    kapak_t = 58.0 + 2.0
    simsek = np.maximum(np.maximum(m, r - 27.0), np.maximum(t - (kapak_t + p(0.1)), (kapak_t - 3.0) - t))
    return np.minimum(yazi, simsek)


def _altigen(px, py, a):
    """Düz kenarları arası 2a olan altıgene uzaklık (2D)."""
    kx, ky, kz = -0.8660254, 0.5, 0.57735027
    px, py = np.abs(px), np.abs(py)
    m = 2 * np.minimum(kx * px + ky * py, 0)
    px, py = px - m * kx, py - m * ky
    px = px - np.clip(px, -kz * a, kz * a)
    py = py - a
    return np.sqrt(px * px + py * py) * np.sign(py)


def _on_fren(r, th, t):
    """Ön fren: disk Ø355, t -60..-18 (arka kapamaya oturur), kenarında havalandırma kanatları, yüzünde delik
    grupları; Brembo 4 pistonlu kaliper."""
    disk = sd_kutu2(r, t, 134.5, -39.0, 42.5, 21.0, 2.0)
    sv = (np.mod(th * 40 / (2 * math.pi) + 0.5, 1.0) - 0.5) * (2 * math.pi * r / 40)
    disk = np.maximum(disk, -np.maximum(np.maximum(np.abs(t + 39.0) - 6.0, 171.0 - r), 3.0 - np.abs(sv)))
    G = 24
    ag = np.mod(th, 2 * math.pi / G)
    delik = 1e3
    for rk, ak in ((116.0, 1.6), (137.0, 5.4), (158.0, 9.2)):
        a_k = math.radians(ak)
        delik = np.minimum(delik, np.sqrt((r - rk) ** 2 + (rk * (ag - a_k)) ** 2) - 4.2)
    disk = np.maximum(disk, -np.maximum(delik, -32.0 - t))
    sapka = sd_kutu2(r, t, 0.0, -36.0, 96.0, 28.0, 4.0)
    # Brembo kaliper: disk kenarını saran yay biçimli gövde, kubbeli dış yüz, logo plakası, hava alma nipeli
    dth = np.mod(th - KALIPER_ACI + math.pi, 2 * math.pi) - math.pi
    s = r * dth
    kal2 = sd_kutu2(r, s, 170.0, 0.0, 42.0, 125.0, 30.0)
    tc = 32.0 + 6.0 * np.clip(1.0 - (s / 125.0) ** 2, 0.0, 1.0) - 4.0 * ((r - 170.0) / 42.0) ** 2
    kaliper = smax(kal2, t - tc, 10.0)
    kaliper = np.maximum(kaliper, -70.0 - t)
    kaliper = np.maximum(kaliper, -np.maximum(np.maximum(np.abs(s) - 46.0, 203.0 - r), np.abs(t + 22.0) - 18.0))  # balata penceresi
    plaka2 = sd_kutu2(r, s, 168.0, 0.0, 17.0, 84.0, 8.0)
    plaka = np.maximum(plaka2, np.maximum(t - (tc + p(0.1)), (tc - 4.0) - t))
    kaliper = np.minimum(kaliper, plaka)
    nipel = np.maximum(np.sqrt((r - 196.0) ** 2 + (s + 106.0) ** 2) - 4.5, np.maximum(t - (tc + 7.0), 0.0 - t))
    kaliper = np.minimum(kaliper, nipel)
    return np.minimum(np.minimum(disk, sapka), kaliper)


def _arka_fren(r, th, t):
    """Arka fren (Astra J OPC): Ø315 ince havalandırmasız disk (t -38..-18), yüzünde küçük delik grupları;
    disk kenarına binen kompakt tek pistonlu yüzer kaliper (logo plakası yok): dış yüzde iki parmaklı köprü,
    iki ucunda kılavuz pimi körükleri ve taşıyıcı braket."""
    disk = sd_kutu2(r, t, 125.0, -28.0, 32.5, 10.0, 2.0)
    G = 20
    ag = np.mod(th, 2 * math.pi / G)
    delik = 1e3
    for rk, ak in ((112.0, 2.0), (128.0, 6.5), (144.0, 11.0)):
        a_k = math.radians(ak)
        delik = np.minimum(delik, np.sqrt((r - rk) ** 2 + (rk * (ag - a_k)) ** 2) - 3.6)
    disk = np.maximum(disk, -np.maximum(delik, -30.0 - t))
    sapka = sd_kutu2(r, t, 0.0, -36.0, 96.0, 28.0, 4.0)
    dth = np.mod(th - ARKA_KALIPER_ACI + math.pi, 2 * math.pi) - math.pi
    s = r * dth
    # gövde: disk kenarını saran (r 128..188), yay boyunca ±58 mm, dış yüzü t≈12 kubbeli
    kal2 = sd_kutu2(r, s, 158.0, 0.0, 30.0, 58.0, 16.0)
    tc = 8.0 + 4.0 * np.clip(1.0 - (s / 58.0) ** 2, 0.0, 1.0) - 3.0 * ((r - 158.0) / 30.0) ** 2
    kaliper = smax(kal2, t - tc, 6.0)
    kaliper = np.maximum(kaliper, -60.0 - t)
    # parmaklar arası yarık (dış yüzde balata sırtı görünür)
    kaliper = np.maximum(kaliper, -np.maximum(np.maximum(np.abs(s) - 9.0, 150.0 - r), -6.0 - t))
    # taşıyıcı braket: kaliperin iki ucundan aks yönüne inen ince kulaklar
    braket = np.maximum(sd_kutu2(r, s, 150.0, 0.0, 38.0, 82.0, 10.0), np.maximum(t - 0.0, -14.0 - t))
    braket = np.maximum(braket, 112.0 - r)
    # kılavuz pimi körükleri (iki uçta, dışa bakan kısa silindirler)
    pim = 1e3
    for sc in (-66.0, 66.0):
        pim = np.minimum(pim, np.maximum(np.sqrt((r - 176.0) ** 2 + (s - sc) ** 2) - 7.0, np.maximum(t - 6.0, -20.0 - t)))
    kaliper = np.minimum(np.minimum(kaliper, braket), pim)
    return np.minimum(np.minimum(disk, sapka), kaliper)


# jant kolu planı (gerçek mm, açıklama tekerlek_sdf içinde): kol çiftinin dış kenarının kol eksenine dik uzaklığı (r'ye
# göre) ve çatal penceresi hiperbolü (uç yarıçapı, uçtaki eğrilik yarıçapı, asimptot eğiminin karesi)
KOL_DIS = Egri([(85.0, 62.0), (95.0, 56.0), (100.0, 50.0), (110.0, 38.5), (120.0, 32.0), (130.0, 28.5), (140.0, 27.5),
                (150.0, 30.0), (160.0, 35.5), (170.0, 42.0), (180.0, 49.0), (200.0, 62.0), (230.0, 79.0), (250.0, 88.0),
                (275.0, 100.0)])
CATAL_R, CATAL_RHO, CATAL_M2 = 159.0, 15.3, 0.207


def tekerlek_sdf(dx, dz, t, yon=None, arka_aks=False):
    """yon: None -> ayna simetrik olmayan detaylar (yazı, şimşek) eklenmez (gövdede asimetrik() ekler);
    +1/-1 -> dış yüze bakana göre sağ yön = yon * dx. arka_aks: True -> arka aks freni (küçük disk, tek pistonlu
    yüzer kaliper), False -> ön aks Brembo freni."""
    r = np.sqrt(dx * dx + dz * dz)
    th = np.arctan2(dz, dx)

    # ---------------------------------------------------------------- lastik
    tsw = lastik_yanak(r)
    gsw = np.sqrt(1.0 + LASTIK_YANAK.t(r) ** 2)
    r_dis = TEKER_R - 4.0 * (t / 110.0) ** 2                     # hafif bombeli sırt
    yan = (t - tsw) / gsw
    lastik = smax(yan, r - r_dis, 20.0)
    lastik = smax(lastik, -TEKER_YG - t, 20.0)                    # arka yanak düz (tablaya oturur)
    lastik = np.maximum(lastik, 276.0 - r)
    ham = lastik
    # 4 çevresel kanal (asimetrik desen: dış omuz blokları daha iri)
    for tc, w in ((-78.0, 5.5), (-25.0, 6.0), (29.0, 6.0), (80.0, 5.5)):
        kanal = np.maximum(np.abs(t - tc) - w, (r_dis - 8.0) - r)
        lastik = np.maximum(lastik, -kanal)
    # omuz enine kanalları: sırttan omuzun yuvarlağına kıvrılır; yanağa inmez (r>=332), yandan bakınca yalnız
    # lastiğin en dış kenarında diş çentikleri görünür
    N = 60
    u = np.mod(th * N / (2 * math.pi), 1.0)
    for isaret, kay in ((1.0, 0.0), (-1.0, 0.5)):
        ta = isaret * t
        s = (np.mod(u + kay + 0.006 * np.maximum(ta - 86.0, 0) + 0.5, 1.0) - 0.5) * (2 * math.pi * r / N)
        omuz = np.maximum(np.maximum(np.abs(s) - 3.6, 86.0 - ta), np.maximum(-ham - 7.0, 332.0 - r))
        lastik = np.maximum(lastik, -omuz)
    # ara bloklarda kanaldan çıkan eğik kör yarıklar (her iki adımda bir)
    for tc0, tc1, kay in ((34.0, 75.0, 0.25), (-73.0, -30.0, 0.75)):
        tm = np.clip((t - tc0) / (tc1 - tc0), 0, 1)
        s = (np.mod(u / 2 + kay + 0.35 * tm + 0.5, 1.0) - 0.5) * (4 * math.pi * r / N)
        yarik = np.maximum(np.maximum(np.abs(s) - 3.0, np.maximum(tc0 + 12.0 - t, t - tc1)), (r_dis - 6.0) - r)
        lastik = np.maximum(lastik, -yarik)
    # yanakta ince süs çizgisi (yazı bandının dışı)
    halka = np.maximum(np.abs(r - 323.5) - 2.6, np.maximum(-ham - 1.8, -t))
    lastik = np.maximum(lastik, -halka)

    # ---------------------------------------------------------------- jant gövdesi
    arka = np.maximum(r - 240.0, np.maximum(-TEKER_YG - t, t + 60.0))          # arka kapama (düz taban)
    # jant çanağı (pencerelerden görünür): iç yüzü dudağın altından (r≈261, t=80) içe doğru konik daralır (t≈30'da
    # r=238), böylece pencerelerin dudak tarafı fotoğraftaki gibi derin ve gölgeli görünür (düz bir basamak değil)
    kovan = sd_kutu2(r, t, 258.5, -21.0, 20.5, 101.0, 3.0)
    r_ic = 238.0 + 23.0 * np.clip((t - 30.0) / 50.0, 0.0, 1.0) ** 1.5
    kovan = np.maximum(kovan, (r_ic - r) * 0.93)
    dudak = sd_kutu2(r, t, 271.75, 92.0, 10.75, 14.0, 5.0)                     # ön dudak (r 261-282.5), yüzü t≈106
    dudak = np.maximum(dudak, -np.maximum(np.abs(r - 263.5) - 3.0, 104.5 - t))  # dudak iç kenarında ince işleme çizgisi

    # 5 "kemik" kol (2D), kol eksenine göre katlanmış açı; y: kol eksenine dik uzaklık. Ölçüler 34848925543 ön teker
    # fotoğrafının elips düzeltmeli kutupsal açılımında parlak yüz / karanlık pencere eşiğinden ve pencere ana hatlarının
    # fotoğraf üzerine bindirilmesinden (5 kolun ortalaması): gövde göbekte kısa ve geniş (r=100'de ~100 mm), r≈140'ta
    # ~55 mm'ye boğumlanır, sonra kol çiftinin dış kenarı KOL_DIS(r) ile kavisli açılır; çatal içi pencere r≈159'da
    # ~15 mm yarıçaplı yuvarlak uçla başlayan hiperbol (y² = 2ρu + m²u², u = r - 159: r=170'te ±19, 200'de ±40,
    # 250'de ±67 mm); aradaki iki kol ~20 mm, çataldan dudağa yay gibi kavislenip büyük kavislerle janta bağlanır.
    # Kollar arası pencereler ~120 mm genişliğinde iner, göbeğe doğru yuvarlak bir V ile daralıp r≈97'de biter.
    a = np.mod(th - JANT_FAZ + math.pi / 5, 2 * math.pi / 5) - math.pi / 5
    aa = np.abs(a)
    q = np.clip((262.0 - r) / 172.0, 0.0, 1.0)
    y = r * np.sin(aa)
    zarf = (y - KOL_DIS(r)) / np.sqrt(1.0 + KOL_DIS.t(r) ** 2)                # kol çiftinin dış kenarı
    u = r - CATAL_R
    fc = y * y - 2.0 * CATAL_RHO * u - CATAL_M2 * u * u                      # çatal penceresi (hiperbol), içi < 0
    catal = fc / (2.0 * np.sqrt(y * y + (CATAL_RHO + CATAL_M2 * u) ** 2) + 1e-6)
    kol2 = np.maximum(zarf, -catal)
    kol2 = smin(kol2, r - 99.0, 14.0)                                       # kollar arası pencerenin yuvarlak ucu
    kol2 = np.maximum(kol2, 70.0 - r)
    kol2 = smin(kol2, 263.0 - r, 22.0)                                      # jant dudağına geniş kavisli ayaklar
    kol2 = np.maximum(kol2, r - 268.0)
    # yüz: kol çiftinin işlenmiş yüzü (çatal penceresini ~30 mm genişlikte U biçiminde saran bant) dudağa yakın düzlemde
    # (t≈106 -> 96) kalır; U'nun dibinden (r≈130) göbeğe doğru tek gövde dik bir çanakla ~34 mm içeri dalar (t≈64)
    yk = y * np.clip((CATAL_R - r) / 20.0, 0.0, 1.0)
    duc = np.sqrt(np.maximum(CATAL_R - r, 0.0) ** 2 + yk * yk)               # çatal penceresinin ucuna uzaklık
    tf = 106.0 - 10.0 * q ** 1.3 - 34.0 * sstep(30.0, 75.0, duc)
    sirt = 1.0 - sstep(130.0, 190.0, r)
    tust = tf - sirt * (7.0 - 0.35 * np.minimum(-kol2, 20.0))
    talt = tf - 38.0 - 12.0 * np.clip((200.0 - r) / 110.0, 0.0, 1.0)
    derin = np.maximum(tust - t, 0.0)
    kol3 = kol2 - 0.12 * derin                                              # yan yüzler arkaya doğru genişler
    kol = smax(kol3, t - tust, 3.0)
    kol = np.maximum(kol, (t - tust + kol2 + 4.0) * 0.70710678)            # ön yüz kenar pahı (işlenmiş yüz)
    kol = np.maximum(kol, talt - t)

    # göbek: hafif çanak yüz, 5 bijon yuvası (kollar arasında), kapak oluğu ve kubbeli kapak
    thub = 56.0 + 7.0 * sstep(30.0, 86.0, r)
    gobek = np.maximum(smax(r - 86.0, t - thub, 6.0), -TEKER_YG - t)
    ab = np.mod(th - JANT_FAZ, 2 * math.pi / 5) - math.pi / 5               # bijonlar kol aralarında
    bx, bz = r * np.cos(ab) - 57.5, r * np.sin(ab)
    rb = np.sqrt(bx * bx + bz * bz)
    yuva = np.maximum(rb - 12.5, 40.0 - t)
    bas = np.maximum(_altigen(bx, bz, 8.5), np.maximum(t - 50.0, 34.0 - t))
    bas = np.maximum(bas, (t - 50.0 + rb - 7.0) * 0.70710678)
    pul = np.maximum(rb - 10.5, np.maximum(t - 43.0, 34.0 - t))
    oluk = np.maximum(np.abs(r - 31.5) - 2.0, 50.0 - t)                    # kapak çevresi oluk
    kapak = np.maximum(r - 29.5, t - (58.0 + 2.0 * (1.0 - (r / 29.5) ** 2)))
    kapak = np.minimum(kapak, np.maximum(np.abs(r - 22.0) - 2.6, np.maximum(t - (60.0 + p(0.1)), 52.0 - t)))  # amblem halkası
    kapak = np.maximum(kapak, 40.0 - t)

    # ---------------------------------------------------------------- fren
    if arka_aks:
        fren = _arka_fren(r, th, t)
    else:
        fren = _on_fren(r, th, t)
    # sibop: kol arası pencerede, çanaktan dışa eğik çıkar
    dv = np.mod(th - JANT_FAZ, 2 * math.pi) - math.pi / 5
    pr, ps = r * np.cos(dv), r * np.sin(dv)
    hx = np.clip(((pr - 241.0) * -15.0 + (t - 48.0) * 32.0) / (15.0 ** 2 + 32.0 ** 2), 0.0, 1.0)
    sibop = np.sqrt((pr - 241.0 + 15.0 * hx) ** 2 + ps ** 2 + (t - 48.0 - 32.0 * hx) ** 2) - 4.0
    sibop = np.minimum(sibop, np.maximum(np.sqrt((pr - 228.0) ** 2 + ps ** 2) - 5.5, np.abs(t - 76.0) - 6.0))

    jant = np.minimum(np.minimum(arka, kovan), dudak)
    jant = smin(jant, kol, 4.0)
    jant = smin(jant, gobek, 18.0)
    jant = smax(jant, -yuva, 2.0)
    jant = np.maximum(jant, -oluk)
    jant = np.minimum(jant, np.minimum(np.minimum(bas, pul), kapak))
    sonuc = np.minimum(np.minimum(lastik, jant), np.minimum(fren, sibop))
    if yon is not None:
        sonuc = np.minimum(sonuc, teker_asimetrik_detay(yon * dx, dz + 0 * dx, t))
    return sonuc


def teker_asimetrik(d, kes):
    """Tek parça gövdede (ayna kopyalı hacim üzerinde) iki yanın tekerlerine yön doğru yazı ve şimşek ekler."""
    for xa in (AKS_ON, AKS_ARKA):
        for isaret in (1.0, -1.0):
            y0, y1 = TEKER_Y + 55.0, TEKER_Y + 132.0
            g = kes(xa - 345, xa + 345, *(sorted((isaret * y0, isaret * y1))), AKS_Z - 345, AKS_Z + 345)
            if g is None:
                continue
            sl, X, Y, Z = g
            # sağ (+Y) yanda dışarıdan bakınca sağ = -X, sol yanda +X
            e = teker_asimetrik_detay(-isaret * (X - xa) + 0 * Z, Z - AKS_Z + 0 * X, isaret * Y - TEKER_Y)
            d[sl] = np.minimum(d[sl], e)


# ============================================================ blok değerlendirme
# kapı kolu tutamağı (yarım hacimde, iki yanda) ve sürücü kapısındaki kilit göbeği (asimetrik(), yalnız -Y).
# Gri 54188905081 ve siyah OPC 34848925543 (ikisi de sol yan) fotoğraflarında kilit göbeği tutamağın arka
# ucunda (önden ~160 mm); omuz çizgisine göre tutamak ~12 mm daha aşağıda (merkez z 803). Sol direksiyonlu
# araçta yolcu (sağ, +Y) kapısının kolunda kilit yok.
KOL_Z = 803.0
KILIT_X = 2643.0


def kol_tutamak(X, Z):
    """Tutamak: (işaretli 2B maske, merkez z, yüzeyden yükseklik)."""
    yari = 13.0 + 2.5 * sstep(2480.0, 2680.0, X)                              # yarı yükseklik 13 -> 15.5
    zc = KOL_Z + 1.5 * sstep(2480.0, 2680.0, X)
    k = sd_kutu2(X, Z, 2580.0, zc, 100.0, yari, yari - 0.5)
    q = np.clip(-k / yari, 0.0, 1.0)
    return k, zc, p(0.5) * np.sqrt(q * (2.0 - q))


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
        # cam sınırı: yanlarda kabin yan yüzüne paralel düzlem (A direği genişliği korunur), altta kaputun arka
        # kenarı (cam, kaputun altına kadar iner; aradaki kanal silecek yuvasıdır), üstte x 2160. Kenarlar
        # p(0.1)-p(0.16) ile yuvarlatılır: ızgarada testere dişi/merdiven bırakmaz.
        YUM = p(0.1)

        def oy(dd, maske, derinlik, r=YUM):
            return smax(dd, -smax(-(z0 + derinlik), maske, r), r)

        def dol(dd, maske, seviye):
            return smin(dd, smax(z0 - seviye, maske, YUM), YUM)

        yan = Y - (hw_kabin(X, Z) - p(2.4))
        xk = kaput_arka_x(Y)
        m = smax(smax(xk - X, X - 2160.0, p(1.0)), yan, p(1.0))
        dd = oy(dd, m, CAM, p(0.16))
        # siyah serigrafi bandı (fotoğraf: A direği boyunca camın dış kenarında ~45 mm, üstte ~35 mm koyu bant):
        # camdan 0.15 mm kabarık (ızgara adımının ~2 katı: kenarı merdivensiz çıkar), boyamada kılavuz
        # (basamak 8 mm'lik yumuşak rampa: keskin basamak ızgarada testere dişi bırakıyordu)
        ic = smax(X - 2125.0, yan + 45.0, p(0.6))
        dd = dol(dd, smax(m, cam_taban_x(Y) - X, YUM), -(CAM - p(0.15) * sstep(-4.0, 4.0, ic)))
        # cam tabanı: kanal tabanı camla aynı seviyede; camın alt kenarı (torpido paneli ek yeri) ince oluk
        dd = oy(dd, np.maximum(np.abs(X - cam_taban_x(Y)) - p(0.1), m), CAM + p(0.15))
        # cam yıkama fıskiyeleri: kaputun arka kenarının hemen önünde, ortadan ±300
        r2 = np.sqrt(((X - (xk - 15.0)) / 9.0) ** 2 + ((np.abs(Y) - 300.0) / 6.0) ** 2)
        dd = dol(dd, (r2 - 1.0) * 6.0, p(0.22) * np.sqrt(np.clip(1.0 - r2 ** 2, 0.0, 1.0)))
        return dd
    uygula((1340, 2200, 0, 900, 960, 1500), on_cam)

    def arka_cam(dd, z0, X, Y, Z):
        # sınır arka_cam_maske(): alt kenar ortada z 1030, D direğinde |y| 578'e açılır, spoyler hariç.
        # Kenar 4 mm pahla iner (dik basamak eğik yan kenarda ızgarada testere dişi bırakıyordu)
        m = arka_cam_maske(X, Y, Z)
        # direğin yana bakan kısmında (y ≈ kabin yan yüzü) cam yuvası sığ (%40): 0.45 mm'lik oyuk orada siyah OPC çapraz
        # siluetini ~15 mm içeri çekiyordu. Sığlaşma yalnız Y - hk'ye bağlı ve geniş geçişli: yan kenar boyunca derinlik
        # sabit kalır (eskiden z 1110-1145 / x 4050-4090 eşiklerinde değişip köşede basamak bırakıyordu)
        hk = hw_kabin(X, Z)
        sig = 1.0 - 0.6 * sstep(hk - 70.0, hk, Y)
        # spoylerin alt dudağına yaklaşırken yuva sığlaşır: tam derinlikte spoylerin altına girip gövde içinde ince
        # kapalı boşluklar bırakıyordu
        sig = sig * sstep(6.0, 24.0, spoyler(X, Y, Z))
        return oyuk(dd, z0, m, CAM * sig * sstep(-0.5, 4.0, -m))
    uygula((3780, 4420, 0, 900, 1000, 1400), arka_cam)

    def yuzey_izdusum(z0, X, Y, Z, bant=20.0, normal=False):
        """Noktayı detaysız yüzeye (govde_temel normali boyunca) indirir: eğik yüzeylerde (cam üst kenarı
        yukarı bakar) X-Z düzlemindeki şerit maskesi kabartmayı kırpmasın diye. Gradyan noktanın kendisinde
        merkezi farkla alınır (blok/parça sınırlarında aynı değer -> parçalar arası dikiş açılmaz).
        normal=True: yüzey normalinin y bileşeni de döner (bant dışında 0)."""
        Xf, Yf, Zf = (np.broadcast_to(a, z0.shape) for a in (X, Y, Z))
        Xb, Zb = Xf.astype(float).copy(), Zf.astype(float).copy()
        ny = np.zeros(z0.shape)
        sec = np.abs(z0) < bant
        if sec.any():
            x, y, z, d = Xf[sec], Yf[sec], Zf[sec], z0[sec]
            gx = govde_temel(x + 1.0, y, z) - govde_temel(x - 1.0, y, z)
            gy = govde_temel(x, y + 1.0, z) - govde_temel(x, y - 1.0, z)
            gz = govde_temel(x, y, z + 1.0) - govde_temel(x, y, z - 1.0)
            n = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-9
            Xb[sec] = x - d * gx / n
            Zb[sec] = z - d * gz / n
            ny[sec] = gy / n
        return (Xb, Zb, ny) if normal else (Xb, Zb)

    def krom_isaretli(Xb, Zb):
        """KROM_YOL merkez çizgisine işaretli uzaklık: altında (cam tarafı) negatif, üstünde ve uçların ötesinde
        pozitif."""
        sk = sd_cizgi(Xb, Zb, KROM_YOL)
        # kavis x'e göre artan kısımla (uçta camın sivri ucunu saran geri dönüş varsa o hariç) ayrılır
        i0, i1 = int(np.argmin(KROM_YOL[:, 0])), int(np.argmax(KROM_YOL[:, 0])) + 1
        zc = np.interp(Xb, KROM_YOL[i0:i1, 0], KROM_YOL[i0:i1, 1])
        alt = (Zb < zc) & (Xb > KROM_YOL[i0, 0]) & (Xb < KROM_YOL[i1 - 1, 0])
        return np.where(alt, -sk, sk), sk

    def kabin_kenari(X):
        """Cam/krom maskelerinin iç sınırı (kabin yan yüzünden içeri, baskı mm cinsinden gerçek mm). A direği
        dibinde (x<1700) yüzey yukarı döndüğünden daha derin: eski sabit sınır ön üçgen camı ve krom ucunu
        y-düzlemiyle tırtıklı kesiyordu."""
        return p(1.2) + p(1.3) * (1.0 - sstep(1650.0, 1800.0, X))

    def yan_cam(dd, z0, X, Y, Z):
        Xb, Zb = yuzey_izdusum(z0, X, Y, Z)
        tk, _ = krom_isaretli(Xb, Zb)
        ust = tk + 8.0                                  # üst sınır = krom şeridinin alt kenarı (krom 16 mm geniş)
        (ax0, az0), (ax1, az1) = ON_CAM_ARKA
        L = math.hypot(ax1 - ax0, az1 - az0)
        arka_k = ((Xb - ax0) * (az1 - az0) - (Zb - az0) * (ax1 - ax0)) / L     # + : kapı camının arka kenarının arkası
        on = smax(smax(ON_CAM_ALT(Xb) - Zb, ust, 3.0), arka_k, 4.0)
        arka = smax(smax(ARKA_CAM_ALT(Xb) - Zb, ust, 3.0), ARKA_CAM_ON_X - Xb, 3.0)
        m = np.minimum(on, arka)
        dis = (hw_kabin(X, Z) - kabin_kenari(X)) - Y
        # cam kenarı 4 mm pahla iner (eğik cam üst kenarında dik basamak ızgarada testere dişi bırakıyordu)
        dd = oyuk(dd, z0, np.maximum(m, dis), CAM * sstep(-0.5, 4.0, -m))
        # ön üçgen cam ayracı (AYNA_CITA): cam düzleminde düz siyah çıta, camdan yalnız 0.1 mm kabarık, kenarları
        # keskin; altta ayna taban plakasına (aynı seviye, ayna()), üstte A direği kromuna (krom merkez çizgisine
        # kadar uzanır, krom üstüne biner). Eskiden çıta gövde yüzeyinde kalıp 0.1 mm bombeyle camdan ~0.55 mm
        # kabarık yuvarlak çubuk gibi duruyordu.
        cita = AYNA_CITA(Xb, Zb)
        cm = np.maximum(np.maximum(cita, tk), np.maximum(ON_CAM_ALT(Xb) - 2.0 - Zb, dis))
        return doldur(dd, z0, cm, -(CAM - p(0.1)))
    uygula((1290, 3620, 450, 900, 880, 1440), yan_cam)

    # GTC'nin imzası olan krom DLO kavisi: cam üst kenarı boyunca A direğinden arka cam ucuna kabarık şerit.
    # 16 mm geniş, 0.3 mm yüksek; kesit yuvarlak sırtlı ve pahlı (yükseklik kenarda sıfıra iner: dik duvar ve
    # kenarda ağ boncuklanması olmaz)
    def krom(dd, z0, X, Y, Z):
        Xb, Zb = yuzey_izdusum(z0, X, Y, Z)
        s_ = sd_cizgi(Xb, Zb, KROM_YOL)
        m = np.maximum(s_ - 8.0, (hw_kabin(X, Z) - p(0.3) - kabin_kenari(X)) - Y)
        h = p(0.3) * sstep(0.0, 4.5, 8.0 - s_) * (1.0 - 0.3 * (s_ / 8.0) ** 2)
        dd = doldur(dd, z0, m, h)
        # B direği: arka yan camın ön kenarında ince dikey krom çıta (x 2805-2820), aynı kesit
        b = np.abs(Xb - 2812.5)
        mb = np.maximum(np.maximum(b - 7.5, np.maximum(1078.0 - Zb, Zb - 1372.0)), (hw_kabin(X, Z) - p(1.5)) - Y)
        return doldur(dd, z0, mb, p(0.22) * sstep(0.0, 3.0, 7.5 - b))
    uygula((1290, 3640, 450, 900, 890, 1440), krom)

    # ---------- kaput çizgisi (üstten)
    def kaput(dd, z0, X, Y, Z):
        # U kesitli oluk (kenarı yüzeye teğet iner): dik duvarlı eski oluk eğik yüzeyde ızgarada kesik kesik/
        # merdivenli çıkıyordu. Farın üst kenarı boyunca (y 600-780) kaput kenarı far camının kenarıdır: orada oluk
        # söner ve far cam kenarı oluğu (farlar 11) tek çizgi olarak devam eder; eskiden kaput oluğu camın ~7 mm
        # içinden, cam kenarı oluğuna paralel gidip ızgarada kesik kesik ikinci bir çizgi bırakıyordu.
        s = sd_poligon(X, Y, KAPUT)
        w = p(0.26)
        q = np.clip(1.0 - (s / w) ** 2, 0.0, 1.0)
        fm = np.maximum(FAR_UZ(far_u(X, Y), Z), FAR_YAN(X, Z))
        derin = p(0.5) * q * np.sqrt(q) * sstep(-4.0, 3.0, fm)
        m = np.maximum(np.maximum(np.abs(s) - w, 690 - Z), X - (kaput_arka_x(Y) - 2.0))
        return oyuk(dd, z0, m, derin)
    uygula((90, 1430, 0, 790, 680, 1040), kaput)

    # ---------- ön: üst ızgara (alt kenarı aşağı sarkan mercek/tekne şekli), krom kanat, rozet
    def ust_izgara(dd, z0, X, Y, Z):
        m = np.maximum(UST_IZGARA(Y, Z), X - 300)
        dd = oyuk(dd, z0, m, p(1.0))
        # kanadın altındaki siyah yarık + ince krom alt şerit (0.34 mm, yuvarlak sırtlı, kanat ucuna kıvrılır)
        ak = sd_cizgi(np.abs(Y), Z, IZ_ALT_KROM)
        dd = doldur(dd, z0, np.maximum(ak - p(0.17), m), -p(0.5) + p(0.3) * np.sqrt(np.clip(1 - (ak / p(0.17)) ** 2, 0, 1)))
        # krom kanat çıtası: kenarlarda yüzeyin 0.35 mm altında, KANAT_SIRT boyunca gövde yüzeyine çıkan
        # keskin sırtlı (çatı kesitli) ön yüz -> üstte açık, altta koyu iki yüz
        kd = KANAT(Y, Z)
        sirt = sd_cizgi(np.abs(Y), Z, KANAT_SIRT)
        dd = doldur(dd, z0, np.maximum(kd, X - 300), -p(0.35) * sirt / (sirt - kd + 1e-6))
        # rozet: amblem asimetrik() içindeki opel_amblem() ile işlenir (merkez y 0, z 686)
        return dd
    uygula((-40, 300, 0, 500, 610, 760), ust_izgara)

    # ---------- farlar: cam hattı FAR_UZ (far haritası u, z) ∩ yandan FAR_YAN. Cam yok sayılır: gövde
    # derin bir hazne olarak oyulur, detaylar hazne tabanından katman katman yükselir (seviyeler baskı mm,
    # gövde yüzeyinin altında): çerçeve -0.35, üst yansıtıcı -0.7 (dişli), hazne tabanı -1.4,
    # kaş/DRL ışık kılavuzu -0.22, projektör halkası -0.4, mercek tepesi -0.38, çanak dibi -1.55.
    def farlar(dd, z0, X, Y, Z):
        U = far_u(X, Y)
        m = np.maximum(FAR_UZ(U, Z), FAR_YAN(X, Z))

        def far_m(x, y, z):
            return np.maximum(FAR_UZ(far_u(x, y), z), FAR_YAN(x, z))

        # cam hattına yüzey üzerinde ölçülen işaretli uzaklık (yalnız hat ve yüzey yakınında; uzakta m)
        ms = np.broadcast_to(m, z0.shape).astype(float).copy()
        sec = (np.abs(z0) < 30.0) & (np.abs(ms) < 40.0)
        if sec.any():
            Xf, Yf, Zf = (np.broadcast_to(a, z0.shape)[sec].astype(float) for a in (X, Y, Z))
            d = z0[sec]
            g = [govde_temel(Xf + e[0], Yf + e[1], Zf + e[2]) - govde_temel(Xf - e[0], Yf - e[1], Zf - e[2])
                 for e in ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0))]
            nrm = np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2) + 1e-9
            nx, ny, nz = g[0] / nrm, g[1] / nrm, g[2] / nrm
            xb, yb, zb = Xf - d * nx, Yf - d * ny, Zf - d * nz
            mb = far_m(xb, yb, zb)
            h = 0.5
            gm = [(far_m(xb + h, yb, zb) - far_m(xb - h, yb, zb)) / (2 * h),
                  (far_m(xb, yb + h, zb) - far_m(xb, yb - h, zb)) / (2 * h),
                  (far_m(xb, yb, zb + h) - far_m(xb, yb, zb - h)) / (2 * h)]
            gn = gm[0] * nx + gm[1] * ny + gm[2] * nz
            gt = np.sqrt(np.maximum(gm[0] ** 2 + gm[1] ** 2 + gm[2] ** 2 - gn ** 2, 0.0))
            ms[sec] = mb / np.maximum(gt, 0.3)

        # kenarlar ~0.08 mm yuvarlatılır: baskıda keskin kalır, ağda merdivenlenme (testere dişi) olmaz
        YUM = p(0.08)

        def oy(dd, maske, derinlik):
            return smax(dd, -smax(-(z0 + derinlik), maske, YUM), YUM)

        def dol(dd, maske, seviye):
            return smin(dd, smax(z0 - seviye, maske, YUM), YUM)

        def kat(dd, maske, seviye):
            """maske içinde yüzeyi tam 'seviye'ye getirir (oyar + doldurur)."""
            maske = np.maximum(maske, m)
            return dol(oy(dd, maske, -seviye), maske, seviye)

        def serit(pts, yari):
            return sd_cizgi(U, Z, pts) - yari

        def dis(aralik, yon):
            """yon açısıyla eğik paralel dişlerin eksenine uzaklık (aralık)."""
            t = U * math.cos(yon) + Z * math.sin(yon)
            return np.abs(np.mod(t, aralik) - aralik / 2)

        def kabart(dd, uz, yari, sinir, taban, tepe):
            """eksene uzaklığı 'uz' olan, yarı genişliği 'yari' yuvarlak sırtlı kabartma (sinir<0 bölgesinde)."""
            k = np.sqrt(np.clip(1 - (uz / yari) ** 2, 0, 1))
            return dol(dd, np.maximum(np.maximum(uz - yari, sinir), m), taban + (tepe - taban) * k)

        # 1) siyah çerçeve: camın altında, gövdeden 0.35 mm keskin basamak
        dd = kat(dd, m, -p(0.35))
        # 2) üst yansıtıcı (kaş üstü bant + dış uçta sinyal üçgeni): -0.7, üstünde ince eğik optik doku. Fotoğrafta
        # bu bant ince bir kumlama/benek dokusudur: eski 0.75 mm aralıklı 0.28 mm'lik dişler 6-7 kaba panjur gibi
        # okunuyordu. Aralık 0.3 mm, yükseklik 0.06 mm.
        ust = FAR_UST(U, Z)
        dd = kat(dd, ust, -p(0.7))
        dd = kabart(dd, dis(p(0.3), math.radians(-62)), p(0.1), ust + p(0.15), -p(0.7), -p(0.64))
        # 3) derin iç hazne
        hz = FAR_HAZNE(U, Z)
        dd = kat(dd, hz, -p(1.4))
        # 4) iç uzun yansıtıcı (iç uç tarafı): 0.35 mm kabarık plaka, 0.28 mm oluklarla 2x4 fasete bölünmüş
        plaka = np.maximum(hz + p(0.3), U - 598.0)
        dd = kat(dd, plaka, -p(1.05))
        oluk = np.minimum(np.minimum(np.abs(U - 462.0), np.abs(U - 510.0)), np.abs(U - 556.0)) - p(0.14)
        oluk = np.minimum(oluk, np.abs(Z - (713.0 + 0.12 * (U - 500.0))) - p(0.14))
        dd = oy(dd, np.maximum(np.maximum(oluk, plaka), m), p(1.4))
        # 5) projektör altındaki alt yansıtıcı: öne eğik yüz (alt kenarı -0.6, üst kenarı -1.0)
        alt = FAR_ALT_REF(U, Z)
        dd = kat(dd, alt, -p(0.6) - p(0.4) * sstep(660.0, 682.0, Z))
        # 6) uzun far çanağı: kabarık kenar + parabolik çanak + ortada ampul kapağı
        r = np.sqrt((U - FAR_UZUN[0]) ** 2 + (Z - FAR_UZUN[1]) ** 2)
        dd = kat(dd, r - 29.0, -p(0.8))
        dd = kat(dd, r - 24.0, -p(0.9) - p(0.65) * np.clip(1 - (r / 24.0) ** 2, 0, 1))
        dd = kat(dd, r - 6.5, -p(1.0))
        # 7) projektör: siyah çanak (halka arkası derin gölge) + krom halka + bombeli mercek
        r = np.sqrt(((U - FAR_PROJ[0]) * 1.08) ** 2 + (Z - FAR_PROJ[1]) ** 2)
        dd = kat(dd, r - 47.0, -p(1.75))
        dd = kat(dd, np.maximum(r - 42.0, 32.0 - r), -p(0.4) - p(0.12) * sstep(38.0, 42.0, r))
        dd = kat(dd, r - 31.0, -p(1.0) + p(0.62) * np.sqrt(np.clip(1 - (r / 31.0) ** 2, 0, 1)))
        # 8) dış bölme: küçük sinyal çanağı (halka + çanak + ampul)
        r = np.sqrt((U - 788.0) ** 2 + (Z - 734.0) ** 2)
        dd = kat(dd, np.maximum(r - 15.0, 9.0 - r), -p(0.85))
        dd = kat(dd, r - 9.0, -p(1.3))
        dd = kat(dd, r - 4.0, -p(1.0))
        # 9) krom ayraç (uzun far çanağı ile projektör arası) ve alt ışık kılavuzu (iç uçtan ayraca)
        dd = dol(dd, np.maximum(serit([(664.0, 689.0), (667.0, 757.0)], p(0.17)), m), -p(0.45))
        dd = dol(dd, np.maximum(serit([(410, 693), (500, 694), (590, 694), (628, 690), (652, 693), (665, 702)],
                                             p(0.17)), m), -p(0.4))
        # 10) kanat DRL ışık kılavuzu / krom kaş: hazne üst kenarı boyunca iç uçtan dışa, dış uçta projektörün
        # yanından aşağı kıvrılır (kanca). Yuvarlak sırtlı, hazne tabanından 1.2 mm yüksek.
        kas = sd_cizgi(U, Z, [(440, 724), (500, 736), (550, 743), (600, 748), (655, 755), (705, 764), (755, 772),
                               (797, 777), (806, 772), (809, 750), (807, 722), (800, 700)])
        dd = dol(dd, np.maximum(kas - p(0.21), m), -p(0.22) - p(0.12) * (kas / p(0.21)) ** 2)
        # 11) cam kenarı: dış hat boyunca 0.35 mm genişlik, 0.7 mm derin gölge oluğu (camın gövdeye oturduğu ek yeri).
        # Genişlik yüzey üzerinde ölçülür: nokta detaysız yüzeye indirilir, far haritasındaki uzaklık yüzeydeki
        # gradyanıyla bölünür. Eskiden (u, z) prizmasındaki 0.25 mm'lik bant, farın üst kenarında neredeyse yatay
        # yüzeyi yalayarak kesilip ince bir yarık oluyor, ızgarada noktalı (- - -) delikler bırakıyordu.
        # (kenarları ~0.14 mm yuvarlatılır: 0.08 mm'lik keskin dudak ızgarada testere dişli bir parlak çizgi bırakıyordu)
        R = p(0.14)
        dd = smax(dd, -smax(-(z0 + p(0.7)), np.maximum(-ms - p(0.35), ms), R), R)
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
        # köşe girişleri (OPC fren soğutma)
        A = KOSE_AGIZ(Y, Z)
        # heykelsi oyuk: ağzın üst-dış yanında tampon yüzeyi içe çekilir (en çok 0.25 mm, ağız dudağında);
        # üst sınırı KOSE_KIRIM boyunca eğim kırığı -> ağzın tepesinden çamurluğa yükselen keskin kırım.
        # (0.42 mm ve 35-45 mm'lik yan geçişlerle ağzın yanında belirgin bir çukur gibi okunuyordu: derinlik
        # azaltıldı, yan/alt geçişler genişletildi; kırım aynı)
        c = np.clip((KOSE_KIRIM(np.clip(Y, 700.0, 910.0)) - Z) / 75.0, 0.0, 1.0)
        f = (p(0.25) * (1 - (1 - c) ** 2) * (1 - sstep(0.0, 210.0, A)) * sstep(690.0, 790.0, Y)
             * (1 - sstep(845.0, 925.0, Y)) * sstep(235.0, 340.0, Z) * (1 - sstep(450.0, 545.0, X)))
        dd = dd + f
        zs = z0 + f
        # derin, karanlık boşluk (dudak dik kenarlı); x<385: ön teker kuyusuna en az ~1.1 mm et kalır
        on_k = X - 385.0
        dd = oyuk(dd, zs, np.maximum(A, on_k), p(2.1))
        # saten plaka: kenarda 0.25 mm basamak, kanala doğru çan ağzı gibi 1.3 mm'ye iner
        kd = KOSE_KANAL(Y, Z)
        pm = np.maximum(np.maximum(KOSE_PLAKA(Y, Z), -kd), np.maximum(A, on))
        dd = doldur(dd, zs, pm, -(p(0.25) + p(1.05) * kose_t(Y, Z) ** 1.7))
        # siyah kanal çerçevesi (0.38 mm et, yüzü -0.9 mm: plakadan ~0.4 mm, boşluktan 1.2 mm çıkıntılı) + içeride
        # (-1.95 mm) petek ağ (et 0.26, delik 0.36 mm), arkası -2.6 mm
        ic = kd + p(0.38)
        dd = doldur(dd, zs, np.maximum(np.maximum(kd, -ic), on), -p(0.9))
        dd = oyuk(dd, zs, np.maximum(ic, on_k), p(2.6))
        dd = doldur(dd, zs, np.maximum(np.maximum(petek(Y, Z, p(0.36)) - p(0.13), ic), on), -p(1.95))
        # çeki kancası kapağı yalnız sağda (+Y): asimetrik()
        # splitter dudağı: tampon yüzü aşağıya doğru giderek öne şişer ve ON_DUDAK_ALT'ta keskin bir alt kenarla
        # biter (fotoğrafta en alttaki öne bakan kenar; üst yüzü yüze teğet akar, raf/basamak yok). Gövdenin alt yüzü
        # bu kenarın hemen altına çekildiğinden (govde_temel z_alt) dudağın altında ikinci bir şişkinlik kalmaz.
        # Dudak köşeyi döner ve yanda teker kemerine doğru (x 440-530) söner; eskiden x 420'de düzlemle kesilip
        # köşede küt bir uç yüzü bırakıyordu.
        YD = p(0.15)
        zb = ON_DUDAK_ALT(np.abs(Y))
        H = 42.0
        t = np.clip((Z - zb) / H, 0.0, 1.0)
        tas = p(0.7) * (1.0 - t) ** 2 * (1 - sstep(440.0, 530.0, X)) * (1 - sstep(880.0, 940.0, np.abs(Y)))
        egim = p(0.7) * 2.0 * (1.0 - t) / H
        # taban: köşe girişinin yanındaki heykelsi oyuklu yüzey (zs); z0 alınınca bandın üstünde oyuk kesilip basamak kalıyordu
        dudak = smax((zs - tas) / np.sqrt(1.0 + egim ** 2), zb - Z, YD)
        # yalnız şişkinlik bandında (üstte tas = 0: yüzeye teğet); bandın üstündeki oyuklar (ızgara, köşe girişi) dolmaz
        dudak = smax(dudak, np.maximum(Z - (zb + H), X - 560.0), YD)
        dd = smin(dd, dudak, YD)
        return dd
    uygula((-60, 560, 0, 960, 150, 620), on_tampon)

    # ---------- far yıkama kapağı: farın altında, köşede yuvarlatılmış eğik paralelkenar (ince panel çizgisi).
    # Hat kalibre ön çapraz fotoğraftan (15922355098) yüzeye geri izdüşümle, far haritasında (u, z) okundu:
    # üst kenar dışa doğru hafif yükselir, iç yan ~50°, dış yan ~60° eğik; iç üst köşe geniş yuvarlak.
    # Kendi kutusunda (eskiden on_tampon kutusu z 620'de kesip çizginin üst yarısını siliyordu). Kesit U (kenarı
    # yüzeye teğet): ızgarada kesik kesik çıkmaz.
    def far_yikama(dd, z0, X, Y, Z):
        s = FAR_YIKAMA(far_u(X, Y), Z)
        w = p(0.2)
        q = np.clip(1.0 - (s / w) ** 2, 0.0, 1.0)
        return oyuk(dd, z0, np.maximum(np.abs(s) - w, X - 400.0), p(0.3) * q * np.sqrt(q))
    uygula((100, 360, 560, 820, 540, 665), far_yikama)

    # ---------- yan: kapı aralıkları, kol
    def kapilar(dd, z0, X, Y, Z):
        # kapı aralığı (ilk üç çizgi): U kesitli oluk, kenarı yüzeye teğet iner (dik duvarlı oluk eğik
        # kesimlerde ızgarada testere dişi bırakıyordu)
        # kapı aralıkları yumuşatılmış eğrilerle (KAPI_ON, KAPI_ARKA: köşelerde dirsek yok). y-düzlemi kırpması
        # yerine oluk derinliği yüzey normaline göre söner (yüzey yukarı döndükçe): eski sınır A direği dibinde
        # oluğun tabanını düz kesip tırtıklı, kör uçlu bırakıyordu. Kesit (1-q^2)^1.5: kenarda yüzeye teğet.
        Xb, Zb, ny = yuzey_izdusum(z0, X, Y, Z, normal=True)
        # marşpiyel basamağının yakınında (z < ~420) izdüşüm kullanılmaz: basamak yüzeyinde gradyan sapıp oluk maskesini
        # gövdenin içine taşıyor, kapı arka-alt köşesinde (x 2661-2729) kopuk kıymıklar/boşluklar bırakıyordu
        wi = sstep(380.0, 460.0, Z)
        Xb, Zb = X + wi * (Xb - X), Z + wi * (Zb - Z)
        mk = np.minimum(sd_cizgi(Xb, Zb, KAPI_ON), sd_cizgi(Xb, Zb, KAPI_ARKA))
        w = KAPI_W
        q = np.clip(1.0 - (mk / w) ** 2, 0.0, 1.0)
        # ön ([3]: on_dikis) ve arka ([4]: arka_dikisler) tampon dikişleri yüzey normaline göre ayrıca işlenir
        return oyuk(dd, z0, mk - w, p(0.55) * q * np.sqrt(q) * sstep(0.2, 0.4, ny))
    uygula((300, 4200, 480, 960, 250, 1480), kapilar)

    # kapı kolu: oval cep (136x56, 0.6 mm derin) üstünde dışa taşan tutamak (200x30, 12 mm / 0.5 mm çıkıntı,
    # yuvarlak kesitli, arkaya doğru hafif kalınlaşır). Kilit göbeği yalnız sürücü kapısında: asimetrik().
    def kol(dd, z0, X, Y, Z):
        dis = 700 - Y
        # cep tutamağın arkasında neredeyse eş merkezli: üstünde ve altında ~10-13 mm görünür (iki fotoğraf)
        cep = np.sqrt(((X - 2575.0) / 68.0) ** 2 + ((Z - (KOL_Z - 6.0)) / 28.0) ** 2) - 1.0
        dd = oyuk(dd, z0, np.maximum(cep * 28.0, dis), p(0.6) * sstep(0.0, 0.25, -cep))
        k, zc, hk = kol_tutamak(X, Z)
        return doldur(dd, z0, np.maximum(k, dis), hk)
    uygula((2440, 2720, 700, 960, 760, 870), kol)

    # yan sinyal (ön çamurluk): 0.25 mm kabarık çerçeve (4 mm), içinde 0.15 mm çukur lens + ince dikey bölme
    def yan_sinyal(dd, z0, X, Y, Z):
        Xb, Zb = yuzey_izdusum(z0, X, Y, Z)
        s_ = YAN_SINYAL(Xb, Zb)
        h = p(0.25) * sstep(0.0, 3.5, -s_) - p(0.15) * sstep(4.5, 6.0, -s_)
        h = h + p(0.06) * np.maximum(0.0, 1.0 - np.abs(Xb - 1325.0) / 3.0) * sstep(5.2, 7.0, -s_)
        return doldur(dd, z0, np.maximum(s_, 760.0 - Y), h)
    uygula((1270, 1370, 760, 960, 685, 740), yan_sinyal)

    def yumusak_islemler(YUM):
        """oyuk/doldur'un kenarı YUM ile yuvarlatılmış halleri (farlardaki gibi): ince detay kenarları eğik geçtiği
        yerlerde ızgarada testere dişi/kesik çizgi bırakmaz, baskıda yine keskin kalır."""
        def oy(dd, z0, maske, derinlik):
            return smax(dd, -smax(-(z0 + derinlik), maske, YUM), YUM)

        def dol(dd, z0, maske, seviye):
            return smin(dd, smax(z0 - seviye, maske, YUM), YUM)
        return oy, dol

    # ---------- arka aralıklar: bagaj kapağı "gülümsemesi" (arka()), D direği yan aralığı ve arka tampon dikişi.
    # Üçü aynı kesit: 0.44 mm genişlik, 0.55 mm derin U oluk (kenar yüzeye teğet iner: ızgarada testere dişi yok).
    # Stop tarafında lens-gövde oluğunun ortasında (st = -0.15 mm), cam tarafında cam kenarında biterler: ne kısa
    # kalır ne de lens/cam üstüne taşar.
    ARALIK_W, ARALIK_D = p(0.22), p(0.55)

    def aralik_oluk(dd, z0, mesafe, sinir, olcek=1.0):
        return oyuk(dd, z0, np.maximum(mesafe - ARALIK_W, sinir),
                    ARALIK_D * olcek * np.clip(1.0 - (mesafe / ARALIK_W) ** 2, 0.0, 1.0))

    # ön tampon - çamurluk dikişi (KAPI_CIZGILERI[3]): arka dikişlerle aynı kesit, 3B yüzey eğrisine teğet uzaklık.
    # Farın dış alt köşesinde cam kenarı oluğunun ortasında (far hattının 4 mm içi) başlar: dikiş far çevresindeki
    # gölge oluğuna basamaksız, kopuksuz bağlanır; alt uçta kemerin kıvrık dudağının sırtında söner (arkadaki gibi).
    # Eskiden Y prizmasıydı: köşeye sarılan kısımda yüzey öne döndükçe oluk genişleyip sığlaşıyordu.
    def on_dikis(dd, z0, X, Y, Z):
        sec = np.abs(z0) < 30.0
        if not sec.any():
            return dd
        X, Y, Z = (np.broadcast_to(a, z0.shape)[sec] for a in (X, Y, Z))
        z0s = z0[sec]
        d3 = cizgi3(X, Y, Z, on_dikis_3b())
        mes = np.sqrt(np.maximum(d3 * d3 - z0s * z0s, 0.0))
        fm = np.maximum(FAR_UZ(far_u(X, Y), Z), FAR_YAN(X, Z))
        r_k = np.sqrt((X - AKS_ON) ** 2 + (Z - AKS_Z) ** 2)
        dd = dd.copy()
        dd[sec] = aralik_oluk(dd[sec], z0s, mes, -(fm + 4.0), sstep(CAMUR_R + 6.0, CAMUR_R + 26.0, r_k))
        return dd
    uygula((330, 720, 640, 960, 540, 790), on_dikis)

    def stop_maske(X, Y, Z):
        return STOP(stop_u(X, Y), Z)

    def arka_dikisler(dd, z0, X, Y, Z):
        # 3B çizgiye uzaklığın yüzeye teğet bileşeni (normal bileşen ~ z0): oluk genişliği yüzey eğiminden bağımsız.
        # Yalnız yüzeyin ±30 mm'sindeki noktalarda hesaplanır.
        sec = np.abs(z0) < 30.0
        if not sec.any():
            return dd
        X, Y, Z = (np.broadcast_to(a, z0.shape)[sec] for a in (X, Y, Z))
        z0s, out = z0[sec], dd[sec]
        st_sinir = -(stop_maske(X, Y, Z) + p(0.15))
        # tampon dikişi kemerin kıvrık dudağının üstünden geçerken sığlaşarak dudak sırtında (r ≈ CAMUR_R + 7) söner:
        # tam derinlikte dudağı kesince kemer kenarında kopuk kıymıklar kalıyordu
        r_k = np.sqrt((X - AKS_ARKA) ** 2 + (Z - AKS_Z) ** 2)
        for pts, ek, olcek in ((bagaj_yan_3b(), -arka_cam_maske(X, Y, Z), 1.0),
                               (tampon_dikis_3b(), None, sstep(CAMUR_R + 6.0, CAMUR_R + 26.0, r_k))):
            d3 = cizgi3(X, Y, Z, pts)
            mes = np.sqrt(np.maximum(d3 * d3 - z0s * z0s, 0.0))
            sinir = st_sinir if ek is None else np.maximum(st_sinir, ek)
            out = aralik_oluk(out, z0s, mes, sinir, olcek)
        dd = dd.copy()
        dd[sec] = out
        return dd
    uygula((3860, 4200, 520, 960, 560, 1140), arka_dikisler)

    # ---------- arka
    def arka(dd, z0, X, Y, Z):
        oyuk, doldur = yumusak_islemler(p(0.08))
        ay = np.abs(Y)
        # bagaj kapağı aralığı: plaka altında "gülümseme" yayı (orta z 712), stobun alt kenarına çıkar,
        # stobun içinden (iç/dış parça ayrımı) yukarı geçer, cam köşesine ve D direği boyunca spoylere çıkar
        # yay stop haritasındaki U koordinatında çizilir ve stobun iç/dış birim ayrımının (STOP_ARALIK) alt ucunda
        # biter: eskiden y ile çizildiği için x<4400'de U = y + (4400 - x) farkı kadar (~20 mm) ayrımın dışında
        # stoba değiyordu. Uç stobun lens-gövde oluğunun ortasında kesilir.
        u0, v0 = STOP_ARALIK[0]
        uc_ = (u0, float(STOP_ALT(u0) + v0 * (STOP_UST(u0) - STOP_ALT(u0))))
        # (beyaz OPC 8366545858 geri izdüşümü: U 245 z 721, 316 728, 387 739, 433 751, 480 767, lambaya U 526'da varır)
        bagaj = sd_cizgi(stop_u(X, Y), Z, [(0, 712), (120, 715), (245, 721), (316, 728), (387, 739), (433, 751),
                                           (480, 767), uc_, (uc_[0] + 8.0, uc_[1] + 12.0)])
        dd = aralik_oluk(dd, z0, bagaj, np.maximum(4200 - X, -(stop_maske(X, Y, Z) + p(0.15))))
        # amblem altında aralığın alt kenarı çentik gibi aşağı genişler (beyaz OPC 8366545858)
        dd = oyuk(dd, z0, np.maximum(GULUMSEME_CENTIK(Y, Z), 4380 - X), ARALIK_D)
        # rozet: amblem asimetrik() içindeki opel_amblem() ile işlenir (halka Ø110, merkez y 0, z 853)
        on = 4250 - X
        # tampon kırımı: KIRIM_Z altında yüz KIRIM_D geride (yüzey normali boyunca); alttaki tüm detaylar (plaka,
        # reflektör, sensör, difüzör, egzoz) yeni yüze göre işlenir (z0 kaydırılır). Difüzörün altında (z < 200)
        # ve köşede (|y| 700-780) söner.
        z0u = z0
        kz = KIRIM_Z(ay)
        kd = KIRIM_D * (1.0 - sstep(kz - 3.5, kz + 3.5, Z)) * (1.0 - sstep(700.0, 780.0, ay)) * sstep(150.0, 200.0, Z)
        dd = np.where(kd > 1e-6, np.maximum(dd, z0 + kd), dd)
        z0 = z0 + kd
        # plaka lambası yarıkları: kırım basamağında (üst yüze göre)
        for yc in (120.0,):
            pl = sd_kutu2(ay, Z, yc, kz + 1.0, 48, 4.5, 3)
            dd = oyuk(dd, z0u, np.maximum(pl, on), p(0.75))
        # plaka yuvası: yamuk, tampona gömük, kenarları pahlı
        yuva = PLAKA_YUVA(Y, Z)
        dd = oyuk(dd, z0, np.maximum(yuva, on), p(0.3) + p(0.25) * sstep(-p(0.6), 0.0, yuva))
        # plaka: kırımın hemen altında, yuvanın üstünü kapatır (520 mm AB plakası yuvadan geniş, alt yüze oturur);
        # yüzden 0.05 mm kabarık, kabartma kenar çerçevesi, solda AB şeridi, 2 vida başı
        # (plakasız seçenekte yuvada iki koyu tapa kalır; "GTC" kabartması asimetrik() içinde)
        vidalar = (-135.0, 135.0)
        if ARKA_PLAKA:
            plaka = sd_kutu2(Y, Z, 0, 538, 260, 55, 8)
            dd = doldur(dd, z0, np.maximum(plaka, on), p(0.05))
            dd = doldur(dd, z0, np.maximum(np.abs(plaka + p(0.25)) - p(0.13), on), p(0.17))
            dd = doldur(dd, z0, np.maximum(sd_kutu2(Y, Z, -232, 538, 17, 44, 4), on), p(0.12))
            for yc in vidalar:
                vd = np.sqrt((Y - yc) ** 2 + (Z - 538) ** 2)
                dd = doldur(dd, z0, np.maximum(vd - p(0.32), on), p(0.05) + p(0.15) * np.sqrt(np.clip(1 - (vd / p(0.32)) ** 2, 0, 1)))
        else:
            for yc, zc in PLAKA_TAPA:
                dd = oyuk(dd, z0, np.maximum(np.sqrt((Y - yc) ** 2 + (Z - zc) ** 2) - 10.0, on), p(0.6))
        # reflektörler: çukur yuva, içinde lens ve dikey prizma kaburgaları
        rf = REFLEKTOR(ay, Z)
        dd = oyuk(dd, z0, np.maximum(rf, on), p(0.45))
        dd = doldur(dd, z0, np.maximum(rf + p(0.14), on), -p(0.2))
        dd = doldur(dd, z0, np.maximum(rf + p(0.32), on), -p(0.12))                       # lensin ortası kabarık
        # park sensörleri: ince oluklu halka, ortası hafif kubbeli kapak
        for yc, zc in SENSOR:
            rs = np.sqrt((ay - yc) ** 2 + (Z - zc) ** 2)
            dd = oyuk(dd, z0, np.maximum(np.abs(rs - 11.5) - p(0.13), on), p(0.35))
            dd = doldur(dd, z0, np.maximum(rs - 8.0, on), p(0.06) * (1 - (rs / 8.0) ** 2))
        # OPC difüzör: tamponun altında geri çekilmiş siyah parça; üst kenarda tampon dudağı gölge veren çıkıntı
        dfz = DIFUZOR(Y, Z)
        dif = np.maximum(dfz, on)
        dd = oyuk(dd, z0, dif, p(0.75) + p(0.45) * sstep(300.0, 360.0, Z))
        dd = oyuk(dd, z0, np.maximum(np.abs(dfz + p(0.12)) - p(0.12), on), p(1.6))       # üst/yan kenar gölge oluğu
        # dikey difüzör kanatları: tampon yüzeyine kadar çıkan 0.45 mm plakalar; alt kenarları yatay (z DIF_KANAT_ALT),
        # difüzör rampasının altında üçgen plaka olarak sarkar (beyaz OPC tam arka: kanat uçları z≈200'de biter)
        for yc in DIF_KANAT:
            kn = np.abs(ay - yc) - p(0.225)
            dd = doldur(dd, z0, np.maximum(kn, np.maximum(dfz + p(0.3), np.maximum(Z - 352.0, DIF_KANAT_ALT - Z))), -p(0.08))
            alt_ = np.maximum(kn, np.maximum(np.maximum(DIF_KANAT_ALT - Z, Z - 352.0),
                                             np.maximum(4240.0 - X, X - (4446.0 - 0.55 * np.maximum(0.0, 300.0 - Z)))))
            dd = smin(dd, alt_, 4.0)
        # trapez egzoz uçları: dışa taşan krom çerçeve (dış pah + iç basamak), ≥3 mm derin ağız, içinde yuvarlak boru
        egs = EGZOZ(ay, Z)
        eg = np.maximum(egs, on)
        dd = doldur(dd, z0, np.maximum(np.maximum(egs, -egs - p(0.5)), on), p(0.1) - p(0.2) * sstep(-p(0.12), 0.0, egs))
        dd = oyuk(dd, z0, np.maximum(np.maximum(egs + p(0.5), -egs - p(0.72)), on), p(0.35))   # iç basamak
        dd = oyuk(dd, z0, eg + p(0.72), p(3.4))                                                # derin ağız
        return dd
    uygula((4100, 4520, 0, 960, 130, 1150), arka)

    # ---------- arka stoplar: köşeyi saran iki parçalı lamba (iç: bagaj kapağında, dış: çamurlukta)
    #  seviyeler (baskı mm, gövde yüzeyine göre): lens -0.45 | dış kırmızı çerçeve -0.28 | odalar -0.85
    #  krom bıçak -0.06 | LED kılavuzları -0.10 (kubbeli) | ince kılavuzlar -0.22 | kenar oluğu -1.1
    _cer = YPoligon(stop_uv(STOP_CERCEVE + [(1300, -0.05), (1300, 1.05)]), 4)
    _oda_ic = YPoligon(stop_uv(STOP_ODA_IC + [STOP_ODA_IC[0]])[:-1], 4)
    _oda_dis = YPoligon(stop_uv(STOP_ODA_DIS + [STOP_ODA_DIS[0]])[:-1], 4)
    _krom, _led_ic, _led_dis = stop_uv(STOP_KROM), stop_uv(STOP_LED_IC), stop_uv(STOP_LED_DIS)
    _kan1, _kan2 = stop_uv(STOP_KANAT[0]), stop_uv(STOP_KANAT[1])
    _cer_c, _aralik, _led_alt = stop_uv(STOP_CERCEVE[1:]), stop_uv(STOP_ARALIK), stop_uv(STOP_LED_ALT)

    def stoplar(dd, z0, X, Y, Z):
        oyuk, doldur = yumusak_islemler(p(0.08))
        U = stop_u(X, Y)
        st = np.maximum(STOP(U, Z), 3880.0 - X)
        dd = oyuk(dd, z0, st, p(0.45) * sstep(-0.5, 4.0, -st))                   # lens yüzeyi (kenarda 4 mm pah)
        dd = doldur(dd, z0, np.maximum(_cer(U, Z), st + p(0.3)), -p(0.28))       # dış (yan) kırmızı çerçeve kademesi
        dd = doldur(dd, z0, np.maximum(sd_cizgi(U, Z, _cer_c) - p(0.15), st + p(0.25)), -p(0.12))  # iç çerçeve dudağı
        oda = np.maximum(np.minimum(_oda_ic(U, Z), _oda_dis(U, Z)), st + p(0.2))
        dd = oyuk(dd, z0, oda, p(0.85))                                          # geri vites / sinyal odaları
        q = np.mod(U - 400.0, 46.0)
        kab = np.maximum(np.minimum(q, 46.0 - q) - p(0.14), oda + p(0.15))       # oda içi reflektör kaburgaları
        dd = doldur(dd, z0, kab, -p(0.6))
        dd = doldur(dd, z0, np.maximum(sd_cizgi(U, Z, _krom) - p(0.21), st + p(0.1)), -p(0.06))   # krom bıçak
        for led in (_led_ic, _led_dis):                                          # LED ışık kılavuzları (kubbeli)
            s_ = sd_cizgi(U, Z, led)
            dd = doldur(dd, z0, np.maximum(s_ - p(0.26), st + p(0.1)), -p(0.22))
            dd = doldur(dd, z0, np.maximum(s_ - p(0.13), st + p(0.1)), -p(0.08))
        ince = np.minimum(sd_cizgi(U, Z, _kan1), sd_cizgi(U, Z, _kan2))
        ince = np.minimum(ince, sd_cizgi(U, Z, _led_alt))
        dd = doldur(dd, z0, np.maximum(ince - p(0.14), st + p(0.1)), -p(0.2))     # ince kılavuz çizgileri
        # lens-gövde oluğu: gövde kenarından 5 mm'lik pahla iner (dik kenar eğik geçişlerde testere dişi bırakıyordu)
        dd = oyuk(dd, z0, np.maximum(np.abs(st + p(0.15)) - p(0.14), 3880.0 - X), p(1.1) * sstep(-0.5, 5.0, -st))
        # iç/dış birim aralığı: lens-gövde oluğunun ortasında biter (eskiden 0.05 mm dışına taşıp stobun üst
        # kenarında çentik bırakıyordu); alt ucu bagaj kapağı yayının ucuyla aynı noktada
        dd = oyuk(dd, z0, np.maximum(sd_cizgi(U, Z, _aralik) - p(0.15), st + p(0.15)), p(1.2))
        return dd
    uygula((3880, 4520, 300, 960, 745, 1000), stoplar)

    # ---------- 3. stop lambası: spoyler arka yüzünde ince yatay şerit
    def ucuncu_stop(dd, z0, X, Y, Z):
        # spoyler arka yüzünde çukur lens (280x20), içinde 9 kabarık LED noktası. Yalnız arka yüzün ~15 mm
        # yakınında (spoyler arka kenarı burada x ≈ 4016): eskiden x 3985'ten itibaren kesip spoylerin alt yüzüne
        # yakın, gövde içinde kapalı bir boşluk bırakıyordu
        on = 4002 - X
        m = np.maximum(sd_kutu2(Y, Z, 0, 1291, 140, 10, 6), on)
        dd = oyuk(dd, z0, m, p(0.35))
        q = np.mod(Y + 140.0, 31.0)
        led = np.sqrt((q - 15.5) ** 2 + (Z - 1291) ** 2) - p(0.2)
        dd = doldur(dd, z0, np.maximum(led, np.maximum(m + p(0.08), on)), -p(0.15))
        return dd
    uygula((3940, 4080, 0, 200, 1260, 1330), ucuncu_stop)

    # ---------- köpekbalığı anten (26608068581): tavan ortasında, spoylerin ~60 mm önünde. Yandan: önde alçak
    # gövde (burnu yuvarlak, ~20 mm), üstünden ~33° (dikeyden ~57°) geriye yatık hücum kenarıyla yükselen yüzgeç,
    # tepe tavandan 65 mm, arka kenar neredeyse dikey. Kesit tabanda 55 mm, ortada ~25, tepede ~10 mm'ye incelir.
    def anten(dd, z0, X, Y, Z):
        H = Z - (Z_KABIN(X) - 40.0 * (Y / 650.0) ** 2)        # tavandan yükseklik
        prof = ANTEN_PROFIL(X, H)
        t = np.clip(H / 66.0, 0.0, 1.0)
        yari = (5.0 + 22.5 * (1.0 - t) ** 1.6) * np.sqrt(np.clip((X - 3574.0) / 45.0, 0.05, 1.0)) \
            * (1.0 - 0.15 * sstep(3690.0, 3735.0, X))
        g = smax(prof, (np.abs(Y) - yari) * 0.85, 3.0)
        return smin(dd, g, p(0.12))
    uygula((3555, 3755, 0, 45, 1330, 1470), anten)

    # ---------- aynalar: uzun damla gövde + ince kısa ayak (Astra J: sinyalsiz kapak, alt kısmı siyah taban)
    # Plan: iç ucu |y| 830'da en uzun (x boyu ~160, burun), dış uca doğru kısalır (~120) ve dış uçta yarım daireye yakın
    # kapanır (|y| 1010 -> aynalarla genişlik 2020). Yandan uzun damla: arkada düz cam yüzü (x 1745), üst çizgi buruna doğru
    # alçalır, alt çizgi hafif yükselir; burun orta-alt yükseklikte (siyah OPC 34848925543 ve gri 54188905081
    # fotoğrafları; eski kapak kısa, üstü düz bir kapsüldü). Yükseklik bele göre (cam alt kenarı önde z ≈ 930): kapak
    # üstü iç uçta ~1020, dış uçta ~1062, alt (siyah taban) 955 -> 930. Üst kenar eskiden 35 mm daha yüksekti (1055-1100):
    # iki yan fotoğraf burada çelişir (gri 54188905081 aynayı/krom ucunu siyah OPC'den 60-80 mm yüksek gösterir); üçüncü
    # kanıt kalibre ön çapraz (15922355098) kapağın üst siluetini 30 mm, siyah OPC 45-50 mm alçak gösterir, alt siluet
    # ikisinde de 0-6 mm içinde: üst 35-38 mm indirildi, alt aynı. Cam yüzü D biçimli: iç kenar düz, dış uç yuvarlak.
    # Cam 0.45 mm gömük, çevresinde
    # 7 mm kapak dudağı. Siyah alt taban ayrım çizgisi (oluk + tabanda 0.08 mm içe basamak).
    AY_X, AY_Y0, AY_Y1 = 1745.0, 830.0, 1008.0

    def ay_yuz(Y):
        # cam yüzü neredeyse enine (dış ucu ~3° önde): gri fotoğrafta arkadan cam yüzü geniş D olarak görünür (eski
        # ~10° içe dönük yüz orada %40 dar kalıyordu)
        return AY_X - 0.05 * (Y - 920.0)

    # dış uç: kesit AY_YE'den sonra küçülmez, süper elips kesit Y yönünde elipsle (AY_UC) kapanır -> yuvarlak,
    # pürüzsüz uç (D biçimli cam yüzünün yuvarlak tarafı)
    AY_YE = AY_Y0 + 0.58 * (AY_Y1 - AY_Y0)
    AY_UC = AY_Y1 - AY_YE
    AY_US = 2.6                                                  # kesit üssü (eski kapsülle aynı; şekli damla profili verir)

    def ayna_kesit(Y):
        tau = np.clip((np.minimum(Y, AY_YE) - AY_Y0) / (AY_YE - AY_Y0), 0.0, 1.0)
        # x boyu (cam yüzünden öne): burun iç uçta (kapıya yakın), dış uca doğru kısalır -> önden ve arkadan bakışta
        # burun içe-öne bakan damla (gri fotoğrafta kapak dış uçta cam yüzünün yalnız ~20 mm dışına taşar)
        boy = 160.0 - 40.0 * tau
        e = 1.0 - (1.0 - tau) ** 2                             # D yüz: üst/alt kenar dış uca doğru kavisli
        zlo, zhi = 955.0 - 25.0 * e, 1020.0 + 42.0 * e
        return boy, zlo, zhi

    def ayna_govde(X, Y, Z, duz=True):
        boy, zlo, zhi = ayna_kesit(Y)
        a = boy
        yz = ay_yuz(Y)
        u = np.maximum(yz - X, 0.0) / a
        # damla: üst çizgi buruna doğru kesit yüksekliğinin %24'ü kadar alçalır (arka yarıda düz), alt çizgi %5 yükselir
        hb = zhi - zlo
        ust = zhi - 0.24 * hb * np.minimum(u, 1.0) ** 2.4
        alt = zlo + 0.05 * hb * np.minimum(u, 1.0) ** 2
        b, zc = 0.5 * (ust - alt), 0.5 * (ust + alt)
        v = np.abs(Z - zc) / b
        w = np.maximum(Y - AY_YE, 0.0) / AY_UC
        f = np.sqrt((u ** AY_US + v ** AY_US) ** (2.0 / AY_US) + w * w) - 1.0
        g = f * np.minimum(np.minimum(a, 0.5 * hb), AY_UC)
        if duz:
            g = smax(g, X - yz, 6.0)                             # düz cam yüzü
        return smax(g, AY_Y0 - Y, 30.0)                          # iç uç (geniş yuvarlak: cam yüzü köşeleri pahlı değil)

    # cam yüzündeki kapak dış hattı (y, z) ve 7 mm (0.29 mm) eşit kapak dudağıyla cam hattı: kapak kesitinden
    # gerçek 2B uzaklık. İç köşeler geniş yarıçaplı (22 mm): cam yuvası eskiden köşeli dikdörtgen gibiydi
    _yy = np.linspace(AY_Y0 + 4.0, AY_Y1, 64)
    _b, _zl, _zh = ayna_kesit(_yy)
    _zc, _h = 0.5 * (_zl + _zh), 0.5 * (_zh - _zl) * np.sqrt(np.clip(1.0 - (np.maximum(_yy - AY_YE, 0.0) / AY_UC) ** 2,
                                                                       0.0, 1.0))
    AYNA_YUZ = YPoligon([(a_, c_ + h_) for a_, c_, h_ in zip(_yy, _zc, _h)][:-1]
                        + [(a_, c_ - h_) for a_, c_, h_ in zip(_yy[::-1], _zc[::-1], _h[::-1])], 22.0)

    def ayna(dd, z0, X, Y, Z):
        # yaklaşık alanı noktadaki gradyanla ölçekle (uç kapakta basamaklanma olmaz)
        g = ayna_govde(X, Y, Z)
        gx = ayna_govde(X + 0.5, Y, Z) - ayna_govde(X - 0.5, Y, Z)
        gy = ayna_govde(X, Y + 0.5, Z) - ayna_govde(X, Y - 0.5, Z)
        gz = ayna_govde(X, Y, Z + 0.5) - ayna_govde(X, Y, Z - 0.5)
        g = g / np.clip(np.sqrt(gx * gx + gy * gy + gz * gz), 0.3, 3.0)
        # siyah alt taban: ayrım çizgisinin altı 0.08 mm içeride, çizgide 0.5 mm genişlik oluk (cam yüzünde yok)
        zs = 968.0 - 8.0 * np.clip((Y - AY_Y0) / (AY_Y1 - AY_Y0), 0.0, 1.0)
        yz = ay_yuz(Y)
        # cam: kapak dış hattından 7 mm içeride, 0.45 mm gömük; kenarı 3 mm pahla iner (dudak kenarı temiz)
        c2 = AYNA_YUZ(Y, Z) + 7.0
        cam = np.maximum(c2, yz - 14.0 - X)
        kapak_yuzu = -cam                                        # cam yuvası dışı
        g = g + p(0.08) * (1.0 - sstep(zs - 1.5, zs + 1.5, Z)) * sstep(0.0, 3.0, cam)
        g = oyuk(g, g.copy(), np.maximum(np.abs(Z - zs) - p(0.25), kapak_yuzu), p(0.22))
        g = np.maximum(g, -np.maximum(cam, (yz - p(0.45) * sstep(0.0, 3.0, -c2)) - X))   # düz cam tabanı
        # ayak: siyah ayraç yelkeninin alt kısmından (|y| 770, cam yüzeyinin içinden) kapağın iç alt ucuna; ince
        # (x 1652-1702, z 948-990), dışa doğru önden incelir. Eskiden 85 mm boyunda kutu gibi kalın bir ayaktı.
        ty = np.clip((Y - 770.0) / 70.0, 0.0, 1.0)
        kama = smax(smax(1652.0 + 18.0 * ty - X, X - 1702.0, 8.0),
                    smax(smax(948.0 - Z, Z - (990.0 - 6.0 * ty), 8.0), np.maximum(770.0 - Y, Y - 845.0), 6.0), 8.0)
        kama = smax(kama, (X - 1702.0) - 0.6 * (Z - 948.0) + 8.0, 6.0)   # arka alt köşe pahlı
        g = smin(g, kama, 10.0)
        dd = smin(dd, g, p(0.3))
        # ayak taban plakası (siyah yelken): ayraçla aynı düz parça, cam düzleminden 0.1 mm kabarık; bel çizgisinin
        # altında gövdeye karışır
        yelken = np.maximum(YPoligon_AYNA(X, Z), 720.0 - Y)
        return doldur(dd, z0, yelken, -(CAM - p(0.1)))
    uygula((1500, 1800, 660, 1060, 900, 1130), ayna)
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

    # Opel amblemi: kabartma taban + çukur koyu zemin + kubbeli krom halka + pahlı, sırtlı şimşek.
    # Şimşek ayna simetrik değil (nokta simetrik), bu yüzden yarım hacimde değil burada işlenir.
    # u: bakana göre sağa, v: yukarı; ölçüler halka yarıçapı R biriminde.
    simsek = np.array([(-1.18, 0.03), (0.02, 0.30), (0.21, 0.06), (1.18, -0.03), (-0.02, -0.30), (-0.21, -0.06)])

    def yuzey_x(yc, zc, x0, x1, dis_yon):
        """Gövde temel yüzeyinin (detaysız) verilen (y, z) noktasındaki x konumu."""
        xs = np.arange(x0, x1, 0.5)
        dv = govde_temel(xs[:, None, None], np.array([[[abs(yc)]]]), np.array([[[zc]]]))[:, 0, 0]
        ic = np.nonzero(dv < 0)[0]
        if len(ic) == 0:
            return None
        return xs[ic[0]] if dis_yon < 0 else xs[ic[-1]]

    def opel_amblem(sl, X, U, V, R, xp, dis_yon):
        """Amblem düz bir montaj düzlemine (x = xp) oturur; dis_yon: -1 ön (dışarı -x), +1 arka (dışarı +x).
        Her katman düzlemden 'yükseklik' kadar dışarı çıkan ve gövdeye 1.5 mm gömülen bir kabartmadır."""
        r = np.sqrt(U * U + V * V)

        def kabartma(maske, yukseklik):
            dis = (X - (xp + dis_yon * yukseklik)) * dis_yon          # > 0: yüzün dışında
            ic = (xp - dis_yon * p(1.5) - X) * dis_yon                 # > 0: gömülme sınırının içinde
            e = np.maximum(maske, np.maximum(dis, ic))
            d[sl] = np.minimum(d[sl], e)

        # 1) taban: halkanın biraz dışına taşan yuvarlak kaide, kenarı pahlı
        taban = r - 1.10 * R
        kabartma(taban, p(0.25) * np.clip(-taban / (0.10 * R), 0.15, 1.0))
        # 2) zemin: halka içi hafif çukur (boyamada siyah)
        cukur = np.maximum(r - 0.84 * R, (X - (xp + dis_yon * p(0.12))) * -dis_yon)
        d[sl] = np.maximum(d[sl], -cukur)
        # 3) halka: yarı yuvarlak kesitli krom halka
        w = 0.13 * R
        kub = np.sqrt(np.clip(1.0 - ((r - 0.97 * R) / w) ** 2, 0.0, 1.0))
        kabartma(np.abs(r - 0.97 * R) - w, p(0.30) + p(0.22) * kub)
        # 4) şimşek: kenarları pahlı, ortası sırtlı, uçları halkanın dışına taşar
        m = sd_poligon(U / R, V / R, simsek) * R
        kabartma(m, p(0.30) + p(0.32) * np.sqrt(np.clip(-m / (0.075 * R), 0.0, 1.0)))

    g = kes(-40, 320, -90, 90, 620, 760)       # ön rozet: merkez (y 0, z 686), R 48; önden bakınca sağ = -Y
    if g:
        sl, X, Y, Z = g
        xp = yuzey_x(0.0, 686.0, -40, 320, -1)
        if xp is not None:
            opel_amblem(sl, X, -Y + 0 * X, Z - 686.0 + 0 * X, 48.0, xp - p(0.15), -1)
    # arka rozet: merkez (y 0, z 853), R 55; arkadan bakınca sağ = +Y. Beyaz OPC tam arka kalibre kamerasında halka
    # merkezi (729, 759) px -> z 853 (eskiden halkanın üst yarısındaki bir nokta merkez alınıp 876 yazılmıştı;
    # gülümseme z 712, plaka yuvası, stoplar aynı kamerayla örtüşür, amblem onlara göre 23 mm yüksek kalıyordu)
    g = kes(4280, 4520, -90, 90, 780, 940)
    if g:
        sl, X, Y, Z = g
        xp = yuzey_x(0.0, 853.0, 4200, 4520, +1)
        if xp is not None:
            opel_amblem(sl, X, Y + 0 * X, Z - 853.0 + 0 * X, 55.0, xp + p(0.10), +1)

    # arka silecek (OPC aero kol): göbek camın alt kenarında ortanın hemen sağında, kol ve süpürge sola (-Y)
    # yatık; kalın kol kapağı + altında ince lastik süpürge + kubbeli göbek kapağı
    g = kes(4200, 4440, -420, 120, 1010, 1120)
    if g:
        sl, X, Y, Z = g
        cur = d[sl]
        supurge = sd_cizgi(Y, Z, [(-348, 1049), (-200, 1050), (5, 1052)]) - p(0.26)
        kol = sd_cizgi(Y, Z, [(-255, 1061), (-120, 1062), (30, 1060)]) - p(0.3)
        gob = sd_cizgi(Y, Z, [(-5, 1058), (52, 1056)]) - p(0.5)
        e = np.minimum(np.maximum(cur - p(0.55), supurge), np.maximum(cur - p(0.85), kol))
        e = np.minimum(e, np.maximum(cur - p(0.2) - p(0.9) * np.sqrt(np.clip(-gob / p(0.5), 0, 1)), gob))
        d[sl] = np.minimum(cur, e)
    # arka cam fıskiyesi: spoyler arka yüzünde fren lambasının solunda küçük delik
    g = kes(3940, 4080, -230, -120, 1250, 1330)
    if g:
        sl, X, Y, Z = g
        cur = d[sl]
        fis = np.maximum(np.sqrt((Y + 177.0) ** 2 + (Z - 1292.0) ** 2) - p(0.3), 4000.0 - X)
        d[sl] = np.maximum(cur, -np.maximum(fis, -(cur + p(0.4))))
    # arka yazılar: "opc" rozeti (bagaj kapağı, +Y) ve plakasızsa yuvadaki "GTC"; çizgi yazı tipi (_FONT),
    # çizgi kalınlığı 0.27 mm, 0.3 mm kabartma (arkadan bakınca sağ = +Y)
    def kabartma_yazi(g, metin, h, sx, egim, bosluk, yc, zc, v_merkez, yari, yuk):
        sl, X, Y, Z = g
        _, gen = duz_yazi_2d(np.zeros(1), np.zeros(1), metin, h, sx, egim, bosluk)
        u = Y - (yc - gen / 2) + 0 * X
        v = Z - (zc - v_merkez * h) + 0 * X
        dy, _ = duz_yazi_2d(u, v, metin, h, sx, egim, bosluk)
        cur = d[sl]
        d[sl] = np.minimum(cur, np.maximum(cur - yuk, dy - yari))
    g = kes(4300, 4520, 300, 445, 735, 815)
    if g:
        kabartma_yazi(g, "opc", 28.0, 0.98, 0.0, 8.0, OPC_ROZET[0], OPC_ROZET[1], 0.21, p(0.135), p(0.3))
    if not ARKA_PLAKA:
        g = kes(4300, 4520, -100, 100, 500, 556)
        if g:
            kabartma_yazi(g, "GTC", 21.0, 3.65, 0.28, 9.5, GTC_YAZI[0], GTC_YAZI[1], 0.5, p(0.135), p(0.3))
    # çeki kancası kapağı (arka tampon, +Y tarafı): ince panel çizgisi
    g = kes(4300, 4520, 380, 540, 370, 500)
    if g:
        sl, X, Y, Z = g
        cur = d[sl]
        ck = sd_kutu2(Y, Z, 452.0, 435.0, 40.0, 38.0, 9.0)
        d[sl] = np.maximum(cur, -np.maximum(np.abs(ck) - p(0.13), -(cur + p(0.4))))

    # yakıt kapağı (sağ arka, +Y): yuvarlak Ø176, arka teker aksının arkasında, stobun sivri ucunun önünde.
    # Mavi OPC 26608068581 kalibre kamerasında kapak kenarından 8 noktanın gövdeye geri izdüşümü:
    # x 3740..3902, z 813..979, |y| 752..833; 3B daire uydurma -> merkez (3817, 792, 895), R 88.8 (artık ±4.5),
    # düzlem normali gövde normaliyle aynı (0.26, 0.89, 0.38). Kapak gövde yüzeyine teğet düzlemde yuvarlak
    # (yüzey normali boyunca silindir). Oluk analitik uzaklık alanıdır: (s, yüzeyden derinlik) düzleminde yuvarlak
    # dipli kapsül, gövdeden ağız kenarı yuvarlatılarak (smax) çıkarılır; kapak yüzünün hafif kubbesi oluğa teğet
    # söner. Eskiden kubbe 0.04 mm'lik dik basamakla başlıyor, oluk sqrt profiliyle dik iniyor ve ikisi sert
    # max/min ile birleşiyordu: ızgarada (1.92 mm) oluğun iki kenarı yakından testere dişli çıkıyordu.
    g = kes(3680, 3935, 640, 960, 790, 1015)
    if g:
        sl, X, Y, Z = g
        xc, zc, rk = 3817.0, 895.0, 88.0
        ys = np.arange(600.0, 960.0, 0.25)
        dv = govde_temel(np.array([[[xc]]]), ys[None, :, None], np.array([[[zc]]]))[0, :, 0]
        yc = float(ys[np.nonzero(dv < 0)[0][-1]])
        def gt(px, py, pz):
            return float(govde_temel(np.array([[[px]]]), np.array([[[py]]]), np.array([[[pz]]]))[0, 0, 0])
        nx, ny, nz = gt(xc + 1, yc, zc) - gt(xc - 1, yc, zc), gt(xc, yc + 1, zc) - gt(xc, yc - 1, zc), \
            gt(xc, yc, zc + 1) - gt(xc, yc, zc - 1)
        nn = math.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / nn, ny / nn, nz / nn
        dx, dy, dz = X - xc, Y - yc, Z - zc
        dn = dx * nx + dy * ny + dz * nz
        r = np.sqrt(np.maximum(dx * dx + dy * dy + dz * dz - dn * dn, 0.0))
        s = r - rk
        cur = d[sl]
        w, D = p(0.2), p(0.45)                                           # oluk yarı genişliği (0.4 mm aralık), derinlik
        yakin = 1.0 - sstep(30.0, 60.0, np.abs(dn))                      # yalnız yüzey civarı (yumuşak sönüm)
        kab = p(0.06) * (1.0 - sstep(0.25 * rk, rk - w, r))              # kapak yüzü: merkezde 0.06 mm, oluğa teğet söner
        e = cur - kab * yakin
        # oluk: (s, yüzeyden derinlik) düzleminde yuvarlak dipli U (kapsül) -> gerçek uzaklık alanı; ağız kenarları
        # 6 mm smax ile yuvarlatılır (ızgara adımının ~3 katı: dairesel kenar merdivenlenmez)
        t = -cur
        kaps = np.sqrt(s * s + np.maximum(t - (D - w), 0.0) ** 2) - w
        d[sl] = np.where(yakin > 0.0, smax(e, -kaps, 6.0), e)
    # silecekler (sol direksiyon, tandem, park konumu; 15922355098 / 26608054501 / 17022087756 fotoğrafları):
    # kaputun arka kenarının arkasındaki kanalda, cam tabanında yatarlar. Göbekler solda (-Y): sürücü kolu A
    # direği dibinden, yolcu kolu ortadan çıkar; iki süpürge de sağa (+Y) uzanır, sürücü süpürgesi (650) camda
    # biraz yukarıda, yolcu süpürgesi (450) altta. Süpürge: altta ince lastik dudak, üstünde aero kiriş (sırtlı),
    # ortada bağlantı klipsi. Kol: göbekten klipse incelen, üstü yuvarlak çubuk; göbekte Ø35 kubbeli kapak.
    g = kes(1355, 1500, -640, 560, 960, 1100)
    if g:
        sl, X, Y, Z = g
        # cam tabanının (oyuk tabanı) işaretli uzaklığı: yükseklikler bu yüzeyden ölçülür
        zem = govde_temel(X, np.abs(Y), Z) + p(0.45)

        def eksen(A, B):
            a, b = np.asarray(A, float), np.asarray(B, float)
            L = float(np.hypot(*(b - a)))
            ux, uy = (b - a) / L
            nx, ny = (uy, -ux) if uy > 0 else (-uy, ux)          # n: camda yukarı (+X yönlü)
            s_ = (X - a[0]) * ux + (Y - a[1]) * uy
            t_ = (X - a[0]) * nx + (Y - a[1]) * ny
            return s_, t_, L

        def supurge(A, B):
            s_, t_, L = eksen(A, B)
            uc = np.maximum(-s_, s_ - L)
            ucy = 0.55 + 0.45 * sstep(0.0, 22.0, np.minimum(s_, L - s_))      # uçlarda alçalan kapaklar
            lastik = np.maximum(np.maximum(np.abs(t_ + 1.0) - 3.2, uc + 4.0), zem - 6.5)
            kiris_h = (6.0 + 5.0 * np.sqrt(np.clip(1 - (t_ / 9.5) ** 2, 0, 1))
                       + 5.5 * np.clip(1 - np.abs(t_ - 1.5) / 4.0, 0, 1)) * ucy
            kiris = np.maximum(np.maximum(np.abs(t_) - 9.5, uc), np.maximum(zem - kiris_h, 5.0 - zem))
            klips = np.maximum(sd_kutu2(s_, t_, L / 2, 0.5, 21.0, 10.5, 4.0), zem - 20.5)
            return np.minimum(np.minimum(lastik, kiris), klips), L

        def kol(P, C, h0, h1):
            s_, t_, L = eksen(P, C)
            v = np.clip(s_ / L, 0, 1)
            w = 8.5 - 3.0 * v
            h = (h0 + (h1 - h0) * v) - 3.5 * np.clip(t_ / w, -1, 1) ** 2
            govde = np.maximum(np.maximum(np.abs(t_) - w, np.maximum(-s_, s_ - L - 6.0)), zem - h)
            # kolun ortası camdan kalkık (altında gölge boşluğu), göbekte ve klipste oturur
            alt = 7.0 * sstep(0.12, 0.25, v) * (1 - sstep(0.80, 0.92, v))
            return np.maximum(govde, alt - zem)

        def gobek(P):
            r = np.sqrt((X - P[0]) ** 2 + (Y - P[1]) ** 2)
            h = 13.0 + 4.0 * np.sqrt(np.clip(1 - (r / 17.5) ** 2, 0, 1)) + 3.0 * (r < 6.0)
            return np.maximum(r - 17.5, zem - h)

        SD = ((1418.0, -360.0), (1428.0, 290.0))   # sürücü süpürgesi (cam tabanında)
        SY = ((1390.0, 20.0), (1405.0, 470.0))     # yolcu süpürgesi (önde, kaput kenarının hemen arkasında)
        GD, GY = (1400.0, -470.0), (1386.0, -30.0)  # göbekler
        e = None
        for (A, B), G in ((SD, GD), (SY, GY)):
            sp, L = supurge(A, B)
            a, b = np.asarray(A), np.asarray(B)
            C = tuple(a + (b - a) * 0.5)
            k = kol(G, C, 19.0, 21.5)
            sil = np.minimum(np.minimum(sp, k), gobek(G))
            e = sil if e is None else np.minimum(e, sil)
        d[sl] = smin(d[sl], e * 0.9, p(0.03))
    # çeki kancası kapağı (ön tampon, yalnız +Y): siyah çıtanın üst kenarına binen, içe yatık paralelkenar
    # (15922355098 kalibre kamerasından yüzeye izdüşüm: üst kenar y 428..509 z 543, alt kenar 386..467 z 481);
    # alt kenarında küçük tırnak yuvası
    g = kes(-40, 260, 360, 540, 455, 570)
    if g:
        sl, X, Y, Z = g
        ck = sd_poligon(Y, Z, [(428.0, 543.0), (509.0, 544.0), (467.0, 481.0), (386.0, 481.0)])
        # kapak siyah çıtanın üst kenarının ÜSTÜNE biner (fotoğrafta çıta kenarı kapağın altında kaybolur): kapak
        # içinde çıta basamağı/oluğu gövde yüzeyine doldurulur; eskiden çıtanın kenar oluğu kapağı ortadan bölüyordu
        z0 = govde_temel(X, Y, Z)
        d[sl] = np.minimum(d[sl], np.maximum(z0, ck))
        cur = d[sl]
        cy = np.maximum(np.abs(ck) - p(0.11), -(cur + p(0.25)))
        tirnak = np.maximum(sd_kutu2(Y, Z, 432.0, 487.0, 8.0, 2.2, 2.0), -(cur + p(0.3)))
        d[sl] = np.maximum(cur, -np.minimum(cy, tirnak))

    # kapı kilidi göbeği (Ø25, ince halka oluğu + kubbe, tutamağın arka ucunda): yalnız sürücü kapısında (sol, -Y)
    g = kes(KILIT_X - 22, KILIT_X + 22, -960, -700, KOL_Z - 22, KOL_Z + 24)
    if g:
        sl, X, Y, Z = g
        Ya = -Y
        z0 = govde_temel(X, Ya, Z)
        dis = 700.0 - Ya
        k, zc, hk = kol_tutamak(X, Z)
        kl = np.sqrt((X - KILIT_X) ** 2 + (Z - zc) ** 2)
        cur = oyuk(d[sl], z0 - hk, np.maximum(np.abs(kl - 12.5) - p(0.06), dis), p(0.12))
        d[sl] = doldur(cur, z0, np.maximum(kl - 10.0, dis), p(0.5) + p(0.06) * np.clip(1 - (kl / 10.0) ** 2, 0, 1))

    if mod == "tek":
        teker_asimetrik(d, kes)
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
        d[sl] = smax(d[sl], -camurluk_kuyusu(Xs, Ys, Zs, xa), p(0.25))
        r = np.sqrt((Xs - xa) ** 2 + (Zs - AKS_Z) ** 2)
        if mod == "tek":
            stub = np.maximum(r - p(2.0), np.maximum(615.0 - Ys, Ys - 700.0))
            d[sl] = np.minimum(d[sl], stub)
            d[sl] = np.minimum(d[sl], tekerlek_sdf(Xs - xa, Zs - AKS_Z, Ys - TEKER_Y, arka_aks=(xa == AKS_ARKA)))
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


def katlanma_cevir(V, F, tol=0.01, tur=12, log=print):
    """Sadeleştirmenin bıraktığı ters dönmüş ince üçgenleri (dilimleyicide ve görüntüde çizik gibi görünür)
    uzun kenarlarını komşu yüzle çevirerek giderir. Yalnız neredeyse düzlemsel dörtgenlerde (sapma < tol mm) ve
    çevirme sonrası iki yüz birbirine <25° olduğunda uygulanır; köşeler yerinden oynamaz, topoloji manifold kalır."""
    import trimesh
    F = F.copy()
    for it in range(tur):
        t = trimesh.Trimesh(V, F, process=False)
        ang = np.degrees(t.face_adjacency_angles); adj = t.face_adjacency
        mn = np.full(len(F), 999.); np.minimum.at(mn, adj[:, 0], ang); np.minimum.at(mn, adj[:, 1], ang)
        katl = mn > 60
        sel = np.where((katl[adj[:, 0]] | katl[adj[:, 1]]) & (ang > 60))[0]
        if len(sel) == 0: break
        e = t.face_adjacency_edges[sel]; us = t.face_adjacency_unshared[sel]; f0 = adj[sel, 0]; f1 = adj[sel, 1]
        a, b, c, d = e[:, 0], e[:, 1], us[:, 0], us[:, 1]
        F0 = F[f0]; pos_a = np.argmax(F0 == a[:, None], axis=1)
        ab = F0[np.arange(len(F0)), (pos_a + 1) % 3] == b
        A = np.where(ab, a, b); B = np.where(ab, b, a)
        N1 = np.c_[A, d, c]; N2 = np.c_[d, B, c]
        def nrm(T):
            n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]); l = np.linalg.norm(n, axis=1)
            return n / (l[:, None] + 1e-15), l
        n1, l1 = nrm(N1); n2, l2 = nrm(N2)
        # referans: katlanmamış (büyük) yüzün normali
        nf0, _ = nrm(F[f0]); nf1, _ = nrm(F[f1])
        ref = np.where(katl[f0][:, None], nf1, nf0)
        ok = ((n1 * n2).sum(1) > np.cos(np.radians(25))) & ((n1 * ref).sum(1) > 0.8) & ((n2 * ref).sum(1) > 0.8)
        ok &= (l1 > 1e-9) & (l2 > 1e-9) & (c != d)
        dist = np.abs(((V[d] - V[c]) * ref).sum(1))
        ok &= dist < tol
        ek = np.sort(t.edges_unique, axis=1); key = ek[:, 0].astype(np.int64) * len(V) + ek[:, 1]
        cd = np.sort(np.c_[c, d], axis=1); kcd = cd[:, 0].astype(np.int64) * len(V) + cd[:, 1]
        ok &= ~np.isin(kcd, key)
        idx = np.where(ok)[0]
        kul = np.zeros(len(F), bool); vk = np.zeros(len(V), bool); s = []
        for i in idx:
            if kul[f0[i]] or kul[f1[i]] or vk[c[i]] or vk[d[i]]: continue
            kul[f0[i]] = kul[f1[i]] = True; vk[c[i]] = vk[d[i]] = True; s.append(i)
        s = np.array(s, int)
        if len(s) == 0: break
        F[f0[s]] = N1[s]; F[f1[s]] = N2[s]
    log(f"  katlanmış ince üçgen düzeltmesi: {int(katl.sum())} kaldı ({it + 1} tur)")
    return F


def kaydet(v, f, yol):
    import trimesh
    tm = trimesh.Trimesh(v, f, process=False)
    par = tm.split(only_watertight=False)
    if len(par) > 1:
        tm = max(par, key=lambda m: len(m.faces))
        atilan = [m for m in par if m is not tm]
        hacim = sum(abs(m.volume) for m in atilan)
        print(f"  uyarı: ana gövdeye bağlı olmayan {len(atilan)} parça atıldı "
              f"(toplam {hacim:.3f} mm3, en büyüğü {max(len(m.faces) for m in atilan)} üçgen)")
    tm = trimesh.Trimesh(tm.vertices, katlanma_cevir(tm.vertices, tm.faces), process=False)
    tm.vertices[:, 2] -= tm.vertices[:, 2].min()
    tm.export(yol)
    e = tm.extents
    print(f"  -> {yol}: {e[0]:.1f} x {e[1]:.1f} x {e[2]:.1f} mm, {len(tm.faces)} üçgen, "
          f"su geçirmez={tm.is_watertight}, hacim={tm.volume / 1000:.1f} cm3")
    return tm


def teker_mesh(h_baski, n, tolerans, yol, arka=False):
    """Kit tekerleği; arka=True -> arka aks freni (küçük disk, tek pistonlu kaliper)."""
    from skimage.measure import marching_cubes
    h = h_baski * n
    a = np.arange(-360.0, 360.0 + h, h)
    t = np.arange(-140.0, 140.0 + h, h)
    DZ, T = a[None, :, None], t[None, None, :]
    vol = np.empty((len(a), len(a), len(t)), np.float32)
    for i0 in range(0, len(a), 24):
        DX = a[i0:i0 + 24, None, None]
        # baskıda yukarıdan bakınca sağ = +dx (dx, dz, t sağ elli eksen takımı)
        v = tekerlek_sdf(DX, DZ, T, yon=1.0, arka_aks=arka)
        delik = np.maximum(np.sqrt(DX ** 2 + DZ ** 2) - p(TEKER_DELIK_R), T - 40.0)
        vol[i0:i0 + 24] = np.maximum(v, -delik)
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
    ap.add_argument("--plakasiz", action="store_true", help="arka plaka takılmaz; yuvadaki kabartma GTC yazısı görünür")
    a = ap.parse_args()
    global ARKA_PLAKA
    if a.plakasiz:
        ARKA_PLAKA = False
    n = a.olcek
    if a.parca in ("tek", "hepsi"):
        print("[tek parça]")
        v, f = govde_mesh("tek", a.h, n)
        print(f"  {len(f)} üçgen -> sadeleştiriliyor (tolerans {a.tolerans} mm)")
        vs, fs = sadelestir(v, f, a.tolerans)
        del v, f  # 24M üçgenlik ham ağı bırak; yoksa kayıtta bellek biter
        gc.collect()
        kaydet(vs, fs, f"{a.onek}_tek_parca.stl")
    if a.parca in ("kit", "hepsi"):
        print("[kit gövde]")
        v, f = govde_mesh("kit", a.h, n)
        print(f"  {len(f)} üçgen -> sadeleştiriliyor (tolerans {a.tolerans} mm)")
        vs, fs = sadelestir(v, f, a.tolerans)
        del v, f  # 24M üçgenlik ham ağı bırak; yoksa kayıtta bellek biter
        gc.collect()
        kaydet(vs, fs, f"{a.onek}_govde.stl")
        print("[teker: ön]")
        teker_mesh(min(a.h, 0.07), n, a.tolerans, f"{a.onek}_teker_on.stl")
        print("[teker: arka]")
        teker_mesh(min(a.h, 0.07), n, a.tolerans, f"{a.onek}_teker_arka.stl", arka=True)


if __name__ == "__main__":
    main()
