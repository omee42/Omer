#!/usr/bin/env python3
"""
Opel Astra J Sedan 1.6 CDTI Sport (2012–2018, 4 kapı, makyajlı ön, dizel turbo) — 3D baskı maketi üretici.

Yöntem ve araçlar: .claude/skills/arac-3d (SKILL.md). Ortak çekirdek: scripts/arac_cekirdek.py.
Tüm geometri gerçek aracın milimetre ölçüleriyle tanımlanır, baskı ölçeğine en sonda küçültülür.
    boy 4658, genişlik 1814 (aynasız), yükseklik 1500, aks mesafesi 2685, iz ön 1544 / arka 1558,
    Sport donanım: 17" on kollu (beş çift, kıvrık) jant, 225/45 R17 lastik; bagajda ASTRA ve CDTI yazıları.
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
                           sd_kutu3, sd_elipsoid, oyuk, doldur, petek, yumusat, duz_yazi_2d, govde_mesh, sadelestir,
                           kaydet, teker_mesh)

# ============================================================ ana ölçüler (gerçek mm)
L = 4658.0
GEN_YARI = 907.0          # 1814 / 2
YUKSEKLIK = 1500.0
AKS_MES = 2685.0
AKS_ON = 925.0           # yan fotoğraflardan (y6/y8 kameraları) 895-925; uç ölçüleri 925'i verir
AKS_ARKA = AKS_ON + AKS_MES
TEKER_R = 317.0           # Sport: 225/45 R17 -> çap 634
JANT_R = 216.0            # 17 inç oturma yarıçapı
JANT_DIS = 229.0          # görünen jant dudağı
TEKER_YG = 108.0          # lastik yarı genişliği (dış yanak y≈±880)
TEKER_Y_ON, TEKER_Y_ARKA = 772.0, 779.0   # iz / 2
AKS_Z = TEKER_R
CAMUR_R = 372.0          # kemer dudağı tepesi z≈690-697 (y8)
KUYU_Y = 640.0
AKS_DELIK_R = 1.05        # kit, baskı mm: 1.75 mm filament gövdede serbest döner
TEKER_DELIK_R = 0.90      # kit, baskı mm: tekerlekte sıkı geçme
KUTU = (-40.0, L + 60.0, 1000.0, -20.0, YUKSEKLIK + 40.0)   # üretim hacmi (x0, x1, y_yari, z0, z1)


# ============================================================ profiller (gerçek mm)
# Kaynak: iki tam yan fotoğraf, perspektif kamera teker merkezleri / lastik kenarları / tampon uçları / tavan ile
# oturtuldu (sedan/kam/y6.json: "Opel Astra J Stufenheck 1", y8.json: "...in profile in Aardenburg"). Noktalar
# ızgaralı kırpmalardan okundu ve yüzey düzlemine (yan yüz y≈850-900, tavan y≈-200, ön/arka yüz y≈300) geri
# izdüşüldü; iki fotoğraf 20-35 mm içinde uyuşur. Ön aks 925 (ön yüz x≈0, arka yüz 4642 @ y=300).
# yan profil, orta hat: alt gövde üst çizgisi (ön yüz -> kaput -> torpido; bagaj güvertesi -> dudak)
Z_UST = Egri([(0, 705), (20, 750), (100, 792), (255, 840), (414, 875), (625, 920), (838, 965), (1051, 1015),
              (1212, 1057), (1400, 1076), (2400, 1146), (3000, 1195), (3400, 1217), (3700, 1205), (3950, 1180), (4040, 1165), (4184, 1142),
              (4328, 1131), (4473, 1124), (4589, 1114), (4630, 1092), (4650, 1050), (4658, 1000)])
# orta hattan yana düşüş (alt gövde üst yüzü): kaput ve bagaj kemeri; ortada bel hizasına iner (yan bel BEL_Z)
KAPUT_BOMBE = Egri([(0, 60), (300, 75), (800, 85), (1212, 95), (1400, 110), (3900, 110), (4100, 80), (4500, 70), (4658, 70)])
# bel (yan cam alt çizgisi, kromun alt kenarı) yan yüzde, y≈850
BEL_Z = Egri([(1220, 962), (1430, 977), (1640, 985), (1840, 1005), (2130, 1024), (2440, 1050), (2900, 1075),
              (3095, 1081), (3246, 1087), (3398, 1102), (3524, 1129), (3649, 1189), (3755, 1275)])
Z_ALT = Egri([(0, 260), (40, 200), (105, 182), (300, 180), (700, 190), (1100, 205), (3400, 205), (3950, 200),
              (4150, 215), (4350, 238), (4500, 250), (4600, 280), (4658, 350)])
HW_PLAN = Egri([(0, 640), (80, 740), (180, 810), (300, 855), (450, 882), (650, 897), (925, 904), (1600, 906), (2400, 907), (3555, 907), (4200, 900), (4450, 885), (4560, 850), (4620, 790), (4658, 690)])
# kesit biçimi (yüksekliğe göre yarı genişlik katsayısı): eşikte içe kıvrım, omuzda içe dönüş
F_DIKEY = Egri([(170, 0.86), (210, 0.93), (250, 0.965), (300, 0.982), (380, 0.992), (500, 0.998), (620, 1.0), (760, 0.995),
                (880, 0.978), (980, 0.952), (1060, 0.925), (1140, 0.89)])
ON_PLAN = Egri([(0, 0), (200, 10), (400, 40), (550, 80), (650, 125), (740, 185), (810, 260), (860, 350), (892, 460), (906, 560)])
ON_YAN = Egri([(140, 105), (195, 40), (273, 6), (400, 0), (585, 0), (700, 6), (760, 18)])
ARKA_PLAN = Egri([(0, 0), (250, 10), (450, 32), (600, 72), (720, 132), (810, 212), (870, 312), (900, 420), (907, 520)])
ARKA_YAN = Egri([(180, 110), (250, 60), (300, 30), (390, 8), (500, 0), (600, 6), (710, 22), (850, 45), (955, 68), (1060, 80),
                 (1100, 62), (1114, 55)])
# kabin (cam evi): orta hat tavan / ön cam / arka cam çizgisi (y≈-200 siluetinden)
Z_KABIN = Egri([(1150, 1030), (1212, 1057), (1369, 1090), (1642, 1205), (1916, 1335), (2193, 1428), (2471, 1487),
                (2750, 1497), (3034, 1495), (3317, 1468), (3603, 1390), (3749, 1280), (3894, 1205), (3990, 1185),
                (4060, 1168)])
# cam evinin bel hizasında (Z_BEL) yarı genişliği ve yan eğimi: bel y≈860, tavan rayı (DLO üstü) y≈690
HW_KABIN = Egri([(1150, 630), (1250, 675), (1450, 765), (1700, 815), (2000, 832), (2600, 838), (3200, 832),
                 (3500, 812), (3750, 765), (3950, 690), (4060, 630)])
EGIM_X = Egri([(1300, 0.44), (2600, 0.50), (3400, 0.55), (3900, 0.62)])   # cam evi yan eğimi (tumblehome) dy/dz
Z_BEL = 1040.0            # bel (cam alt) hizası referansı
# cam hattı (DLO): bel BEL_Z (y≈840) ve üst kenar (tavan rayı) DLO_UST (y≈760); arkada C direğindeki üçgen camla
# sivri uçta (x≈3755, z≈1275) birleşir (y8 tam yan fotoğrafı)
DLO_UST = Egri([(1210, 1000), (1386, 1062), (1605, 1218), (1813, 1326), (1882, 1338), (2134, 1419), (2273, 1433),
                (2520, 1440), (2896, 1428), (3125, 1412), (3253, 1386), (3355, 1370), (3535, 1336), (3715, 1288),
                (3755, 1275)])
# arka kapı camı ile C direğindeki sabit üçgen cam arasındaki bölücü (y8: üstte x≈3230, belde x≈3360)
CAM_BOLUCU = ((3230.0, 1388.0), (3360.0, 1094.0))
# kapı çizgileri (yan yüz, x, z): ön kapı ön kenarı, B direği, arka kapı arka kenarı (teker kemerinin önünden döner)
KAPI_ON = [(1385, 985), (1395, 800), (1392, 600), (1380, 420), (1375, 300)]
KAPI_B = [(2430, 1050), (2425, 800), (2420, 600), (2425, 300)]
KAPI_ARKA = [(3360, 1094), (3330, 1000), (3312, 850), (3270, 680), (3215, 560), (3140, 450), (3060, 380), (2950, 330),
             (2900, 315)]
KOL_ON, KOL_ARKA = (2210, 860), (3130, 950)    # kapı kolları (merkez x, z)
SINYAL = (1290, 805)                           # ön çamurluk yan sinyali
DEPO = (3680, 985)                             # yakıt kapağı merkezi (sağ yan)


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
        hw = hw + (4.0 * kab + dudak) * sstep(150, 250, Z)
    d_yan = (np.abs(Y) - hw) / np.sqrt(1 + (dhwx * fz) ** 2 + (hwx * dfz) ** 2)
    bombe = KAPUT_BOMBE(X)
    zu = Z_UST(X) - bombe * (Y / GEN_YARI) ** 2
    # kaputta torpidodan ızgaraya yaklaşan iki yumuşak sırt (V): cowl'da ±225, önde ±125
    yr = 125.0 + 100.0 * np.clip(X / 1212.0, 0, 1)
    zu = zu + 3.0 * np.exp(-((np.abs(Y) - yr) / 90.0) ** 2) * sstep(80.0, 250.0, X) * (1 - sstep(1150.0, 1230.0, X))
    d_ust = (Z - zu) / np.sqrt(1 + Z_UST.t(X) ** 2 + (2 * bombe * Y / GEN_YARI ** 2) ** 2)
    # tamponların yan alt kenarı ortadan yüksek (önde ~240, arkada ~330); arada eşik
    za = Z_ALT(X) + sstep(450.0, 800.0, np.abs(Y)) * (55.0 * (1 - sstep(450.0, 700.0, X))
                                                    + 90.0 * sstep(4300.0, 4500.0, X))
    d_alt = (za - Z) / np.sqrt(1 + Z_ALT.t(X) ** 2)
    d_on = (ON_PLAN(np.abs(Y)) + ON_YAN(Z) - X) / np.sqrt(1 + ON_PLAN.t(np.abs(Y)) ** 2 + ON_YAN.t(Z) ** 2)
    d_arka = (X - (L - ARKA_PLAN(np.abs(Y)) - ARKA_YAN(Z))) / np.sqrt(1 + ARKA_PLAN.t(np.abs(Y)) ** 2 + ARKA_YAN.t(Z) ** 2)
    r_on = p(3.0) - p(2.0) * sstep(650.0, 720.0, Z)          # kaput ön kenarı keskin
    alt = smax(smax(d_yan, d_ust, p(3.5)), d_on, r_on)
    alt = smax(alt, d_arka, p(3.0) - p(2.3) * sstep(1040.0, 1100.0, Z))   # bagaj dudağı keskin
    alt = smax(alt, d_alt, p(1.2))
    zk = Z_KABIN(X) - 45.0 * (Y / 650.0) ** 2
    d_kust = (Z - zk) / np.sqrt(1 + Z_KABIN.t(X) ** 2 + (90.0 * Y / 650.0 ** 2) ** 2)
    eg = EGIM_X(X)
    d_kyan = (np.abs(Y) - hw_kabin(X, Z)) / np.sqrt(1 + eg ** 2 + HW_KABIN.t(X) ** 2)
    kabin = smax(d_kyan, d_kust, p(2.6))
    kabin = smax(kabin, (zu - 40.0) - Z, p(0.5))                 # cam evi tabanı alt gövdenin üst yüzünü izler
    kabin = smax(kabin, smax(1120.0 - X, X - 4060.0, 60.0), 60.0)
    return smin(alt, kabin, p(0.9))


def camurluk_kuyusu(X, Y, Z, xa):
    """Kemer açıklığı: yandan bakışta lastiği saran daire (fotoğraflarda eşiğe kadar yuvarlak iner)."""
    dx, dz = X - xa, Z - AKS_Z
    daire = np.sqrt(dx * dx + dz * dz) - CAMUR_R
    return np.maximum(daire, KUYU_Y - Y)


# ============================================================ tekerlek
# Sport: 17" on kollu jant, kollar beş çift halinde ve dudağa doğru kıvrık (siyah Cosmo sedan 15225211984 yakın plan);
# kol yüzü jant dudağından ~12 mm içeride, hafif içbükey. Göbekte Opel kapağı, beş bijon; kolların arasından disk ve
# kaliper (kolların arkasında). Lastik 225/45 R17: düz yanak, ~10 mm omuz yarıçapı.
JANT_CIFT = 5
JANT_YUZ = 74.0            # jant dudağının dış yüzü (t)
CIFT_ACI = math.radians(10.0)   # bir çiftteki iki kolun göbekteki açısal yarı aralığı


def _kol_mesafe(r, phi, yon):
    """on kıvrık kolun eksenine teğetsel uzaklık (mm); çiftler dudağa doğru hafif açılır."""
    u = np.clip((r - 55.0) / 165.0, 0.0, 1.0)
    kivrim = yon * 0.22 * u ** 1.2
    acik = CIFT_ACI * (1.0 + 0.5 * u)
    best = None
    for i in range(JANT_CIFT):
        for sgn in (-1.0, 1.0):
            a = phi - (math.pi / 2 + 2 * math.pi * i / JANT_CIFT) - kivrim - sgn * acik
            a = (a + math.pi) % (2 * math.pi) - math.pi
            dd = np.abs(a) * np.maximum(r, 1.0)
            best = dd if best is None else np.minimum(best, dd)
    return best


def tekerlek_sdf(dx, dz, t, yon=None):
    """t: dışa doğru eksen (gerçek mm), teker merkezine göre. yon: kolların kıvrılma yönü."""
    yon = 1.0 if yon is None else yon
    r = np.sqrt(dx * dx + dz * dz)
    # lastik: düz yanak, hafif şişkinlik (dudak üstünde jant koruma sırtı), ~10 mm omuz
    sis = 4.0 * np.clip(1 - ((r - 262.0) / 40.0) ** 2, 0, 1)
    lastik = smax(r - TEKER_R, np.abs(t) - (TEKER_YG - 4.0) - sis, 10.0)
    lastik = smax(lastik, JANT_R + 4.0 - r, 3.0)
    for tc in (-62.0, -21.0, 21.0, 62.0):                                  # dört boyuna sırt kanalı
        lastik = smax(lastik, -(np.maximum(np.abs(t - tc) - 4.0, TEKER_R - 7.0 - r)), 1.5)
    # jant: dudak, varil, arka sac, kollar, göbek
    dudak = smax(np.abs(r - (JANT_DIS - 8.0)) - 8.0, np.abs(t - (JANT_YUZ - 9.0)) - 10.0, 2.0)
    varil = np.maximum(np.abs(r - (JANT_R - 3.0)) - 5.0, np.abs(t + 15.0) - 75.0)
    arka = np.maximum(r - (JANT_R - 3.0), np.abs(t + 85.0) - 6.0)
    phi = np.arctan2(dz, dx)
    u = np.clip((r - 55.0) / 165.0, 0, 1)
    gen = 16.0 - 3.0 * u                                                   # kol yarı genişliği
    yuz = JANT_YUZ - 12.0 - 6.0 * np.sin(math.pi * u)                      # hafif içbükey kol yüzü
    kol = smax(_kol_mesafe(r, phi, yon) - gen, np.maximum(t - yuz, (yuz - 40.0) - t), 3.0)
    kol = np.maximum(kol, np.maximum(r - (JANT_DIS - 12.0), 45.0 - r))
    gobek = smax(r - 70.0, np.maximum(t - (JANT_YUZ - 8.0), (JANT_YUZ - 48.0) - t), 6.0)
    jant = np.minimum(np.minimum(dudak, varil), np.minimum(np.minimum(kol, gobek), arka))
    for i in range(5):
        a = 2 * math.pi * i / 5 + math.pi / 2 + math.pi / 5
        bij = np.maximum(np.sqrt((dx - 50.0 * math.cos(a)) ** 2 + (dz - 50.0 * math.sin(a)) ** 2) - 8.0,
                         np.abs(t - (JANT_YUZ - 6.0)) - 4.0)
        jant = np.minimum(jant, bij)
    kapak = smax(r - 30.0, np.abs(t - (JANT_YUZ - 4.0)) - 4.0, 2.0)
    jant = np.minimum(jant, kapak)
    jant = np.maximum(jant, -np.maximum(np.abs(r - 22.0) - 2.2, (JANT_YUZ - 1.0) - t))
    # fren: disk ve kaliper kolların arkasında
    disk = np.maximum(np.maximum(r - 150.0, 70.0 - r), np.abs(t + 30.0) - 12.0)
    disk = np.minimum(disk, np.maximum(r - 78.0, np.abs(t - 10.0) - 58.0))       # göbek: diski janta bağlar
    kal = sd_kutu3(dx, dz, t, (-130.0, 0.0, -30.0), (26.0, 72.0, 22.0), 10.0)
    jant = np.minimum(jant, np.minimum(disk, kal))
    return np.minimum(lastik, jant)


# ------------------------------------------------------------ detay ölçüleri (gerçek mm)
# Ön yüz (|y|, z) çokgenleri: Velbert ön 3/4, "Opel Astra Sedan F" ve siyah Cosmo ön 3/4 fotoğraflarından; ölçek
# aynasız genişlik 1814 ve plakadan (520 x 110). Plaka merkezi z≈510: sis farlarının (z 330-480, yan kamera) bir plaka
# boyu üstünde (Velbert fotoğrafında plaka sis yuvasının hemen üstünde, alt ızgaranın üst bandında).
UST_IZGARA = YPoligon([(-1, 750), (300, 748), (372, 738), (402, 712), (396, 690), (330, 662), (200, 640), (80, 630),
                       (-1, 628)], 6)
KROM_KANAT = [(0, 676), (150, 680), (270, 688), (350, 699), (405, 708), (440, 712)]   # krom kanat merkez çizgisi
LOGO = (0.0, 686.0, 54.0)                                                  # (y, z, halka dış yarıçapı)
ALT_IZGARA = YPoligon([(-1, 578), (330, 575), (372, 560), (392, 470), (425, 330), (420, 285), (360, 268), (-1, 266)], 24)
PLAKA_ON = (0.0, 512.0, 260.0, 55.0)                                       # (y, z, yarı genişlik, yarı yükseklik)
SIS_YUVA = YPoligon([(545, 470), (700, 492), (800, 505), (850, 470), (848, 400), (790, 330), (690, 290), (590, 290),
                     (540, 330), (535, 420)], 20)
SIS_SERIT = [(585, 455), (700, 470), (800, 478)]                           # gündüz/sinyal şeridi (yuva üstünde)
SIS_FAR = (735.0, 370.0, 42.0)                                             # yuvarlak sis farı (y, z, r), altta ve dışta
SIS_KROM = [(560, 300), (700, 296), (800, 322), (852, 400), (856, 470)]     # yuvanın alt ve dış kenarındaki krom L
# far: tüm iç bölge cam (2 mm içeride); üst kenarı kaput aralığı boyunca. Yan parça y6/y8 kameralarıyla.
FAR_ON = YPoligon([(400, 715), (470, 735), (560, 762), (650, 784), (740, 800), (820, 806), (880, 785), (902, 720),
                   (870, 676), (790, 662), (700, 660), (630, 672), (590, 684), (520, 682), (440, 700)], 8)
FAR_YAN = YPoligon([(40, 770), (130, 805), (244, 822), (380, 836), (510, 845), (480, 818), (400, 774), (340, 720),
                    (300, 685), (240, 662), (130, 660), (50, 672), (25, 715)], 8)
FAR_PROJ = [(585, 722, 40.0), (730, 742, 36.0)]                            # far içi çanaklar (y, z, r)
FAR_KIRIK = [(560, 690), (630, 676), (700, 668)]                           # alt kenardaki kırık (projektör altı)
# arka yüz (|y|, z): stop iç parça (bagaj kapağı) ve dış parça (çamurluk); stopun yan parçası (x, z)
STOP_IC = YPoligon([(305, 990), (420, 1008), (540, 1018), (642, 1024), (644, 902), (560, 918), (450, 950), (360, 975)], 8)
STOP_DIS_ARKA = YPoligon([(650, 1024), (760, 1020), (860, 1000), (905, 950), (900, 860), (860, 810), (760, 840),
                          (650, 900)], 8)
STOP_YAN = YPoligon([(3885, 945), (4100, 962), (4400, 975), (4540, 968), (4580, 880), (4560, 820), (4420, 800),
                     (4230, 845), (3990, 895)], 10)
BAGAJ_KAPAK = [(-1, 790), (300, 788), (520, 795), (600, 815), (640, 860), (644, 940), (646, 1020), (660, 1075),
               (700, 1110)]
PLAKA_ARKA = (0.0, 520.0, 262.0, 57.0)
LOGO_ARKA = (0.0, 965.0, 48.0)
KALCA_KIRIM = [(4420, 800), (4250, 772), (4100, 748), (3960, 722)]          # stop altından kemere inen kalça kırımı
# yan: alt kapı "bıçak" kırımı (üstte dışbükey, altta içbükey) (x, z)
OYUK_UST = Egri([(1410, 340), (1700, 400), (2100, 465), (2500, 525), (2900, 590), (3250, 640), (3380, 665)])
ON_DIKIS = [(560, 652), (600, 600), (640, 560), (700, 520)]                # ön tampon - çamurluk dikişi (yan)
KAPUT_YAN = [(510, 845), (700, 875), (900, 910), (1100, 945), (1215, 962)]  # kaput - çamurluk aralığı (yan)
ANTEN = (3200.0, 0.0)                                                      # çubuk anten tabanı (x, y)
CAM = p(0.25)                                                              # cam girintisi
OLUK = p(0.30)                                                             # panel aralığı derinliği
OLUK_W = p(0.16)                                                           # panel aralığı yarı genişliği


def _yan_bolge(Y, esik=700.0):
    """yana bakan yüzey için yumuşak ağırlık (|y| büyük)."""
    return sstep(esik - 60.0, esik, np.abs(Y))


def kapi_cizgileri(X, Z):
    m = sd_cizgi(X, Z, yumusat(KAPI_ON, 15.0))
    m = np.minimum(m, sd_cizgi(X, Z, yumusat(KAPI_B, 15.0)))
    m = np.minimum(m, sd_cizgi(X, Z, yumusat(KAPI_ARKA, 15.0)))
    alt = [(1375, 300), (1700, 292), (2425, 300), (2700, 305), (2900, 315)]
    m = np.minimum(m, sd_cizgi(X, Z, yumusat(alt, 20.0)))
    return m - OLUK_W


def dlo_maske(X, Z):
    """yan cam alanı (x, z): bel ile tavan rayı arası, önde A direği ucuna, arkada C direği ucuna kadar."""
    zb, zu = BEL_Z(X), DLO_UST(X)
    m = np.maximum(zb - Z, Z - zu)
    m = np.maximum(m, np.maximum(1212.0 - X, X - 3755.0))
    return m


def detaylar(d, d0, b):
    def uygula(bb, fn):
        g = b.bolge(*bb)
        if g is not None:
            sl, X, Y, Z = g
            d[sl] = fn(d[sl], d0[sl], X, Y, Z)

    # ---------------- yan cam evi: krom çerçeve, cam girintisi, siyah B direği bandı, cam bölücü
    def yan_cam(dd, z0, X, Y, Z):
        m = dlo_maske(X, Z)
        w = _yan_bolge(Y, 560.0)
        krom = np.maximum(np.abs(m + p(0.55)) - p(0.28), -w + 0.5)
        dd = doldur(dd, z0, krom, -p(0.12))
        cam = np.maximum(m + p(0.95), -w + 0.5)
        dd = oyuk(dd, z0, cam, CAM)
        bd = np.maximum(sd_kutu2(X, Z, 2420.0, 1245.0, 70.0, 220.0, 8.0), m + p(0.95))   # iki kapı çerçevesi: ~140 mm
        dd = doldur(dd, z0, np.maximum(bd, -w + 0.5), CAM - p(0.04))
        (xa, za), (xb, zb) = CAM_BOLUCU
        bol = np.maximum(sd_cizgi(X, Z, [(xa, za + 40), (xb, zb - 30)]) - 17.0, m + p(0.95))
        dd = doldur(dd, z0, np.maximum(bol, -w + 0.5), CAM - p(0.04))
        return dd
    uygula((1150, 3820, 540, 1000, 930, 1470), yan_cam)

    # ---------------- ön cam, arka cam: girinti (arka cam tavandan x≈3330'dan bagaja x≈3960'a)
    def on_arka_cam(dd, z0, X, Y, Z):
        ay = np.abs(Y)
        on = np.maximum(np.maximum(1240.0 - X, X - (1915.0 + 40.0 * (ay / 600.0) ** 2)), ay - (620.0 - 0.10 * (X - 1240.0)))
        arka = np.maximum(np.maximum(3330.0 + 90.0 * (ay / 560.0) ** 2 - X, X - (3965.0 - 30.0 * (ay / 560.0) ** 2)),
                          ay - (590.0 - 0.10 * (X - 3330.0)))
        m = np.minimum(on, arka)
        ust = sstep(1000.0, 1080.0, Z)
        return oyuk(dd, z0, np.maximum(m + p(0.6), 0.5 - ust), CAM)
    uygula((1180, 4020, 0, 760, 1000, 1520), on_arka_cam)

    # ---------------- kapı aralıkları, kollar, yan sinyal, alt kapı bıçak kırımı
    def yan_yuz(dd, z0, X, Y, Z):
        w = _yan_bolge(Y, 760.0)
        g = np.maximum(kapi_cizgileri(X, Z), -w + 0.5)
        dd = oyuk(dd, z0, g, OLUK)
        k = np.maximum(sd_cizgi(X, Z, yumusat(KAPUT_YAN, 20.0)) - OLUK_W, -sstep(560.0, 640.0, np.abs(Y)) + 0.5)
        dd = oyuk(dd, z0, k, OLUK * 0.8)
        t = np.maximum(sd_cizgi(X, Z, ON_DIKIS) - OLUK_W * 0.8, -w + 0.5)
        dd = oyuk(dd, z0, t, OLUK * 0.7)
        # bıçak kırımı: çizginin üstü +3 mm (aşağı doğru sönen dışbükey), altı -6 mm içbükey bant
        zk = OYUK_UST(X)
        uz = sstep(1380.0, 1520.0, X) * (1 - sstep(3300.0, 3420.0, X))
        ust = 3.0 * sstep(zk - 2.0, zk + 3.0, Z) * (1 - sstep(zk + 40.0, zk + 160.0, Z))
        alt = 6.0 * (1 - sstep(zk - 3.0, zk + 2.0, Z)) * sstep(zk - 200.0, zk - 50.0, Z)
        dd = dd - (ust - alt) * uz * w * (Z > 240.0)
        # kollar: cep + 180 mm çekme kolu (15 mm çıkıntılı), ön kolda arka uçta kilit göbeği
        for xk, zk2 in (KOL_ON, KOL_ARKA):
            cep = np.maximum(sd_kutu2(X, Z, xk + 10.0, zk2 - 10.0, 75.0, 28.0, 26.0), -w + 0.5)
            dd = oyuk(dd, z0, cep, p(0.40))
            tut = np.maximum(sd_kutu2(X, Z, xk, zk2, 90.0, 14.0, 13.0), -w + 0.5)
            dd = doldur(dd, z0, tut, -p(0.60))
        kil = np.maximum(np.sqrt((X - (KOL_ON[0] + 78.0)) ** 2 + (Z - KOL_ON[1]) ** 2) - 13.0, -w + 0.5)
        dd = doldur(dd, z0, kil, -p(0.70))
        sin = np.maximum(sd_kutu2(X, Z, SINYAL[0], SINYAL[1], 35.0, 10.0, 9.0), -w + 0.5)
        dd = doldur(dd, z0, sin, -p(0.08))
        return dd
    uygula((450, 3440, 640, 1000, 230, 1000), yan_yuz)

    # ---------------- far: cam yüzeyi 2 mm içeride, iki krom çanak, gündüz şeridi, alt kenar kırığı
    def far(dd, z0, X, Y, Z):
        ay = np.abs(Y)
        on = np.maximum(FAR_ON(ay, Z), X - 280.0)
        yan = np.maximum(FAR_YAN(X, Z), 600.0 - ay)
        m = np.minimum(on, yan)
        dd = oyuk(dd, z0, np.maximum(np.abs(m + 4.0) - 4.0, -m - 30.0), p(0.30))   # çevre oluğu (cam kenarı)
        dd = oyuk(dd, z0, m, p(0.08))
        for yc, zc, r in FAR_PROJ:
            kub = np.sqrt((ay - yc) ** 2 + (Z - zc) ** 2) - r
            dd = oyuk(dd, z0, np.maximum(np.abs(kub + 6.0) - 4.0, m + 6.0), p(0.25))
        drl = np.maximum(np.abs(FAR_ON(ay, Z) + 13.0) - 4.0, X - 280.0)
        dd = doldur(dd, z0, np.maximum(drl, m + 3.0), p(0.08) - p(0.10))
        kr = np.maximum(sd_cizgi(ay, Z, FAR_KIRIK) - 4.0, np.maximum(m + 3.0, X - 280.0))
        dd = oyuk(dd, z0, kr, p(0.20))
        return dd
    uygula((-40, 560, 380, 1000, 630, 880), far)

    # ---------------- ön ızgara, krom kanat, logo, alt ızgara + plaka, sis yuvaları
    def on_yuz(dd, z0, X, Y, Z):
        ay = np.abs(Y)
        on = sstep(260.0, 180.0, X)
        XU, XA = 45.0, 60.0                         # üst / alt ızgara taban düzlemleri (x)
        ui = np.maximum(UST_IZGARA(ay, Z), 0.5 - on)
        dd = np.maximum(dd, -np.maximum(ui, X - XU))
        uc = np.maximum(np.maximum(np.abs(Z - 716.0) - 4.0, UST_IZGARA(ay, Z) + 4.0), 0.5 - on)
        dd = np.minimum(dd, np.maximum(uc, (XU - 16.0) - X))
        kk = np.maximum(sd_cizgi(ay, Z, KROM_KANAT) - 13.0 + 4.0 * sstep(300.0, 440.0, ay), 0.5 - on)
        dd = doldur(dd, z0, kk, -p(0.12))
        yc, zc, R = LOGO
        rr = np.sqrt(Y ** 2 + (Z - zc) ** 2)
        dd = doldur(dd, z0, np.maximum(rr - (R + 2.0), 0.5 - on), -p(0.05))            # logo taban diski
        dd = doldur(dd, z0, np.maximum(np.abs(rr - (R - 7.0)) - 6.0, 0.5 - on), -p(0.40))
        ai = np.maximum(ALT_IZGARA(ay, Z), 0.5 - on)
        dd = np.maximum(dd, -np.maximum(ai, X - XA))
        bar = np.maximum(np.maximum(np.abs(((Z - 290.0) % 55.0) - 27.5) - 7.0, ALT_IZGARA(ay, Z) + 8.0), 0.5 - on)
        dd = np.minimum(dd, np.maximum(bar, (XA - 22.0) - X))
        y0, z1, hy, hz = PLAKA_ON
        dd = np.minimum(dd, np.maximum(sd_kutu2(ay, Z, 0.0, z1, hy, hz, 6.0), np.maximum(-2.0 - X, X - XA)))
        sy = np.maximum(SIS_YUVA(ay, Z), X - 450.0)
        dd = oyuk(dd, z0, sy, p(0.40))
        ser = np.maximum(sd_cizgi(ay, Z, SIS_SERIT) - 9.0, np.maximum(SIS_YUVA(ay, Z) + 6.0, X - 450.0))
        dd = doldur(dd, z0, ser, p(0.20))
        sf = np.maximum(np.sqrt((ay - SIS_FAR[0]) ** 2 + (Z - SIS_FAR[1]) ** 2) - SIS_FAR[2], X - 450.0)
        dd = doldur(dd, z0, sf, p(0.22))
        dd = oyuk(dd, z0, np.maximum(np.abs(sf + 5.0) - 3.0, X - 450.0), p(0.32))
        kr = np.maximum(sd_cizgi(ay, Z, SIS_KROM) - 7.0, X - 450.0)
        dd = doldur(dd, z0, kr, -p(0.12))
        dd = doldur(dd, z0, np.maximum(np.maximum(np.abs(Z - 192.0) - 12.0, ay - 520.0), 0.5 - on), -p(0.10))  # çene dudağı
        return dd
    uygula((-40, 480, 0, 880, 160, 760), on_yuz)

    # ---------------- arka: stoplar, bagaj kapağı aralığı, plaka cebi, logo, alt siyah bant
    def arka_yuz(dd, z0, X, Y, Z):
        ay = np.abs(Y)
        ark = sstep(L - 260.0, L - 160.0, X)
        st = np.minimum(np.maximum(STOP_IC(ay, Z), 0.5 - ark), np.maximum(STOP_DIS_ARKA(ay, Z), 0.5 - ark))
        st = np.minimum(st, np.maximum(STOP_YAN(X, Z), 640.0 - ay))
        dd = oyuk(dd, z0, np.maximum(np.abs(st + 4.0) - 4.0, -st - 40.0), p(0.30))     # stop kenar oluğu
        dd = oyuk(dd, z0, st, p(0.06))
        ser = np.maximum(np.abs(((Z - 965.0) % 38.0) - 19.0) - 4.0, st + 8.0)
        dd = doldur(dd, z0, ser, p(0.06) - p(0.10))
        bk = np.maximum(sd_cizgi(ay, Z, yumusat(BAGAJ_KAPAK, 15.0)) - OLUK_W, 0.5 - ark)
        dd = oyuk(dd, z0, bk, OLUK)
        y0, z1, hy, hz = PLAKA_ARKA
        XP = L - 22.0                                   # plaka cebi taban düzlemi; plaka yüzü XP + 6
        cep = np.maximum(sd_kutu2(ay, Z, 0.0, z1, hy + 10.0, hz + 10.0, 15.0), 0.5 - ark)
        dd = np.maximum(dd, -np.maximum(cep, XP - X))
        dd = np.minimum(dd, np.maximum(sd_kutu2(ay, Z, 0.0, z1, hy, hz, 5.0), np.maximum(X - (XP + 6.0), XP - 30.0 - X)))
        yc, zc, R = LOGO_ARKA
        rr = np.sqrt(Y ** 2 + (Z - zc) ** 2)
        dd = doldur(dd, z0, np.maximum(np.abs(rr - (R - 6.0)) - 5.5, 0.5 - ark), -p(0.30))
        dd = oyuk(dd, z0, np.maximum(np.maximum(Z - 330.0, 235.0 - Z), np.maximum(0.5 - ark, ay - 820.0)), p(0.12))
        return dd
    uygula((L - 300, L + 40, 0, 920, 200, 1140), arka_yuz)

    def arka_yan(dd, z0, X, Y, Z):
        w = _yan_bolge(Y, 700.0)
        st = np.maximum(STOP_YAN(X, Z), -w + 0.5)
        dd = oyuk(dd, z0, np.maximum(np.abs(st + 4.0) - 4.0, -st - 40.0), p(0.30))
        dd = oyuk(dd, z0, st, p(0.06))
        ser = np.maximum(np.abs(((Z - 870.0) % 38.0) - 19.0) - 4.0, st + 8.0)
        dd = doldur(dd, z0, ser, p(0.06) - p(0.10))
        kk = np.maximum(sd_cizgi(X, Z, KALCA_KIRIM) - 10.0, -w + 0.5)
        dd = dd - 2.5 * np.clip(-kk / 10.0, 0, 1) * (kk < 0)
        return dd
    uygula((3850, L, 640, 920, 600, 1000), arka_yan)

    # ---------------- çubuk anten tabanı
    def anten(dd, z0, X, Y, Z):
        a = sd_elipsoid(X, Y, Z, (ANTEN[0], 0.0, 1490.0), (30.0, 11.0, 26.0))
        cub = np.maximum(np.sqrt((X - ANTEN[0] - 8.0) ** 2 + Y ** 2) - 4.0, np.maximum(1495.0 - Z, Z - 1545.0))
        return np.minimum(dd, np.minimum(a, cub))
    uygula((3150, 3250, 0, 30, 1440, 1560), anten)

    # ---------------- aynalar: kapı camının ön köşesinde, bel üstünde ayakla; kabuk ~230 x 130, ~100 mm dışa taşar
    def ayna(dd, z0, X, Y, Z):
        g = sd_elipsoid(X, Y, Z, (1690.0, 925.0, 1085.0), (115.0, 85.0, 66.0))
        g = smax(g, X - 1772.0, 10.0)                                       # cam yüzü (arkaya bakan, düşey)
        ayak = sd_elipsoid(X, Y, Z, (1640.0, 850.0, 1035.0), (75.0, 55.0, 38.0))
        g = smin(g, ayak, 25.0)
        cam = np.maximum(sd_kutu2(Y, Z, 930.0, 1085.0, 70.0, 50.0, 28.0), np.maximum(1765.0 - X, X - 1790.0))
        g = np.maximum(g, -cam)
        return np.minimum(dd, g)
    uygula((1520, 1800, 760, 1020, 980, 1160), ayna)
    return d


def asimetrik(d, x, y, z, mod):
    """Ayna simetrik olmayan detaylar (tam hacim): yakıt kapağı (sağ), bagaj yazıları (ASTRA solda, CDTI sağda)."""
    def kes(x0, x1, y0, y1, z0, z1):
        ix = slice(*np.searchsorted(x, [x0, x1]))
        iy = slice(*np.searchsorted(y, [y0, y1]))
        iz = slice(*np.searchsorted(z, [z0, z1]))
        if ix.start >= ix.stop or iy.start >= iy.stop or iz.start >= iz.stop:
            return None
        return (ix, iy, iz), x[ix][:, None, None], y[iy][None, :, None], z[iz][None, None, :]
    # yakıt kapağı: sağ arka çamurlukta 165 mm çaplı dairesel aralık
    g = kes(3560, 3820, 780, 960, 840, 1100)
    if g is not None:
        sl, X, Y, Z = g
        hal = np.abs(np.sqrt((X - DEPO[0]) ** 2 + (Z - DEPO[1]) ** 2) - 82.0) - 3.5
        z0 = d[sl].copy()
        d[sl] = np.maximum(d[sl], -np.maximum(hal, -(z0 + p(0.25))))
    # bagaj kapağı yazıları (arkadan bakana göre: ASTRA solda = -y, CDTI sağda = +y), kabartma 3 mm
    for metin, yc in (("ASTRA", -420.0), ("CDTI", 420.0)):
        g = kes(L - 140, L + 10, min(yc - 120, yc + 120), max(yc - 120, yc + 120), 820, 880)
        if g is None:
            continue
        sl, X, Y, Z = g
        h = 26.0
        _, gen = duz_yazi_2d(0.0, 0.0, metin, h, sx=1.1, bosluk=6.0)
        u = Y - (yc - gen / 2.0)                     # arkadan bakınca soldan sağa +y
        dd, _ = duz_yazi_2d(u, Z - 838.0, metin, h, sx=1.1, bosluk=6.0)
        harf = dd - 3.2
        z0 = d[sl].copy()
        d[sl] = np.minimum(d[sl], np.maximum(harf, z0 - p(0.13)))
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
