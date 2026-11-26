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
                           sd_kutu3, sd_elipsoid, oyuk, doldur, petek, yumusat, duz_yazi_2d, govde_mesh, sadelestir,
                           kaydet, teker_mesh)

# ============================================================ ana ölçüler (gerçek mm)
L = 4658.0
GEN_YARI = 907.0          # 1814 / 2
YUKSEKLIK = 1500.0
AKS_MES = 2685.0
AKS_ON = 925.0           # yan fotoğraflardan (y6/y8 kameraları) 895-925; uç ölçüleri 925'i verir
AKS_ARKA = AKS_ON + AKS_MES
TEKER_R = 324.0           # 205/60 R16 -> çap 652 (yan fotoğraf kameraları R≈322 ile en iyi oturur)
JANT_R = 203.0            # 16 inç oturma yarıçapı
JANT_DIS = 214.0          # görünen jant dudağı
TEKER_YG = 103.0          # lastik yarı genişliği
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
Z_UST = Egri([(0, 700), (20, 755), (100, 800), (255, 840), (414, 875), (625, 920), (838, 965), (1051, 1015),
              (1212, 1057), (1400, 1076), (2400, 1146), (3000, 1195), (3400, 1217), (3700, 1205), (3950, 1180), (4040, 1165), (4184, 1142),
              (4328, 1131), (4473, 1124), (4589, 1114), (4630, 1092), (4650, 1050), (4658, 1000)])
# orta hattan yana düşüş (alt gövde üst yüzü): kaput ve bagaj kemeri; ortada bel hizasına iner (yan bel BEL_Z)
KAPUT_BOMBE = Egri([(0, 60), (300, 75), (800, 85), (1212, 95), (1400, 110), (3900, 110), (4100, 80), (4500, 70), (4658, 70)])
# bel (yan cam alt çizgisi, kromun alt kenarı) yan yüzde, y≈850
BEL_Z = Egri([(1220, 962), (1430, 977), (1640, 985), (1840, 1005), (2130, 1024), (2440, 1050), (2900, 1075),
              (3095, 1081), (3246, 1087), (3398, 1102), (3524, 1129), (3649, 1189), (3755, 1275)])
Z_ALT = Egri([(0, 260), (40, 190), (105, 150), (300, 150), (700, 165), (1100, 175), (3500, 175), (4200, 175),
              (4300, 185), (4450, 215), (4540, 250), (4600, 300), (4658, 380)])
HW_PLAN = Egri([(0, 860), (300, 885), (925, 904), (1600, 906), (2400, 907), (3555, 907), (4200, 900), (4658, 880)])
# kesit biçimi (yüksekliğe göre yarı genişlik katsayısı): eşikte içe kıvrım, omuzda içe dönüş
F_DIKEY = Egri([(150, 0.80), (200, 0.86), (250, 0.925), (300, 0.965), (380, 0.988), (500, 0.998), (620, 1.0), (760, 0.995),
                (880, 0.978), (980, 0.952), (1060, 0.925), (1140, 0.89)])
ON_PLAN = Egri([(0, 0), (200, 12), (400, 50), (550, 100), (680, 168), (780, 250), (850, 345), (890, 445), (906, 540)])
ON_YAN = Egri([(140, 105), (195, 40), (273, 6), (400, 0), (585, 0), (700, 6), (760, 18)])
ARKA_PLAN = Egri([(0, 0), (250, 10), (450, 32), (600, 72), (720, 132), (810, 212), (870, 312), (900, 420), (907, 520)])
ARKA_YAN = Egri([(180, 110), (250, 60), (300, 30), (390, 8), (500, 0), (600, 6), (710, 22), (850, 45), (955, 62), (1060, 70),
                 (1114, 70)])
# kabin (cam evi): orta hat tavan / ön cam / arka cam çizgisi (y≈-200 siluetinden)
Z_KABIN = Egri([(1150, 1030), (1212, 1057), (1369, 1090), (1642, 1205), (1916, 1335), (2193, 1428), (2471, 1487),
                (2750, 1497), (3034, 1495), (3317, 1468), (3603, 1390), (3749, 1280), (3894, 1205), (3990, 1185),
                (4060, 1168)])
# cam evinin bel hizasında (Z_BEL) yarı genişliği ve yan eğimi: bel y≈860, tavan rayı (DLO üstü) y≈690
HW_KABIN = Egri([(1150, 640), (1250, 690), (1450, 785), (1700, 838), (2000, 856), (2600, 862), (3200, 858),
                 (3500, 838), (3750, 785), (3950, 700), (4060, 640)])
EGIM_X = Egri([(1300, 0.40), (2600, 0.44), (3400, 0.48), (3900, 0.55)])   # cam evi yan eğimi (tumblehome) dy/dz
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
KOL_ON, KOL_ARKA = (2210, 860), (3110, 950)    # kapı kolları (merkez x, z)
SINYAL = (1300, 792)                           # ön çamurluk yan sinyali
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
    kabin = smax(kabin, smax(1120.0 - X, X - 4060.0, 60.0), 60.0)
    return smin(alt, kabin, p(0.9))


def camurluk_kuyusu(X, Y, Z, xa):
    """Kemer açıklığı: yandan bakışta lastiği saran daire (fotoğraflarda eşiğe kadar yuvarlak iner)."""
    dx, dz = X - xa, Z - AKS_Z
    daire = np.sqrt(dx * dx + dz * dz) - CAMUR_R
    return np.maximum(daire, KUYU_Y - Y)


# ============================================================ tekerlek
# 16" beş kollu alaşım jant (Edition donanımı; Velbert ön/arka 3/4 fotoğrafları): her kol jant dudağına doğru
# saat yönünde kıvrılır ve ortasından ince bir oluk geçer; kollar dudakta ~22°, göbekte ~34° genişliktedir.
# Göbekte Opel kapağı, beş bijon. Arkada disk fren ve kaliper kolların arasından görünür. Lastik 205/60 R16.
JANT_KOL = 5
JANT_YUZ = 68.0            # jant dış yüzü (t), lastik yanağından ~35 mm içeride
LASTIK_YANAK_SIS = 9.0     # yanak şişkinliği (mm)


def _kol_mesafe(r, phi, yon):
    """beş kıvrık kolun eksenine teğetsel uzaklık (mm)."""
    kivrim = yon * 0.42 * np.clip((r - 60.0) / 150.0, 0.0, 1.0) ** 1.3
    best = None
    for i in range(JANT_KOL):
        a = phi - (math.pi / 2 + 2 * math.pi * i / JANT_KOL) - kivrim
        a = (a + math.pi) % (2 * math.pi) - math.pi
        dd = np.abs(a) * np.maximum(r, 1.0)
        best = dd if best is None else np.minimum(best, dd)
    return best


def tekerlek_sdf(dx, dz, t, yon=None):
    """t: dışa doğru eksen (gerçek mm), teker merkezine göre. yon: kolların kıvrılma yönü (görünüşte saat yönü)."""
    yon = 1.0 if yon is None else yon
    r = np.sqrt(dx * dx + dz * dz)
    # lastik: yuvarlak omuzlu, yanakları şişkin halka
    lastik = smax(r - TEKER_R, np.abs(t) - (TEKER_YG - LASTIK_YANAK_SIS) - LASTIK_YANAK_SIS
                  * np.clip(1 - ((r - 275.0) / 55.0) ** 2, 0, 1), 28.0)
    lastik = smax(lastik, JANT_R + 4.0 - r, 3.0)
    # sırt dişleri: dört boyuna kanal
    for tc in (-55.0, -18.0, 18.0, 55.0):
        lastik = smax(lastik, -(np.maximum(np.abs(t - tc) - 4.5, TEKER_R - 7.0 - r)), 1.5)
    # jant: dudak halkası, içte çanak, kollar, göbek
    dudak = smax(np.abs(r - (JANT_DIS - 9.0)) - 9.0, np.abs(t - (JANT_YUZ - 10.0)) - 12.0, 2.0)
    canak = np.maximum(np.abs(r - (JANT_R - 3.0)) - 5.0, np.abs(t + 20.0) - 70.0)       # jant varili (halka)
    arka = np.maximum(r - (JANT_R - 3.0), np.abs(t + 82.0) - 6.0)                          # arka koruma sacı (koyu zemin)
    canak = np.minimum(canak, arka)
    phi = np.arctan2(dz, dx)
    gen = 34.0 - 12.0 * np.clip((r - 60.0) / 150.0, 0, 1)                 # kol yarı genişliği (mm)
    km = _kol_mesafe(r, phi, yon) - gen
    yuz = JANT_YUZ - 6.0 - 16.0 * np.clip((r - 60.0) / 150.0, 0, 1) ** 2  # kol yüzü dudağa doğru hafif içe çöker
    kol = smax(km, np.maximum(t - yuz, (yuz - 45.0) - t), 4.0)
    kol = np.maximum(kol, np.maximum(r - (JANT_DIS - 10.0), 40.0 - r))
    oluk = np.maximum(_kol_mesafe(r, phi, yon) - 4.0, np.maximum(yuz - 3.0 - t, np.maximum(80.0 - r, r - 190.0)))
    kol = np.maximum(kol, -oluk)
    gobek = smax(r - 72.0, np.maximum(t - (JANT_YUZ - 4.0), (JANT_YUZ - 50.0) - t), 6.0)
    jant = np.minimum(np.minimum(dudak, canak), np.minimum(kol, gobek))
    # bijonlar ve kapak
    for i in range(5):
        a = 2 * math.pi * i / 5 + math.pi / 2
        bij = np.maximum(np.sqrt((dx - 50.0 * math.cos(a)) ** 2 + (dz - 50.0 * math.sin(a)) ** 2) - 8.5,
                         np.abs(t - (JANT_YUZ - 2.0)) - 4.0)
        jant = np.minimum(jant, bij)
    kapak = smax(r - 30.0, np.abs(t - (JANT_YUZ + 0.5)) - 3.5, 2.0)
    jant = np.minimum(jant, kapak)
    jant = np.maximum(jant, -np.maximum(np.abs(r - 22.0) - 2.5, (JANT_YUZ + 3.0) - t))     # kapakta halka çizgisi
    # fren diski ve kaliper (kolların arasından görünür)
    disk = np.maximum(np.maximum(r - 140.0, 75.0 - r), np.abs(t + 25.0) - 12.0)
    kal = sd_kutu3(dx, dz, t, (-120.0, 0.0, -15.0), (28.0, 70.0, 28.0), 10.0)
    jant = np.minimum(jant, np.minimum(disk, kal))
    return np.minimum(lastik, jant)


# ------------------------------------------------------------ detay ölçüleri (gerçek mm)
# Ön yüz (y, z) yarım genişlik çokgenleri: "Opel Astra Sedan F", Velbert ön 3/4 ve 5 kapı makyajlı ön fotoğraflarına
# göre (ölçek: aynasız genişlik 1814, logo z≈700 ve plaka merkezi z≈535 Velbert kamerasından).
UST_IZGARA = YPoligon([(-1, 738), (200, 735), (330, 726), (395, 708), (402, 693), (360, 668), (250, 645), (120, 632),
                       (-1, 630)], 6)
KROM_KANAT = [(0, 676), (150, 679), (270, 686), (350, 697), (398, 703)]     # krom kanat çıtası merkez çizgisi
LOGO = (0.0, 684.0, 54.0)                                                  # (y, z, halka dış yarıçapı)
ALT_IZGARA = YPoligon([(-1, 598), (300, 596), (430, 585), (468, 560), (455, 470), (420, 340), (380, 288), (300, 272),
                       (-1, 270)], 28)
PLAKA_ON = (0.0, 525.0, 260.0, 56.0)                                       # (y, z, yarı genişlik, yarı yükseklik)
SIS_YUVA = YPoligon([(560, 470), (690, 498), (780, 520), (842, 500), (850, 430), (800, 355), (700, 318), (585, 315),
                     (540, 352), (538, 430)], 22)
SIS_FAR = (690.0, 405.0, 52.0, 34.0)                                       # sis farı merkezi (y, z) ve yarı eksenler
# far: ön yüz parçası (y, z) ve yan parçası (x, z). Yan parça y6/y8 kameralarıyla geri izdüşüldü.
FAR_ON = YPoligon([(398, 700), (470, 725), (560, 752), (650, 775), (740, 790), (820, 790), (880, 760), (900, 700),
                   (860, 665), (760, 650), (640, 648), (520, 658), (430, 676)], 10)
FAR_YAN = YPoligon([(40, 760), (120, 790), (244, 805), (380, 818), (491, 826), (470, 800), (394, 760), (330, 700),
                    (272, 652), (160, 648), (60, 660), (30, 700)], 10)
FAR_PROJ = [(560, 712, 42.0), (705, 735, 38.0)]                            # far içi çanaklar (y, z, r)
# arka yüz (y, z): stop iç (bagaj kapağı) ve dış (çamurluk) parçaları, bagaj kapağı aralığı, plaka yuvası
STOP_IC = YPoligon([(420, 1000), (520, 1012), (640, 1018), (642, 945), (560, 940), (470, 958), (420, 975)], 8)
STOP_DIS_ARKA = YPoligon([(648, 1018), (760, 1012), (860, 985), (900, 940), (880, 900), (800, 912), (700, 932),
                          (648, 945)], 8)
STOP_YAN = YPoligon([(3880, 941), (4100, 950), (4396, 958), (4500, 950), (4560, 905), (4414, 795), (4227, 844),
                     (3989, 891)], 10)
BAGAJ_KAPAK = [(-1, 790), (300, 788), (520, 795), (600, 815), (640, 860), (644, 940), (646, 1020), (660, 1075),
               (700, 1110)]
PLAKA_ARKA = (0.0, 520.0, 272.0, 66.0)
LOGO_ARKA = (0.0, 975.0, 50.0)
TAMPON_ARKA_DIKIS = [(-1, 770), (400, 768), (640, 770), (820, 760), (895, 720)]
TAMPON_YAN_DIKIS = [(4560, 705), (4380, 690), (4150, 640), (4000, 560), (3960, 470)]
# yan: alt kapı oyuğu (Astra J'nin ön kapı altından arka kemere yükselen içbükey bandı) - üst kırım çizgisi (x, z)
OYUK_UST = Egri([(1450, 455), (1900, 490), (2400, 540), (2900, 600), (3200, 650), (3330, 680)])
ON_DIKIS = [(560, 652), (600, 600), (640, 560), (700, 520)]                # ön tampon - çamurluk dikişi (yan)
KAPUT_YAN = [(491, 826), (700, 865), (900, 905), (1100, 945), (1215, 962)]  # kaput - çamurluk aralığı (yan)
ANTEN = (3380.0, 0.0)                                                      # köpekbalığı anten tabanı (x, y)
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

    # ---------------- yan cam evi: krom çerçeve (çıkıntı), cam girintisi, direkler
    def yan_cam(dd, z0, X, Y, Z):
        m = dlo_maske(X, Z)
        w = _yan_bolge(Y, 600.0)
        krom = np.maximum(np.abs(m + p(0.55)) - p(0.28), -w + 0.5)
        dd = doldur(dd, z0, krom, -p(0.12))
        cam = np.maximum(m + p(0.95), -w + 0.5)
        dd = oyuk(dd, z0, cam, CAM)
        # B direği (siyah kaplama, camdan hafif yüksek) ve arka kapı camı / üçgen cam bölücüsü
        bd = np.maximum(sd_kutu2(X, Z, 2420.0, 1245.0, 28.0, 220.0, 6.0), m + p(0.95))
        dd = doldur(dd, z0, np.maximum(bd, -w + 0.5), CAM - p(0.08))
        (xa, za), (xb, zb) = CAM_BOLUCU
        bol = np.maximum(sd_cizgi(X, Z, [(xa, za + 40), (xb, zb - 30)]) - 11.0, m + p(0.95))
        dd = doldur(dd, z0, np.maximum(bol, -w + 0.5), CAM - p(0.08))
        return dd
    uygula((1150, 3820, 560, 1000, 930, 1470), yan_cam)

    # ---------------- ön cam, arka cam (üstten bakan yüzey: girinti + çerçeve)
    def on_arka_cam(dd, z0, X, Y, Z):
        ay = np.abs(Y)
        # ön cam: x 1215-1960, kenarlarda A direği (|y| ~ 640-700)
        on = np.maximum(np.maximum(1240.0 - X, X - (1915.0 + 40.0 * (ay / 600.0) ** 2)), ay - (640.0 - 0.10 * (X - 1240.0)))
        arka = np.maximum(np.maximum(3640.0 + 60.0 * (ay / 600.0) ** 2 - X, X - 3975.0), ay - (585.0 - 0.18 * (X - 3640.0)))
        m = np.minimum(on, arka)
        ust = sstep(1000.0, 1080.0, Z)
        return oyuk(dd, z0, np.maximum(m + p(0.6), 0.5 - ust), CAM)
    uygula((1180, 4020, 0, 760, 1000, 1520), on_arka_cam)

    # ---------------- kapı aralıkları, kollar, yan sinyal, alt kapı oyuğu
    def yan_yuz(dd, z0, X, Y, Z):
        w = _yan_bolge(Y, 760.0)
        g = np.maximum(kapi_cizgileri(X, Z), -w + 0.5)
        dd = oyuk(dd, z0, g, OLUK)
        # kaput aralığı (çamurluk üstü) ve ön tampon dikişi
        k = np.maximum(sd_cizgi(X, Z, yumusat(KAPUT_YAN, 20.0)) - OLUK_W, -sstep(560.0, 640.0, np.abs(Y)) + 0.5)
        dd = oyuk(dd, z0, k, OLUK * 0.8)
        t = np.maximum(sd_cizgi(X, Z, ON_DIKIS) - OLUK_W * 0.8, -w + 0.5)
        dd = oyuk(dd, z0, t, OLUK * 0.7)
        # alt kapı oyuğu: kırım çizgisinin altında sığ içbükey bant (yüzeyden 5 mm içeri, aşağı doğru söner)
        zk = OYUK_UST(X)
        der = 5.0 * sstep(1400.0, 1550.0, X) * (1 - sstep(3250.0, 3360.0, X)) * (1 - sstep(zk - 2.0, zk + 4.0, Z)) \
            * sstep(zk - 190.0, zk - 60.0, Z)
        dd = np.maximum(dd, z0 + der * w)
        # kollar: cep (oyuk) + tutamak (çıkıntı)
        for xk, zk2 in (KOL_ON, KOL_ARKA):
            cep = np.maximum(sd_kutu2(X, Z, xk + 15.0, zk2 - 8.0, 80.0, 30.0, 28.0), -w + 0.5)
            dd = oyuk(dd, z0, cep, p(0.35))
            tut = np.maximum(sd_kutu2(X, Z, xk, zk2, 95.0, 15.0, 14.0), -w + 0.5)
            dd = doldur(dd, z0, tut, -p(0.30))
        sin = np.maximum(sd_kutu2(X, Z, SINYAL[0], SINYAL[1], 38.0, 12.0, 10.0), -w + 0.5)
        dd = doldur(dd, z0, sin, -p(0.12))
        return dd
    uygula((450, 3420, 640, 1000, 230, 980), yan_yuz)

    # ---------------- far (ön yüz + yan parça): 8 mm girinti, içinde iki çanak ve gündüz farı çizgisi
    def far(dd, z0, X, Y, Z):
        ay = np.abs(Y)
        on = np.maximum(FAR_ON(ay, Z), X - 260.0)
        yan = np.maximum(FAR_YAN(X, Z), 600.0 - ay)
        m = np.minimum(on, yan)
        dd = oyuk(dd, z0, m, p(0.32))
        for yc, zc, r in FAR_PROJ:
            kub = np.sqrt((ay - yc) ** 2 + (Z - zc) ** 2) - r
            dd = doldur(dd, z0, np.maximum(kub, m + 6.0), p(0.32) - p(0.22) * np.sqrt(np.clip(1 - ((ay - yc) ** 2 + (Z - zc) ** 2) / r ** 2, 0, 1)))
        drl = np.maximum(np.abs(FAR_ON(ay, Z) + 14.0) - 5.0, X - 260.0)
        dd = doldur(dd, z0, np.maximum(drl, m + 4.0), p(0.32) - p(0.18))
        return dd
    uygula((-40, 520, 380, 1000, 620, 860), far)

    # ---------------- ön ızgara, logo, alt ızgara + plaka, sis yuvaları
    def on_yuz(dd, z0, X, Y, Z):
        ay = np.abs(Y)
        on = sstep(260.0, 180.0, X)            # ön yüz ağırlığı
        ui = np.maximum(UST_IZGARA(ay, Z), 0.5 - on)
        dd = oyuk(dd, z0, ui, p(0.9))
        kk = np.maximum(sd_cizgi(ay, Z, KROM_KANAT) - 11.0, 0.5 - on)
        dd = doldur(dd, z0, kk, -p(0.05))
        yc, zc, R = LOGO
        rr = np.sqrt(Y ** 2 + (Z - zc) ** 2)
        halka = np.maximum(np.abs(rr - (R - 7.0)) - 6.0, 0.5 - on)
        dd = doldur(dd, z0, halka, -p(0.20))
        ai = np.maximum(ALT_IZGARA(ay, Z), 0.5 - on)
        dd = oyuk(dd, z0, ai, p(1.0))
        # alt ızgara içi: petek ağ (ince çubuklar, ağ derinliği yüzeyden p(0.5) içeride)
        ag = np.maximum(np.maximum(-petek(Y, Z, 26.0) - 3.5, ALT_IZGARA(ay, Z) + 10.0), 0.5 - on)
        dd = doldur(dd, z0, ag, p(0.5))
        # üst ızgara içi: iki ince yatay çıta
        uc = np.maximum(np.maximum(np.minimum(np.abs(Z - 650.0), np.abs(Z - 718.0)) - 5.0, UST_IZGARA(ay, Z) + 4.0), 0.5 - on)
        dd = doldur(dd, z0, uc, p(0.45))
        y0, z1, hy, hz = PLAKA_ON
        pl = np.maximum(sd_kutu2(ay, Z, 0.0, z1, hy, hz, 8.0), 0.5 - on)
        dd = doldur(dd, z0, pl, p(0.15))
        sy = np.maximum(SIS_YUVA(ay, Z), X - 420.0)
        dd = oyuk(dd, z0, sy, p(0.7))
        sf = np.maximum(((ay - SIS_FAR[0]) / SIS_FAR[2]) ** 2 + ((Z - SIS_FAR[1]) / SIS_FAR[3]) ** 2 - 1.0, X - 420.0) * 30.0
        dd = doldur(dd, z0, sf, p(0.35))
        return dd
    uygula((-40, 460, 0, 880, 240, 760), on_yuz)

    # ---------------- arka: stoplar, bagaj kapağı aralığı, plaka yuvası, tampon dikişleri, logo
    def arka_yuz(dd, z0, X, Y, Z):
        ay = np.abs(Y)
        ark = sstep(L - 260.0, L - 160.0, X)
        st = np.minimum(np.maximum(STOP_IC(ay, Z), 0.5 - ark), np.maximum(STOP_DIS_ARKA(ay, Z), 0.5 - ark))
        st = np.minimum(st, np.maximum(STOP_YAN(X, Z), 640.0 - ay))
        dd = oyuk(dd, z0, st, p(0.30))
        # stop içi: iki yatay şerit (LED çizgileri)
        ser = np.maximum(np.abs(((Z - 960.0) % 34.0) - 17.0) - 5.0, st + 6.0)
        dd = doldur(dd, z0, ser, p(0.18))
        bk = np.maximum(sd_cizgi(ay, Z, yumusat(BAGAJ_KAPAK, 15.0)) - OLUK_W, 0.5 - ark)
        dd = oyuk(dd, z0, bk, OLUK)
        td = np.maximum(sd_cizgi(ay, Z, TAMPON_ARKA_DIKIS) - OLUK_W, 0.5 - ark)
        dd = oyuk(dd, z0, td, OLUK * 0.8)
        y0, z1, hy, hz = PLAKA_ARKA
        dd = oyuk(dd, z0, np.maximum(sd_kutu2(ay, Z, 0.0, z1, hy + 25.0, hz + 20.0, 25.0), 0.5 - ark), p(0.35))
        dd = doldur(dd, z0, np.maximum(sd_kutu2(ay, Z, 0.0, z1, hy, hz, 6.0), 0.5 - ark), p(0.20))
        yc, zc, R = LOGO_ARKA
        rr = np.sqrt(Y ** 2 + (Z - zc) ** 2)
        dd = doldur(dd, z0, np.maximum(np.abs(rr - (R - 6.0)) - 5.0, 0.5 - ark), -p(0.20))
        # alt siyah bant (difüzör) - hafif girinti
        dd = oyuk(dd, z0, np.maximum(np.maximum(Z - 300.0, 180.0 - Z), np.maximum(0.5 - ark, ay - 820.0)), p(0.15))
        return dd
    uygula((L - 300, L + 40, 0, 920, 150, 1140), arka_yuz)

    def arka_yan(dd, z0, X, Y, Z):
        w = _yan_bolge(Y, 700.0)
        st = np.maximum(STOP_YAN(X, Z), -w + 0.5)
        dd = oyuk(dd, z0, st, p(0.30))
        ser = np.maximum(np.abs(((Z - 870.0) % 34.0) - 17.0) - 5.0, st + 6.0)
        dd = doldur(dd, z0, ser, p(0.18))
        td = np.maximum(sd_cizgi(X, Z, TAMPON_YAN_DIKIS) - OLUK_W, -w + 0.5)
        return oyuk(dd, z0, td, OLUK * 0.8)
    uygula((3850, L, 640, 920, 430, 990), arka_yan)

    # ---------------- köpekbalığı anten
    def anten(dd, z0, X, Y, Z):
        u = (X - ANTEN[0]) / 150.0
        prof = 1490.0 + 62.0 * np.clip(1 - u, 0, 1) ** 0.6 * np.clip(u + 0.05, 0, 1) ** 0.4
        a = np.maximum(np.maximum(np.abs(Y) - 22.0 * np.clip(1 - u * 0.5, 0.3, 1), Z - prof),
                       np.maximum(ANTEN[0] - 5.0 - X, X - ANTEN[0] - 155.0))
        return np.minimum(dd, a)
    uygula((3360, 3560, 0, 40, 1400, 1580), anten)

    # ---------------- aynalar (A direği dibinde, kapı üstünde)
    def ayna(dd, z0, X, Y, Z):
        # gövde (kabuk): yandan damla biçimi, dışa doğru incelir; kol kapı üst köşesine oturur
        g = sd_elipsoid(X, Y, Z, (1330.0, 950.0, 1050.0), (90.0, 115.0, 58.0))
        g = smax(g, 1262.0 - X, 12.0)
        kol = sd_elipsoid(X, Y, Z, (1300.0, 845.0, 1015.0), (70.0, 70.0, 32.0))
        g = smin(g, kol, 25.0)
        return np.minimum(dd, g)
    uygula((1200, 1460, 740, 1100, 960, 1130), ayna)
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
