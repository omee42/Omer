#!/usr/bin/env python3
"""
3D yazıcı için Opel tarzı (Astra benzeri oranlarda) hatchback maket üretici.

Çıktı: tek parça, su geçirmez (manifold) bir STL dosyası.
Varsayılan boy ~98 mm (yaklaşık 1:43 ölçek). --olcek ile büyütülüp küçültülebilir.

Kullanım:
    pip install manifold3d trimesh numpy
    python3 araba_olustur.py                 # opel_tarzi_hatchback.stl üretir
    python3 araba_olustur.py --olcek 1.5     # %150 boyut (~147 mm)
    python3 araba_olustur.py --cikti x.stl

Koordinatlar (mm): X = boy (ön x=0), Y = genişlik (merkez y=0), Z = yükseklik (zemin z=0).
Not: Marka logosu bilerek eklenmemiştir; ön tampon üzerinde boş, yuvarlak bir rozet yeri var.
"""
import argparse
import math

import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold, JoinType, OpType, set_circular_segments

set_circular_segments(96)

# ---------------------------------------------------------------- ölçüler
BOY = 98.0
YARI_GEN = 21.0          # gövde yarı genişliği (toplam 42 mm)
ON_AKS, ARKA_AKS = 20.0, 78.0
TEKER_R = 9.5            # tekerlek yarıçapı -> aks yüksekliği
TEKER_GEN = 8.0
CAMUR_R = 11.0           # çamurluk boşluğu yarıçapı
CAMUR_DERIN = 9.5        # çamurluk boşluğunun yandan içeri derinliği
CAM_DERIN = 1.0          # cam girintisi derinliği


def poligon(noktalar) -> CrossSection:
    """Nokta sırası saat yönünde olsa bile geçerli (pozitif alanlı) kesit üretir."""
    p = np.asarray(noktalar, dtype=float)
    alan = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    if alan < 0:
        p = p[::-1]
    return CrossSection([p.tolist()])


def yumusat(cs: CrossSection, r: float) -> CrossSection:
    """Dış köşeleri r yarıçapıyla yuvarlar."""
    return cs.offset(-r, JoinType.Round).offset(r, JoinType.Round)


def yan_profil(noktalar, genislik):
    """(x, z) yan profilini Y ekseni boyunca, merkezde simetrik ekstrüde eder."""
    cs = CrossSection([noktalar])
    return yan_profil_cs(cs, genislik)


def yan_profil_cs(cs, genislik):
    # ekstrüzyon (a, b, c) -> (x=a, y=c-gen/2, z=b)
    T = np.array([[1, 0, 0, 0], [0, 0, 1, -genislik / 2], [0, 1, 0, 0]], dtype=float)
    return cs.extrude(genislik).transform(T)


def on_profil_cs(cs, boy, x0=-5.0):
    """(y, z) ön kesitini X ekseni boyunca ekstrüde eder."""
    # ekstrüzyon (a, b, c) -> (x=c+x0, y=a, z=b)
    T = np.array([[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    return cs.extrude(boy).transform(T)


def ust_gorunum_cs(cs, z0, yukseklik):
    return cs.extrude(yukseklik).translate([0, 0, z0])


def kutu(x0, x1, y0, y1, z0, z1):
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def y_silindir(r, y0, y1, x, z):
    """Y ekseni yönünde silindir."""
    c = Manifold.cylinder(y1 - y0, r, r)      # Z yönünde
    c = c.rotate([-90, 0, 0])                 # Z -> +Y
    return c.translate([x, y0, z])


def her_iki_yan(m: Manifold) -> Manifold:
    return m + m.mirror([0, 1, 0])


# ---------------------------------------------------------------- gövde
def alt_govde():
    prof = poligon([
        (0.0, 5.0), (0.0, 14.5), (1.0, 18.0), (4.0, 20.5),
        (27.0, 23.5), (93.0, 24.5), (96.5, 22.5), (98.0, 18.0), (98.0, 5.0),
    ])
    prof = yumusat(prof, 1.6)
    yan = yan_profil_cs(prof, 2 * YARI_GEN)

    on = yumusat(CrossSection.square([2 * YARI_GEN, 30]).translate([-YARI_GEN, 2]), 3.0)
    ust = yumusat(CrossSection.square([BOY, 2 * YARI_GEN]).translate([0, -YARI_GEN]), 6.0)

    g = yan ^ on_profil_cs(on, BOY + 10) ^ ust_gorunum_cs(ust, 0, 40)
    return g


def kabin_profili():
    return yumusat(poligon([
        (26.0, 21.5), (45.5, 35.8), (50.0, 37.5), (76.0, 37.5),
        (86.0, 35.2), (94.5, 25.0), (95.5, 21.5),
    ]), 1.2)


def kabin_kesiti():
    # Yandan bakınca içe doğru eğimli camlar (tumblehome)
    return yumusat(poligon([
        (-19.0, 18.0), (19.0, 18.0), (15.0, 40.0), (-15.0, 40.0),
    ]), 2.5)


def kabin():
    prof = kabin_profili()
    k = yan_profil_cs(prof, 2 * YARI_GEN) ^ on_profil_cs(kabin_kesiti(), BOY + 10)
    ust = yumusat(CrossSection.square([BOY, 2 * YARI_GEN]).translate([0, -YARI_GEN]), 9.0)
    return k ^ ust_gorunum_cs(ust, 0, 45)


def camlar():
    """Yan, ön ve arka camlar için sığ girintiler (çıkarılacak hacim)."""
    kesit = kabin_kesiti()
    kabuk_yan = on_profil_cs(kesit, BOY + 10) - on_profil_cs(kesit.offset(-CAM_DERIN, JoinType.Round), BOY + 10)

    prof = kabin_profili()
    ic = prof.offset(-1.6, JoinType.Round)          # direkler için pay
    # yan camlar: A, B ve C direkleri bırakılarak
    on_yan = ic ^ CrossSection.square([27.0, 12.0]).translate([31.0, 25.0])   # x 31..58
    arka_yan = ic ^ CrossSection.square([27.0, 12.0]).translate([60.5, 25.0])  # x 60.5..87.5
    yan_cam = yan_profil_cs(on_yan + arka_yan, 2 * YARI_GEN + 4) ^ kabuk_yan

    # ön cam ve arka cam: profilin eğimli yüzeyinde ince kabuk
    kabuk_prof = prof - prof.offset(-CAM_DERIN, JoinType.Round)
    ic_kesit = on_profil_cs(kesit.offset(-2.2, JoinType.Round), BOY + 10)
    on_cam = yan_profil_cs(kabuk_prof ^ CrossSection.square([18.5, 10.0]).translate([28.0, 24.8]),
                           2 * YARI_GEN + 4) ^ ic_kesit
    arka_cam = yan_profil_cs(kabuk_prof ^ CrossSection.square([8.5, 9.0]).translate([86.5, 26.0]),
                             2 * YARI_GEN + 4) ^ ic_kesit
    return yan_cam + on_cam + arka_cam


def detaylar_cikar():
    """Far, ızgara, stop lambası, kapı çizgileri gibi girintiler."""
    d = Manifold()
    # farlar (Opel tarzı ince, köşeli ve keskin bakışlı)
    far = poligon([(10.0, 15.0), (19.0, 15.6), (19.5, 18.6), (11.5, 18.2)])
    d += her_iki_yan(on_profil_cs(far, 5.5, x0=-1.0))
    # ızgara: yatay, ince ve ortada kesintisiz bir çubuk (rozet için)
    izgara = CrossSection.square([18.0, 3.6]).translate([-9.0, 10.4])
    d += on_profil_cs(izgara, 2.0, x0=-1.0)
    # alt hava girişi
    alt = yumusat(CrossSection.square([26.0, 2.6]).translate([-13.0, 6.6]), 0.8)
    d += on_profil_cs(alt, 2.0, x0=-1.0)
    # stop lambaları
    stop = poligon([(12.0, 18.0), (19.0, 18.4), (19.0, 22.0), (13.0, 21.8)])
    d += her_iki_yan(on_profil_cs(stop, 3.0, x0=96.6))
    # arka plaka yeri
    d += kutu(97.3, 100.0, -7.5, 7.5, 9.0, 13.5)
    # kapı çizgileri (yanlarda ince oluk)
    for x in (31.0, 58.8, 86.0):
        d += her_iki_yan(kutu(x, x + 0.6, YARI_GEN - 0.5, YARI_GEN + 2, 9.0, 23.6))
    # kapı kolları
    for x in (52.0, 79.0):
        d += her_iki_yan(kutu(x, x + 4.0, YARI_GEN - 0.5, YARI_GEN + 2, 20.0, 21.0))
    # yan omuz çizgisi (karakter çizgisi)
    d += her_iki_yan(kutu(8.0, 92.0, YARI_GEN - 0.35, YARI_GEN + 2, 17.0, 17.5))
    return d


def detaylar_ekle():
    """Rozet yeri ve yan aynalar."""
    d = Manifold()
    rozet = Manifold.cylinder(1.4, 2.6, 2.6).rotate([0, -90, 0]).translate([0.6, 0, 12.2])
    d += rozet
    ayna = kutu(31.0, 34.0, YARI_GEN - 4.0, YARI_GEN + 2.2, 22.5, 26.8)
    d += her_iki_yan(ayna)
    return d


# ---------------------------------------------------------------- tekerlekler
def tekerlek(x, sag=True):
    y_dis = YARI_GEN - 0.6
    y_ic = YARI_GEN - CAMUR_DERIN - 0.3   # çamurluk iç duvarına gömülür -> tek parça
    lastik = y_silindir(TEKER_R, y_ic, y_dis, x, TEKER_R)
    # jant girintisi + göbek
    jant_oyuk = y_silindir(6.4, y_dis - 0.9, y_dis + 1.0, x, TEKER_R)
    gobek = y_silindir(2.4, y_dis - 1.0, y_dis - 0.1, x, TEKER_R)
    t = lastik - jant_oyuk + gobek
    # 5 kollu jant
    for i in range(5):
        a = i * 72.0
        kol = kutu(-0.7, 0.7, y_dis - 1.0, y_dis - 0.2, 0.0, 6.2).rotate([0, a, 0])
        t += kol.translate([x, 0, TEKER_R])
    return t if sag else t.mirror([0, 1, 0])


def camurluklar():
    c = Manifold()
    for x in (ON_AKS, ARKA_AKS):
        c += her_iki_yan(y_silindir(CAMUR_R, YARI_GEN - CAMUR_DERIN, YARI_GEN + 3, x, TEKER_R))
    return c


def araba():
    govde = alt_govde() + kabin()
    govde = govde - camurluklar()
    govde = govde - camlar() - detaylar_cikar()
    govde = govde + detaylar_ekle()
    for x in (ON_AKS, ARKA_AKS):
        govde = govde + tekerlek(x, True) + tekerlek(x, False)
    # sayısal hatadan doğan sıfır hacimli kırıntıları at
    parcalar = [p for p in govde.decompose() if p.volume() > 1.0]
    return Manifold.batch_boolean(parcalar, OpType.Add) if len(parcalar) > 1 else parcalar[0]


def main():
    ap = argparse.ArgumentParser(description="Opel tarzı hatchback STL üretici")
    ap.add_argument("--olcek", type=float, default=1.0, help="ölçek çarpanı (1.0 = ~98 mm boy)")
    ap.add_argument("--cikti", default="opel_tarzi_hatchback.stl")
    a = ap.parse_args()

    m = araba()
    if a.olcek != 1.0:
        m = m.scale([a.olcek] * 3)
    parca = m.decompose()
    mesh = m.to_mesh()
    tm = trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3],
                         faces=np.asarray(mesh.tri_verts), process=False)
    tm.export(a.cikti)

    ext = tm.bounds[1] - tm.bounds[0]
    print(f"Kaydedildi: {a.cikti}")
    print(f"  Boyut (mm): {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f}")
    print(f"  Üçgen: {len(tm.faces)}  Hacim: {tm.volume / 1000:.1f} cm3  "
          f"Su geçirmez: {tm.is_watertight}  Parça sayısı: {len(parca)}")


if __name__ == "__main__":
    main()
