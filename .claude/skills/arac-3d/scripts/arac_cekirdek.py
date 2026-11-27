#!/usr/bin/env python3
"""
arac_cekirdek — fotoğraftan araç maketi için araçtan bağımsız çekirdek.

Araca özgü betik (ör. 3d-model/astra_sedan_olustur.py) gövdeyi işaretli mesafe alanı (SDF) olarak tanımlar;
bu modül şunları sağlar:
  * SDF yardımcıları: Egri (PCHIP), smin/smax, sstep/sstep5, poligon/çizgi/kutu/elipsoid SDF'leri, YPoligon,
    petek ızgarası, oyuk/doldur, yumusat (Catmull-Rom), duz_yazi_2d (çizgi yazı tipi).
  * maskeli 3B Gauss yumuşatma (yerel yüzey düzeltme; sınır kuralları için SKILL.md).
  * Blok: yarım hacimde alt bölge kesme (detayları yalnız gerektiği yerde hesaplamak için).
  * govde_mesh: çok çekirdekli, parça parça marching cubes (sağ-sol ayna simetrisi, isteğe bağlı asimetrik detay).
  * sadelestir / yuzey_duzenle / katlanma_cevir / kaydet: su geçirmezliği koruyan sadeleştirme ve STL yazımı.
  * teker_mesh: kit tekerleği ağı.
Koordinatlar gerçek mm: x önden arkaya, +y sağ, z yukarı (yer z=0). Baskıya en sonda 1/olcek ile küçültülür.
"""
import math
import os
import time

import numpy as np
from scipy.interpolate import PchipInterpolator
from manifold3d import CrossSection, JoinType

K = 24.0  # detay ölçüleri 1:K baskıdaki mm olarak verilir; p() gerçek mm'ye çevirir


def p(v):
    return v * K



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


def sstep5(e0, e1, x):
    """beşinci derece yumuşak basamak (C2: uçlarda eğim ve eğrilik sıfır)."""
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def sstep5_t(e0, e1, x):
    """sstep5'in x'e göre türevi."""
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return 30.0 * t * t * (t - 1.0) ** 2 / (e1 - e0)


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
_FONT["c"] = [_elips_cizgi(0.31, 0.21, 0.31, 0.21, math.radians(58), math.radians(302))]   # logo: c ağzı geniş (eski ±42° kapalı "e" gibi okunuyordu)


_FONT.update({
    "A": [[(0, 0), (0.3, 1), (0.6, 0)], [(0.12, 0.38), (0.48, 0.38)]],
    "S": [[(0.58, 0.86), (0.44, 1), (0.16, 1), (0.02, 0.86), (0.02, 0.66), (0.16, 0.54), (0.44, 0.46), (0.58, 0.34),
           (0.58, 0.14), (0.44, 0), (0.14, 0), (0, 0.14)]],
    "D": [[(0, 0), (0, 1), (0.36, 1), (0.6, 0.78), (0.6, 0.22), (0.36, 0), (0, 0)]],
    "I": [[(0.12, 1), (0.12, 0)]],
    "E": [[(0.6, 1), (0, 1), (0, 0), (0.6, 0)], [(0, 0.52), (0.46, 0.52)]],
    "P": [[(0, 0), (0, 1), (0.44, 1), (0.6, 0.86), (0.6, 0.62), (0.44, 0.48), (0, 0.48)]],
    "O": [[(0.15, 0), (0.45, 0), (0.6, 0.15), (0.6, 0.85), (0.45, 1), (0.15, 1), (0, 0.85), (0, 0.15), (0.15, 0)]],
    "N": [[(0, 0), (0, 1), (0.6, 0), (0.6, 1)]],
    "L": [[(0, 1), (0, 0), (0.55, 0)]],
    "K": [[(0, 0), (0, 1)], [(0.6, 1), (0, 0.42)], [(0.2, 0.58), (0.6, 0)]],
    ".": [[(0.05, 0), (0.1, 0.05), (0.05, 0.1), (0, 0.05), (0.05, 0)]],
})


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




def _ters(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0


# ============================================================ maskeli 3B yumuşatma
def maskeli_yumusat(ham_fn, maske_fn, X, Y, Z, a=50.0, sigma=0.85):
    """ham_fn alanını maske_fn ağırlığıyla 3x3x3 Gauss şablonuyla (aralık a mm) yumuşatır.
    Kurallar (GTC arka çeyreğinden): maske, şablon (±1.7a) keskin bir öğeye (kemer dudağı, stop/tampon kırımı,
    cam kenarı) uzanmayacak kadar ondan uzakta bitmeli; panel aralığının iki yanı aynı ağırlıkta olmalı; geçiş
    bandı en az ~4a geniş olmalı. Yoksa sınır boyunca 1-4 mm kırışık/basamak çıkar."""
    d = ham_fn(X, Y, Z)
    Xb, Yb, Zb = np.broadcast_arrays(X, Y, Z)
    wb = maske_fn(Xb, Yb, Zb)
    sec = wb > 1e-4
    if not np.any(sec):
        return d
    of = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], float)
    w = np.exp(-0.5 * (of ** 2).sum(1) / sigma ** 2)
    w /= w.sum()
    d = np.array(np.broadcast_to(d, Xb.shape), dtype=float)
    xs, ys, zs = Xb[sec], Yb[sec], Zb[sec]
    acc = np.zeros(xs.shape)
    for (ox, oy, oz), wk in zip(of * a, w):
        acc += wk * ham_fn(xs + ox, ys + oy, zs + oz)
    d[sec] = d[sec] + wb[sec] * (acc - d[sec])
    return d


# ============================================================ blok (yarım hacimde alt bölge)
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


# ============================================================ meshleme (parça parça, çok çekirdekli)
_SDF = None   # fork ile çocuk süreçlere geçer: f(Blok, mod) -> yarım hacim SDF (y >= 0)


def _dilim_sdf(a):
    xs, yh, z, mod = a
    return _SDF(Blok(xs, yh, z), mod).astype(np.float32)


def govde_mesh(sdf, mod, h_baski, n, kutu, asimetrik=None, parca=96, dilim=4, seviye=0.0137):
    """sdf(Blok, mod): y >= 0 yarısında gerçek mm SDF; sol yarı ayna ile elde edilir.
    kutu = (x0, x1, y_yari, z0, z1) gerçek mm. asimetrik(tam, x, y, z, mod): tam hacimde yerinde değişiklik.
    Çekirdek sayısı ARAC_NP ortam değişkeniyle (varsayılan: hepsi). Döndürür: (köşeler baskı mm, üçgenler)."""
    import multiprocessing as mp
    from skimage.measure import marching_cubes
    global _SDF
    _SDF = sdf
    np_say = int(os.environ.get("ARAC_NP", str(os.cpu_count() or 1)))
    havuz = mp.get_context("fork").Pool(np_say) if np_say > 1 else None
    h = h_baski * n  # gerçek mm cinsinden ızgara
    x0, x1, yy, z0, z1 = kutu
    x = np.arange(x0, x1, h)
    yh = np.arange(0.0, yy, h)
    z = np.arange(z0, z1, h)
    y = np.concatenate([-yh[:0:-1], yh])
    nx = len(x)
    V, F, ofs = [], [], 0
    t0 = time.time()
    for c0 in range(0, nx - 1, parca):
        c1 = min(nx - 1, c0 + parca)
        xs = x[c0:c1 + 1]
        yarim = np.empty((len(xs), len(yh), len(z)), np.float32)
        isler = [(xs[i0:i0 + dilim], yh, z, mod) for i0 in range(0, len(xs), dilim)]
        sonuc = havuz.map(_dilim_sdf, isler) if havuz is not None else map(_dilim_sdf, isler)
        for i0, blok in zip(range(0, len(xs), dilim), sonuc):
            yarim[i0:i0 + len(blok)] = blok
        tam = np.concatenate([yarim[:, :0:-1, :], yarim], axis=1)
        del yarim
        if asimetrik is not None:
            asimetrik(tam, xs, y, z, mod)
        if c0 == 0:
            tam[0] = 1e3
        if c1 == nx - 1:
            tam[-1] = 1e3
        tam[:, 0] = tam[:, -1] = 1e3
        tam[:, :, 0] = tam[:, :, -1] = 1e3
        if tam.min() < 0:
            v, f, _, _ = marching_cubes(tam, level=seviye, allow_degenerate=False)
            v = v.astype(np.float64)
            v[:, 0] += c0
            V.append(v)
            F.append(f + ofs)
            ofs += len(v)
        print(f"    {mod}: %{100 * c1 / (nx - 1):.0f} ({time.time() - t0:.0f} sn)", flush=True)
    if havuz is not None:
        havuz.close()
        havuz.join()
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


def sadelestir(v, f, tolerans, duzenle=True):
    """Su geçirmezliği koruyan sadeleştirme: yüzey en fazla 'tolerans' mm sapar.
    duzenle=True: ardından ince uzun üçgen yelpazeleri ham yüzeye sadık kalarak düzeltilir (yuzey_duzenle)."""
    from manifold3d import Manifold, Mesh
    m = Manifold(Mesh(vert_properties=np.asarray(v, np.float32), tri_verts=np.asarray(f, np.uint32)))
    if m.status().name != "NoError":
        print(f"    uyarı: manifold kontrolü {m.status()}, sadeleştirme atlandı")
        return v, f
    m = m.simplify(tolerans)
    g = m.to_mesh()
    del m
    vs = np.asarray(g.vert_properties)[:, :3].astype(np.float64)
    fs = np.asarray(g.tri_verts).astype(np.int64)
    del g
    if duzenle:
        vs, fs = yuzey_duzenle(vs, fs, v, f, tolerans)
    return vs, fs


def _ham_normaller(v, f, parca=4_000_000):
    """Ham MC ağının alan ağırlıklı köşe normalleri (float32); bellek için parça parça."""
    N = np.zeros((len(v), 3))
    for i0 in range(0, len(f), parca):
        ff = f[i0:i0 + parca]
        fn = np.cross(v[ff[:, 1]] - v[ff[:, 0]], v[ff[:, 2]] - v[ff[:, 0]])
        for k in range(3):
            for c in range(3):
                N[:, c] += np.bincount(ff[:, k], fn[:, c], minlength=len(v))
        del fn
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-30
    return N.astype(np.float32)


def yuzey_duzenle(V, F, Vr, Fr, tol, tur=8, kilit_aci=20.0, log=print):
    """Sadeleştirme sonrası üçgen kalitesi. simplify() düz/az eğrili kaportada köşeleri seyrek ve düzensiz
    bırakır; aradaki ince uzun yelpaze üçgenlerinin normali (köşeleri yüzeyde olsa bile eğrilik yüzünden)
    gerçek yüzeyden 5-50° sapar ve gölgede ışınsal çizgi / faset olarak görünür. Ham MC ağı bu hatayı taşımaz.
    Her turda: (1) kenar çevirme (Delaunay: şekli düzelten; ya da yüz normalinin köşelerdeki ham normallerden
    sapmasını belirgin azaltan), (2) köşeleri teğet düzlemde alan ağırlıklı ağırlık merkezine doğru
    gevşetip ham MC yüzeyine (en yakın ham köşenin teğet düzlemi) geri izdüşürme. Her adımda yüz merkezi ve kenar
    ortalarının ham yüzeyden sapması 'tol' (ya da yüzün sadeleştirmeden gelen sapması) ile sınırlı; yeni yüzün
    normali köşelerindeki ham normallerden 25°'den fazla sapamaz (keskin kenar üzerinden çevirme/kayma olmaz).
    Ham normalleri 0.15 mm çevrede 'kilit_aci'dan fazla dönen (keskin detay: derz, yazı, logo kenarı) köşeler
    kilitli. Köşe sayısı ve topoloji (kapalı manifold) korunur; yalnız kenar çevrilir ve köşe kayar."""
    from scipy.spatial import cKDTree
    t0 = time.time()
    V = np.asarray(V, np.float64).copy()
    F = np.asarray(F, np.int64).copy()
    nv = len(V)
    Nr = _ham_normaller(Vr, Fr)
    agac = cKDTree(Vr)
    i0 = agac.query(V, k=1, workers=-1)[1]
    n0 = Nr[i0].astype(np.float64)  # sadeleştirilmiş köşeler ham köşelerin alt kümesi
    kilit = np.zeros(nv, bool)
    for a in range(0, nv, 200_000):  # bellek: parça parça 16 komşu
        ik = agac.query(V[a:a + 200_000], k=16, workers=-1)[1]
        kilit[a:a + 200_000] = np.einsum('ikj,ij->ik', Nr[ik], n0[a:a + 200_000]).min(1) < np.cos(np.radians(kilit_aci))

    def sapma(P):
        i = agac.query(P, k=1, workers=-1)[1]
        return np.abs(((P - Vr[i]) * Nr[i]).sum(1))

    def izdus(P):
        i = agac.query(P, k=1, workers=-1)[1]
        n = Nr[i].astype(np.float64)
        return P - ((P - Vr[i]) * n).sum(1)[:, None] * n, i

    def yuz_normal(T, P):
        n = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
        l = np.linalg.norm(n, axis=1)
        return n / (l[:, None] + 1e-30), l

    def yuz_sapma(T, P):
        a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
        return sapma(np.concatenate([(a + b + c) / 3, (a + b) / 2, (b + c) / 2, (c + a) / 2])).reshape(4, -1).max(0)

    cos_n = np.cos(np.radians(25.0))

    def yuz_tutarli(T, P):
        n, l = yuz_normal(T, P)
        ok = l > 1e-12
        for k in range(3):
            ok &= (n * n0[T[:, k]]).sum(1) > cos_n
        return ok

    def nhata(T):  # yüz normalinin köşelerdeki ham normallerden en büyük sapması (°)
        n, _ = yuz_normal(T, V)
        c = np.minimum(np.minimum((n * n0[T[:, 0]]).sum(1), (n * n0[T[:, 1]]).sum(1)), (n * n0[T[:, 2]]).sum(1))
        return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))

    def aci(P, p, q, r):  # p köşesindeki iç açı
        u = P[q] - P[p]
        w = P[r] - P[p]
        return np.arctan2(np.linalg.norm(np.cross(u, w), axis=1), (u * w).sum(1))

    def minaci(T):
        return np.minimum(np.minimum(aci(V, T[:, 0], T[:, 1], T[:, 2]), aci(V, T[:, 1], T[:, 2], T[:, 0])),
                          aci(V, T[:, 2], T[:, 0], T[:, 1]))

    tol_f = np.maximum(tol, yuz_sapma(F, V))  # sadeleştirmenin zaten aştığı yüzlerde mevcut sapmayı aşma
    top_cev = 0
    for it in range(tur):
        # (1) kenar çevirme: karşı açılar toplamı > 180° (Delaunay) ya da normal hatası > 3°; en iyi adaylardan
        #     bağımsız küme (her yüz tek çevirmede, aynı yeni kenar bir kez)
        cev = 0
        for _ in range(4):
            E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
            fid = np.tile(np.arange(len(F)), 3)
            ops = np.concatenate([F[:, 2], F[:, 0], F[:, 1]])
            key = np.minimum(E[:, 0], E[:, 1]) * nv + np.maximum(E[:, 0], E[:, 1])
            o = np.argsort(key, kind='stable')
            ks = key[o]
            if not np.array_equal(ks[0::2], ks[1::2]):
                log("    uyarı: ağ kapalı manifold değil, yüzey düzenleme atlandı")
                return np.asarray(V), F
            ilk, iki = o[0::2], o[1::2]
            f0, f1 = fid[ilk], fid[iki]
            A, B, c, d = E[ilk, 0], E[ilk, 1], ops[ilk], ops[iki]  # f0 = (A, B, c), f1 = (B, A, d)
            toplam = aci(V, c, A, B) + aci(V, d, A, B)
            hf = nhata(F)
            h_eski = np.maximum(hf[f0], hf[f1])
            serb = ~(kilit[A] & kilit[B] & kilit[c] & kilit[d])
            s = np.nonzero(((toplam > np.pi + 1e-3) | ((h_eski > 3.0) & serb)) & (c != d))[0]
            N1 = np.c_[A[s], d[s], c[s]]
            N2 = np.c_[d[s], B[s], c[s]]
            ok = yuz_tutarli(N1, V) & yuz_tutarli(N2, V)
            h_yeni = np.maximum(nhata(N1), nhata(N2))
            # Delaunay (şekil) çevirmesi normal hatasını büyütmesin; ya da normal hatasını belirgin azaltan çevirme
            delo = toplam[s] > np.pi + 1e-3
            delo &= np.minimum(minaci(N1), minaci(N2)) > np.minimum(minaci(F[f0[s]]), minaci(F[f1[s]]))
            ok &= (delo & (h_yeni <= h_eski[s] + 1.0)) | (h_yeni < h_eski[s] - 2.0)
            oncelik = np.where(h_yeni < h_eski[s] - 2.0, 10.0 + h_eski[s] - h_yeni, toplam[s])
            cd = np.minimum(c[s], d[s]) * nv + np.maximum(c[s], d[s])
            ok &= ~np.isin(cd, ks)
            k = np.nonzero(ok)[0]
            tl = np.minimum(tol_f[f0[s[k]]], tol_f[f1[s[k]]])
            ok[k] = (yuz_sapma(N1[k], V) <= tl) & (yuz_sapma(N2[k], V) <= tl)
            s, N1, N2, cd, oncelik = s[ok], N1[ok], N2[ok], cd[ok], oncelik[ok]
            if len(s) == 0:
                break
            o2 = np.argsort(-oncelik, kind='stable')
            s, N1, N2, cd = s[o2], N1[o2], N2[o2], cd[o2]
            sira = np.arange(len(s))
            en_iyi = np.full(len(F), len(s))
            np.minimum.at(en_iyi, f0[s], sira)
            np.minimum.at(en_iyi, f1[s], sira)
            sec = np.nonzero((en_iyi[f0[s]] == sira) & (en_iyi[f1[s]] == sira))[0]
            sec = sec[np.unique(cd[sec], return_index=True)[1]]  # aynı yeni kenar iki kez oluşmasın
            if len(sec) == 0:
                break
            a0, a1 = f0[s[sec]], f1[s[sec]]
            F[a0] = N1[sec]
            F[a1] = N2[sec]
            tl = np.minimum(tol_f[a0], tol_f[a1])
            tol_f[a0] = tl
            tol_f[a1] = tl
            cev += len(sec)
        # (2) teğet gevşetme + ham yüzeye izdüşüm
        n_eski, l2 = yuz_normal(F, V)
        mer = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3
        W = np.zeros(nv)
        T = np.zeros((nv, 3))
        for k in range(3):
            W += np.bincount(F[:, k], l2, minlength=nv)
            for j in range(3):
                T[:, j] += np.bincount(F[:, k], l2 * mer[:, j], minlength=nv)
        dv = T / (W[:, None] + 1e-30) - V
        dv -= (dv * n0).sum(1)[:, None] * n0
        oynayan = ~kilit
        Vn = V.copy()
        Vn[oynayan], _ = izdus(V[oynayan] + 0.5 * dv[oynayan])
        tut_eski = yuz_tutarli(F, V)
        degisen = oynayan.copy()
        while True:  # bozulan yüzlerin köşelerini geri al (her turda en az bir köşe; sonlu)
            fm = np.nonzero(degisen[F].any(1) & oynayan[F].any(1))[0]  # yalnız son adımda değişen yüzler
            Fm = F[fm]
            n_yeni, l_yeni = yuz_normal(Fm, Vn)
            bad = (l_yeni < 1e-12) | ((n_yeni * n_eski[fm]).sum(1) < 0.5) | (yuz_sapma(Fm, Vn) > tol_f[fm])
            bad |= tut_eski[fm] & ~yuz_tutarli(Fm, Vn)
            if not bad.any():
                break
            geri = np.unique(Fm[bad])
            geri = geri[oynayan[geri]]
            Vn[geri] = V[geri]
            oynayan[geri] = False
            degisen[:] = False
            degisen[geri] = True
        V = Vn
        if oynayan.any():
            n0[oynayan] = Nr[agac.query(V[oynayan], k=1, workers=-1)[1]]
        top_cev += cev
    log(f"  yüzey düzenleme: {tur} tur, {top_cev} kenar çevrildi, son turda {int(oynayan.sum())} köşe kaydı, "
        f"{int(kilit.sum())}/{nv} keskin köşe kilitli ({time.time() - t0:.0f} sn)")
    return V, F


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

def teker_mesh(teker_sdf, h_baski, n, tolerans, yol, yaricap=360.0, yari_gen=140.0, delik_r=0.90, delik_t=40.0):
    """Kit tekerleği: teker_sdf(dx, dz, t) gerçek mm (t: dışa doğru eksen). Aksta delik_r (baskı mm) delik."""
    from skimage.measure import marching_cubes
    h = h_baski * n
    a = np.arange(-yaricap, yaricap + h, h)
    t = np.arange(-yari_gen, yari_gen + h, h)
    DZ, T = a[None, :, None], t[None, None, :]
    vol = np.empty((len(a), len(a), len(t)), np.float32)
    for i0 in range(0, len(a), 24):
        DX = a[i0:i0 + 24, None, None]
        v = teker_sdf(DX, DZ, T)
        delik = np.maximum(np.sqrt(DX ** 2 + DZ ** 2) - p(delik_r), T - delik_t)
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
