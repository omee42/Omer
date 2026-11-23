#!/usr/bin/env python3
"""
arac3d — araç teknik çiziminden (blueprint) 3D baskı modeli (STL) üretir.

Girdi: bir JSON ayar dosyası (ölçüler + yan/üst/ön/arka görünüş görselleri ve kırpma kutuları).
Yöntem:
  1. Her görünüşten siluet ve iç çizgiler çıkarılır, bilinen ölçülerle mm'ye kalibre edilir.
  2. Gövde, siluetlerin kesişimi (görsel kabuk) olarak SDF ile kurulur:
       yan görünüş  -> tavan/kaput üst çizgisi ve alt çizgi
       üst görünüş  -> boy boyunca yarı genişlik
       ön/arka      -> yüksekliğe göre kesit biçimi (kabin daralması, omuz, tampon)
  3. Çizimdeki iç çizgiler (panel aralıkları, cam, far, ızgara hatları) yüzeye kazınır;
     her nokta yüzey normaline göre en uygun görünüşten okunur.
  4. İstenen bölgeler (camlar, farlar...) tohum noktasıyla seçilip oyulur.
  5. Tekerlek ve çamurluk kuyuları eklenir, marching cubes ile parça parça yüzey çıkarılır,
     su geçirmezlik korunarak sadeleştirilir ve STL yazılır.

Komutlar:
  python3 arac3d.py siluet  ayar.json     # siluet/çizgi çıkarımı + kontrol görselleri
  python3 arac3d.py model   ayar.json     # STL üretimi (tek parça ve/veya kit)
  python3 arac3d.py kontrol ayar.json     # model siluetini çizimle karşılaştırır (IoU + bindirme)

Gerekenler: pip install numpy scipy scikit-image pillow trimesh manifold3d matplotlib
"""
import argparse
import json
import math
import os
import time

import numpy as np
from PIL import Image
from scipy import ndimage

# ============================================================ genel yardımcılar


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


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


class Tablo1D:
    """Eşit aralıklı örneklerden doğrusal enterpolasyon + türev (uçlarda sabit)."""

    def __init__(self, x0, dx, deger):
        self.x0, self.dx = float(x0), float(dx)
        self.v = np.asarray(deger, float)
        self.t_ = np.gradient(self.v, self.dx)

    def _i(self, x):
        return np.clip((np.asarray(x, float) - self.x0) / self.dx, 0, len(self.v) - 1)

    def __call__(self, x):
        i = self._i(x)
        return np.interp(i, np.arange(len(self.v)), self.v)

    def t(self, x):
        i = self._i(x)
        return np.interp(i, np.arange(len(self.v)), self.t_)


def ayna_ayikla(genislik, pencere_px):
    """Profildeki dar çıkıntıları (dikiz aynaları) gri tonlu açma ile siler; geniş gövde
    biçimi (omuz, çamurluk) korunur. Aynalar ayarda 'aynalar' ile ayrıca eklenir."""
    w = max(3, int(round(pencere_px)) | 1)
    return np.minimum(genislik, ndimage.grey_opening(genislik, size=w, mode="nearest"))


# ============================================================ görünüş işleme


class Gorunum:
    """Kırpılmış ve yönü düzeltilmiş tek bir çizim görünüşü."""

    def __init__(self, ad, ayar, kok):
        self.ad = ad
        g = ayar["gorunumler"][ad]
        img = Image.open(os.path.join(kok, g["dosya"])).convert("RGB")
        a = np.asarray(img, float) / 255.0
        if "kutu" in g:
            x0, y0, x1, y1 = g["kutu"]
            a = a[y0:y1, x0:x1]
        if g.get("yatay_cevir", False) or g.get("on", "sol") == "sag":
            a = a[:, ::-1]
        self.rgb = a
        gri = a.mean(axis=2)
        doyma = a.max(axis=2) - a.min(axis=2)
        esik = g.get("cizgi_esik", ayar.get("cizgi_esik", 0.55))
        self.cizgi = gri < esik
        bos = (gri > g.get("bos_esik", ayar.get("bos_esik", 0.93))) & (doyma < 0.08)
        dolu = ~bos
        kap = int(g.get("kapat_px", ayar.get("kapat_px", 3)))
        m = ndimage.binary_closing(dolu, iterations=kap)
        m = ndimage.binary_fill_holes(m)
        etiket, n = ndimage.label(m)
        if n == 0:
            raise SystemExit(f"{ad}: siluet bulunamadı; kırpma kutusunu ve eşikleri kontrol edin")
        boy = ndimage.sum(m, etiket, range(1, n + 1))
        m = etiket == (1 + int(np.argmax(boy)))
        m = ndimage.binary_opening(m, iterations=1)
        self.sil = m
        rr, cc = np.nonzero(m)
        self.r0, self.r1, self.c0, self.c1 = rr.min(), rr.max(), cc.min(), cc.max()
        kenar = int(g.get("kenar_px", ayar.get("kenar_px", 3)))
        ic = ndimage.binary_erosion(m, iterations=kenar)
        self.ic_cizgi = self.cizgi & ic
        self.mesafe_px = ndimage.distance_transform_edt(~self.ic_cizgi)
        self.ic = ic


class Cizim:
    """Dört görünüşten kalibre edilmiş profiller ve çizgi örnekleyicileri."""

    def __init__(self, ayar, kok):
        self.ayar = ayar
        o = ayar["olcu_mm"]
        self.L, self.W, self.H = float(o["boy"]), float(o["genislik"]), float(o["yukseklik"])
        self.g = {ad: Gorunum(ad, ayar, kok) for ad in ("yan", "ust", "on", "arka") if ad in ayar["gorunumler"]}
        for ad in ("yan", "ust", "on"):
            if ad not in self.g:
                raise SystemExit(f"'{ad}' görünüşü gerekli (arka isteğe bağlı)")

        # --- yan: üst ve alt çizgi
        y = self.g["yan"]
        self.s_yan = self.L / (y.c1 - y.c0 + 1)
        self.yer_yan = y.r1 + 0.5
        kol = np.arange(y.c0, y.c1 + 1)
        ust = np.array([np.nonzero(y.sil[:, c])[0].min() for c in kol], float)
        alt = np.array([np.nonzero(y.sil[:, c])[0].max() for c in kol], float)
        ust = ndimage.gaussian_filter1d(ust, 1.2)
        alt = ndimage.gaussian_filter1d(alt, 1.2)
        z_ust = (self.yer_yan - ust) * self.s_yan
        z_ust = ayna_ayikla(z_ust, 380.0 / self.s_yan)   # kaputun üstüne taşan küçük çıkıntılar (anten vb.)
        xs_u = (kol - y.c0 + 0.5) * self.s_yan
        for ay in ayar.get("aynalar", []):                  # ayna bölgesinde üst çizgiyi komşulardan doldur
            ax0, ax1 = ay["x"] - ay.get("rx", 90) - 80, ay["x"] + ay.get("rx", 90) + 80
            ic = (xs_u > ax0) & (xs_u < ax1)
            if ic.any() and (~ic).any():
                z_ust[ic] = np.interp(xs_u[ic], xs_u[~ic], z_ust[~ic])
        z_alt = (self.yer_yan - (alt + 1)) * self.s_yan
        self.yan_yukseklik = (y.r1 - y.r0 + 1) * self.s_yan
        self.aks = self._akslar(z_alt)
        tek = ayar.get("teker", {})
        self.teker_R = tek.get("cap", 0.0) / 2 or self._teker_yaricap(z_alt)
        yerden = float(o.get("yerden", 130.0))
        xs = (kol - y.c0 + 0.5) * self.s_yan
        bolge = np.zeros_like(xs, bool)
        for xa in self.aks:
            bolge |= np.abs(xs - xa) < self.teker_R * 1.25
        z_alt = np.where(bolge, yerden, np.maximum(z_alt, yerden))
        z_alt = ndimage.gaussian_filter1d(z_alt, 2.0)
        self.UST = Tablo1D(xs[0], self.s_yan, z_ust)
        self.ALT = Tablo1D(xs[0], self.s_yan, z_alt)

        # --- üst görünüş: yarı genişlik
        u = self.g["ust"]
        self.s_ust = self.L / (u.c1 - u.c0 + 1)
        self.rc_ust = (u.r0 + u.r1) / 2.0
        kol = np.arange(u.c0, u.c1 + 1)
        yg = np.array([np.abs(np.nonzero(u.sil[:, c])[0] - self.rc_ust).max() + 0.5 for c in kol]) * self.s_ust
        yg = ayna_ayikla(yg, 450.0 / self.s_ust)
        yg = np.minimum(yg, self.W / 2)
        yg = ndimage.gaussian_filter1d(yg, 1.2)
        self.YG = Tablo1D((kol[0] - u.c0 + 0.5) * self.s_ust, self.s_ust, yg)

        # --- ön/arka: kesit biçimi
        self.kesit = {}
        for ad in ("on", "arka"):
            if ad not in self.g:
                continue
            v = self.g[ad]
            s = self.H / (v.r1 - v.r0 + 1)
            cc = (v.c0 + v.c1) / 2.0
            satir = np.arange(v.r0, v.r1 + 1)
            hw = np.array([(np.abs(np.nonzero(v.sil[r, :])[0] - cc).max() + 0.5) for r in satir]) * s
            hw = ayna_ayikla(hw, 300.0 / s)
            hw = np.minimum(hw, self.W / 2)
            hw = ndimage.gaussian_filter1d(hw, 1.2)
            z = (v.r1 + 0.5 - satir) * s
            sira = np.argsort(z)
            zz, hh = z[sira], hw[sira] / max(hw.max(), 1e-6)
            dz = s
            ornek_z = np.arange(0.0, self.H + dz, dz)
            self.kesit[ad] = Tablo1D(0.0, dz, np.interp(ornek_z, zz, hh, left=hh[0], right=0.0))
            v.s, v.cc = s, cc
        if "arka" not in self.kesit:
            self.kesit["arka"] = self.kesit["on"]

    # ---- aks tespiti: tekerleklerin yere değdiği sütun kümelerinin ortası
    def _akslar(self, z_alt):
        tek = self.ayar.get("teker", {})
        if "on_aks_x" in tek:
            on = float(tek["on_aks_x"])
            wb = float(tek.get("aks_mesafesi", self.ayar["olcu_mm"].get("aks_mesafesi")))
            return [on, on + wb]
        yere = z_alt < 0.02 * self.H
        idx = np.nonzero(yere)[0]
        if len(idx) < 2:
            raise SystemExit("aks bulunamadı: ayarda teker.on_aks_x verin")
        bosluk = np.argmax(np.diff(idx))
        on = idx[: bosluk + 1].mean()
        arka = idx[bosluk + 1:].mean()
        return [(on + 0.5) * self.s_yan, (arka + 0.5) * self.s_yan]

    def _teker_yaricap(self, z_alt):
        return 0.15 * self.L

    # ---- kesit (ön->arka harmanı)
    def kesit_f(self, x, z):
        t = sstep(0.3 * self.L, 0.7 * self.L, x)
        a, b = self.kesit["on"], self.kesit["arka"]
        f = (1 - t) * a(z) + t * b(z)
        df = (1 - t) * a.t(z) + t * b.t(z)
        return f, df

    # ---- çizgi örnekleyicileri (gerçek mm -> çizgiye uzaklık, mm)
    def _ornek(self, gor, satir, sutun, s):
        return ndimage.map_coordinates(gor.mesafe_px, [satir, sutun], order=1, mode="nearest") * s

    def cizgi_yan(self, x, z):
        g = self.g["yan"]
        return self._ornek(g, self.yer_yan - 0.5 - z / self.s_yan, g.c0 - 0.5 + x / self.s_yan, self.s_yan)

    def cizgi_ust(self, x, y):
        g = self.g["ust"]
        c = g.c0 - 0.5 + x / self.s_ust
        a = self._ornek(g, self.rc_ust - y / self.s_ust, c, self.s_ust)
        b = self._ornek(g, self.rc_ust + y / self.s_ust, c, self.s_ust)
        return np.minimum(a, b)

    def cizgi_onarka(self, ad, y, z):
        if ad not in self.g:
            return np.full(np.broadcast(y, z).shape, 1e6)
        g = self.g[ad]
        r = g.r1 + 0.5 - z / g.s
        a = self._ornek(g, r, g.cc - y / g.s, g.s)
        b = self._ornek(g, r, g.cc + y / g.s, g.s)
        return np.minimum(a, b)

    # ---- tohumdan bölge (cam, far...) SDF'i
    def bolge_sdf(self, b):
        g = self.g[b["gorunum"]]
        engel = ndimage.binary_dilation(g.cizgi, iterations=int(b.get("cizgi_kalinlastir", 1)))
        serbest = g.ic & ~engel
        etiket, _ = ndimage.label(serbest)
        c, r = b["px"]
        e = etiket[int(r), int(c)]
        if e == 0:
            raise SystemExit(f"bölge tohumu bir çizgiye ya da siluet dışına denk geliyor: {b}")
        bol = etiket == e
        bol = ndimage.binary_dilation(bol, iterations=int(b.get("cizgi_kalinlastir", 1)))
        ic = ndimage.distance_transform_edt(bol)
        dis = ndimage.distance_transform_edt(~bol)
        return g, (dis - ic)


# ============================================================ SDF


class Arac:
    def __init__(self, cizim, ayar):
        self.c = cizim
        self.a = ayar
        k = float(ayar.get("olcek", 24))
        self.k = k
        o = ayar["olcu_mm"]
        tek = ayar.get("teker", {})
        self.R = cizim.teker_R
        self.jant_R = tek.get("jant_cap", 0.75 * 2 * self.R) / 2
        self.tg = tek.get("genislik", 0.13 * cizim.W) / 2
        iz = (o.get("on_iz") or 0.86 * cizim.W)
        self.ty = iz / 2.0
        self.camur_R = self.R + tek.get("camur_bosluk", 30.0)
        self.kuyu_y = self.ty - self.tg - 30.0
        self.kol = int(tek.get("kol_sayisi", 5))
        d = ayar.get("detay", {})
        self.cizgi_der = d.get("cizgi_derinlik_mm", 0.35) * k
        self.cizgi_gen = d.get("cizgi_genislik_mm", 0.35) * k
        self.baskin = d.get("normal_esik", 0.55)
        self.yuvarlama = d.get("yuvarlama_mm", 1.5) * k
        self.bolgeler = [(b, *cizim.bolge_sdf(b)) for b in ayar.get("bolgeler", [])]

    def p(self, v):
        return v * self.k

    def temel(self, X, Y, Z):
        c = self.c
        yg, dyg = c.YG(X), c.YG.t(X)
        f, df = c.kesit_f(X, Z)
        W = yg * f
        d_y = (Y - W) / np.sqrt(1 + (dyg * f) ** 2 + (yg * df) ** 2)
        d_u = (Z - c.UST(X)) / np.sqrt(1 + c.UST.t(X) ** 2)
        d_a = (c.ALT(X) - Z) / np.sqrt(1 + c.ALT.t(X) ** 2)
        d = smax(d_y, d_u, self.yuvarlama)
        d = smax(d, d_a, self.yuvarlama * 0.5)
        d = np.maximum(d, np.maximum(-X - 5, X - c.L - 5))
        return d

    def kuyu(self, X, Y, Z, xa):
        r = np.sqrt((X - xa) ** 2 + (Z - self.R) ** 2) - self.camur_R
        return np.maximum(r, self.kuyu_y - Y)

    def teker(self, dx, dz, t):
        R, J, tg = self.R, self.jant_R, self.tg
        r = np.sqrt(dx * dx + dz * dz)
        th = np.arctan2(dz, dx)
        lastik = sd_kutu2(r, t, (J + R) / 2 + 4, 0.0, (R - J) / 2 - 4, tg, self.p(1.5))
        for tc in (-0.62 * tg, -0.2 * tg, 0.2 * tg, 0.62 * tg):
            lastik = np.maximum(lastik, -np.maximum(np.abs(t - tc) - self.p(0.22), (R - self.p(0.5)) - r))
        arka = np.maximum(np.maximum(r - J - 8, -tg - t), t + 15.0)
        dudak = sd_kutu2(r, t, J - 3, 0.75 * tg, 16.0, 0.18 * tg, 6.0)
        disk = np.maximum(r - 0.72 * J, np.abs(t + 8.0) - 12.0)
        a = np.mod(th + math.pi, 2 * math.pi / self.kol) - math.pi / self.kol
        delta = np.clip(0.00115 * (r - 0.3 * J), 0.0, None)
        kalin = 0.085 * J - 0.045 * (r - 0.28 * J)
        yuz = 0.9 * tg - 0.16 * (r - 0.28 * J)
        kol = np.maximum(np.maximum(r * np.abs(np.abs(a) - delta) - kalin, r - 0.96 * J), 0.24 * J - r)
        kol = np.maximum(kol, np.maximum(t - yuz, (yuz - 0.38 * tg) - t))
        gobek = np.maximum(r - 0.31 * J, np.maximum(25.0 - t, t - 0.9 * tg))
        jant = np.minimum(np.minimum(arka, dudak), disk)
        jant = smin(jant, kol, 6.0)
        jant = smin(jant, gobek, 8.0)
        return np.minimum(lastik, jant)

    def sdf(self, X, Y, Z, mod):
        """Tam SDF (Y >= 0 yarısı); detay kazıma için hacmin bütününe ihtiyaç duyulduğundan
        yüzey normali çağıran tarafça verilir (detay_uygula)."""
        d0 = self.temel(X, Y, Z)
        d = d0.copy()
        for i, xa in enumerate(self.c.aks):
            d = smax(d, -self.kuyu(X, Y, Z, xa), self.p(0.4))
            r = np.sqrt((X - xa) ** 2 + (Z - self.R) ** 2)
            if mod == "tek":
                d = np.minimum(d, np.maximum(r - self.p(2.0), np.maximum(self.kuyu_y - 20 - Y, Y - (self.ty - self.tg + 20))))
                d = np.minimum(d, self.teker(X - xa, Z - self.R, Y - self.ty))
            else:
                boss = np.maximum(r - self.p(2.3), np.maximum(self.kuyu_y - 12 - Y, Y - (self.ty - self.tg - 5)))
                d = np.minimum(d, boss)
        if mod == "tek":
            d = np.maximum(d, 6.0 - Z)
        for ay in self.a.get("aynalar", []):
            m = sd_elipsoid(X, Y, Z, (ay["x"], ay["y"], ay["z"]), (ay.get("rx", 90), ay.get("ry", 105), ay.get("rz", 65)))
            kol = sd_kutu3(X, Y, Z, (ay["x"], (ay["y"] + ay.get("kol_y", ay["y"] - 150)) / 2, ay["z"] - 40),
                           (35, abs(ay["y"] - ay.get("kol_y", ay["y"] - 150)) / 2 + 20, 14), 10)
            d = smin(d, smin(m, kol, 15), self.p(0.4))
        return d0, d

    def detay_uygula(self, d, d0, X, Y, Z, h):
        """Yüzey normaline göre doğru görünüşten iç çizgileri kazır ve bölgeleri oyar."""
        gx, gy, gz = np.gradient(d0, h)
        n = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-9
        nx, ny, nz = gx / n, gy / n, gz / n
        kabuk = d0 > -(self.cizgi_der + 3 * h)
        if not kabuk.any():
            return d
        c = self.c
        Xb, Yb, Zb = np.broadcast_arrays(X, Y, Z)
        sec = kabuk
        x, y, z = Xb[sec], Yb[sec], Zb[sec]
        nxs, nys, nzs = nx[sec], ny[sec], nz[sec]
        aday = [
            (np.abs(nys), lambda: c.cizgi_yan(x, z)),
            (nzs, lambda: c.cizgi_ust(x, y)),
            (-nxs, lambda: c.cizgi_onarka("on", y, z)),
            (nxs, lambda: c.cizgi_onarka("arka", y, z)),
        ]
        en = np.full(x.shape, 1e6)
        for bask, fn in aday:
            uy = bask > self.baskin
            if uy.any():
                v = np.full(x.shape, 1e6)
                v[uy] = np.broadcast_to(fn(), x.shape)[uy]
                en = np.minimum(en, v)
        maske = en - self.cizgi_gen / 2
        ds, d0s = d[sec], d0[sec]
        ds = np.maximum(ds, -np.maximum(-(d0s + self.cizgi_der), maske))
        # bölgeler (camlar, farlar ...)
        for b, g, sdf_px in self.bolgeler:
            der = b.get("derinlik_mm", 0.45) * self.k
            if b["gorunum"] == "yan":
                bask = np.abs(nys)
                s = c.s_yan
                r, cc = c.yer_yan - 0.5 - z / s, g.c0 - 0.5 + x / s
            elif b["gorunum"] == "ust":
                bask = nzs
                s = c.s_ust
                r, cc = c.rc_ust - np.where(b.get("taraf", 1) > 0, y, -y) / s, g.c0 - 0.5 + x / s
            else:
                bask = -nxs if b["gorunum"] == "on" else nxs
                s = g.s
                r, cc = g.r1 + 0.5 - z / s, g.cc - y / s
            m = ndimage.map_coordinates(sdf_px, [r, cc], order=1, mode="nearest") * s
            if b["gorunum"] in ("on", "arka", "ust"):
                r2 = r if b["gorunum"] != "ust" else c.rc_ust + y / s
                c2 = g.cc + y / s if b["gorunum"] != "ust" else cc
                m = np.minimum(m, ndimage.map_coordinates(sdf_px, [r2, c2], order=1, mode="nearest") * s)
            m = np.maximum(m, b.get("normal_esik", 0.4) - bask)
            ds = np.maximum(ds, -np.maximum(-(d0s + der), m))
        d = d.copy()
        d[sec] = ds
        return d


# ============================================================ meshleme


def _ters(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0


def govde_mesh(arac, mod, h_baski, parca=80, log=print):
    from skimage.measure import marching_cubes
    c = arac.c
    k = arac.k
    h = h_baski * k
    yk = max(c.W / 2, arac.ty + arac.tg) + 120
    for ay in arac.a.get("aynalar", []):
        yk = max(yk, ay["y"] + ay.get("ry", 105) + 60)
    x = np.arange(-40.0, c.L + 40.0, h)
    yh = np.arange(0.0, yk, h)
    z = np.arange(-20.0, max(c.UST.v.max(), c.H) + 60.0, h)
    y = np.concatenate([-yh[:0:-1], yh])
    nx = len(x)
    V, F, ofs = [], [], 0
    t0 = time.time()
    X3, Y3, Z3 = None, yh[None, :, None], z[None, None, :]
    for c0 in range(0, nx - 1, parca):
        c1 = min(nx - 1, c0 + parca)
        a0 = max(0, c0 - 2)                     # normal için 2 dilim taşma
        a1 = min(nx - 1, c1 + 2)
        X3 = x[a0:a1 + 1][:, None, None]
        d0, d = arac.sdf(X3, Y3, Z3, mod)
        d = arac.detay_uygula(d, d0, X3, Y3, Z3, h)
        d = d[c0 - a0:c0 - a0 + (c1 - c0 + 1)].astype(np.float32)
        tam = np.concatenate([d[:, :0:-1, :], d], axis=1)
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
        log(f"    {mod}: %{100 * c1 / (nx - 1):.0f} ({time.time() - t0:.0f} sn)")
    v = np.concatenate(V)
    f = np.concatenate(F)
    anahtar = np.round(v * 1e4).astype(np.int64)
    _, ilk, ters = np.unique(anahtar, axis=0, return_index=True, return_inverse=True)
    v = v[ilk]
    f = ters.reshape(-1)[f]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    v = v * h + np.array([x[0], y[0], z[0]])
    if _ters(v, f):
        f = f[:, ::-1]
    return v / k, f


def teker_mesh(arac, h_baski):
    from skimage.measure import marching_cubes
    k = arac.k
    h = h_baski * k
    R = arac.R
    a = np.arange(-R - 20, R + 20 + h, h)
    t = np.arange(-arac.tg - 20, arac.tg + 20 + h, h)
    DX, DZ, T = a[:, None, None], a[None, :, None], t[None, None, :]
    vol = arac.teker(DX, DZ, T)
    r = np.sqrt(DX ** 2 + DZ ** 2)
    vol = np.maximum(vol, -np.maximum(r - 0.90 * k, T - 40.0)).astype(np.float32)
    vol[0] = vol[-1] = 1e3
    vol[:, 0] = vol[:, -1] = 1e3
    vol[:, :, 0] = vol[:, :, -1] = 1e3
    v, f, _, _ = marching_cubes(vol, level=0.0137, spacing=(h, h, h), allow_degenerate=False)
    v += np.array([a[0], a[0], t[0]])
    if _ters(v, f):
        f = f[:, ::-1]
    return v / k, f


def sadelestir(v, f, tol):
    from manifold3d import Manifold, Mesh
    m = Manifold(Mesh(vert_properties=np.asarray(v, np.float32), tri_verts=np.asarray(f, np.uint32)))
    if m.status().name != "NoError":
        return v, f
    g = m.simplify(tol).to_mesh()
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
          f"su geçirmez={tm.is_watertight}")
    return tm


# ============================================================ komutlar


def ayar_oku(yol):
    with open(yol) as f:
        a = json.load(f)
    return a, os.path.dirname(os.path.abspath(yol))


def cikti_dizini(a, kok):
    d = os.path.join(kok, a.get("cikti", "cikti"))
    os.makedirs(d, exist_ok=True)
    return d


def komut_siluet(a, kok):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = Cizim(a, kok)
    out = cikti_dizini(a, kok)
    fig, ax = plt.subplots(len(c.g), 1, figsize=(14, 4.2 * len(c.g)))
    for axi, (ad, g) in zip(np.atleast_1d(ax), c.g.items()):
        goster = np.ones(g.sil.shape + (3,))
        goster[g.sil] = (0.80, 0.88, 1.0)
        goster[g.ic_cizgi] = (0.85, 0.1, 0.1)
        axi.imshow(goster)
        axi.set_title(f"{ad}: siluet (mavi), kazınacak iç çizgiler (kırmızı)")
        axi.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "siluetler.png"), dpi=70)
    rapor = {
        "yan_olcek_mm_px": c.s_yan, "ust_olcek_mm_px": c.s_ust,
        "yan_yukseklik_mm": round(c.yan_yukseklik), "beklenen_yukseklik_mm": c.H,
        "akslar_x_mm": [round(v) for v in c.aks], "aks_mesafesi_mm": round(c.aks[1] - c.aks[0]),
        "teker_yaricap_mm": round(c.teker_R),
    }
    if "aks_mesafesi" in a["olcu_mm"]:
        rapor["beklenen_aks_mesafesi_mm"] = a["olcu_mm"]["aks_mesafesi"]
    for ad in ("on", "arka"):
        if ad in c.g:
            rapor[f"{ad}_olcek_mm_px"] = c.g[ad].s
    print(json.dumps(rapor, indent=2, ensure_ascii=False))
    print("Görsel:", os.path.join(out, "siluetler.png"))


def komut_model(a, kok, args):
    c = Cizim(a, kok)
    arac = Arac(c, a)
    out = cikti_dizini(a, kok)
    ad = a.get("dosya_adi", "arac")
    h = args.h or a.get("izgara_mm", 0.15)
    tol = a.get("tolerans_mm", 0.005)
    parcalar = args.parca or a.get("parca", "hepsi")
    if parcalar in ("tek", "hepsi"):
        print("[tek parça]")
        v, f = govde_mesh(arac, "tek", h)
        kaydet(*sadelestir(v, f, tol), os.path.join(out, f"{ad}_tek_parca.stl"))
    if parcalar in ("kit", "hepsi"):
        print("[kit gövde]")
        v, f = govde_mesh(arac, "kit", h)
        d_aks = 1.05 * arac.k
        print(f"  (aks deliği Ø{2 * d_aks / arac.k:.2f} mm: dilimleyicide gerekirse delin)")
        kaydet(*sadelestir(v, f, tol), os.path.join(out, f"{ad}_govde.stl"))
        print("[teker]")
        v, f = teker_mesh(arac, min(h, 0.07))
        kaydet(*sadelestir(v, f, tol), os.path.join(out, f"{ad}_teker.stl"))


def komut_kontrol(a, kok):
    """Modelin dik izdüşümlerini çizim siluetleriyle karşılaştırır."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = Cizim(a, kok)
    arac = Arac(c, a)
    out = cikti_dizini(a, kok)
    h = 8.0
    x = np.arange(-40.0, c.L + 40, h)
    y = np.arange(-c.W / 2 - 60, c.W / 2 + 60, h)
    z = np.arange(-20.0, c.H + 60, h)
    dolu = np.zeros((len(x), len(y), len(z)), bool)
    for i in range(0, len(x), 40):
        X = x[i:i + 40][:, None, None]
        _, d = arac.sdf(X, np.abs(y)[None, :, None], z[None, None, :], "tek")
        dolu[i:i + 40] = d < 0
    sonuc = {}
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    def karsilastir(axi, ad, model_mask, ciz_mask, baslik):
        kesisim = (model_mask & ciz_mask).sum()
        birlesim = (model_mask | ciz_mask).sum()
        iou = kesisim / max(birlesim, 1)
        sonuc[ad] = round(float(iou), 4)
        goster = np.ones(model_mask.shape + (3,))
        goster[ciz_mask & ~model_mask] = (0.9, 0.2, 0.2)   # çizimde var, modelde yok
        goster[model_mask & ~ciz_mask] = (0.2, 0.4, 0.95)  # modelde var, çizimde yok
        goster[model_mask & ciz_mask] = (0.82, 0.82, 0.82)
        axi.imshow(goster, origin="lower")
        axi.set_title(f"{baslik}  IoU={iou:.3f}  (kırmızı: eksik, mavi: fazla)")
        axi.axis("off")

    def ciz_orneklem(ad, u, v, s, satir_f, sutun_f):
        g = c.g[ad]
        U, Vv = np.meshgrid(u, v)
        r = satir_f(Vv)
        cc = sutun_f(U)
        return ndimage.map_coordinates(g.sil.astype(float), [r, cc], order=0, mode="constant") > 0.5

    yan_model = dolu.any(axis=1).T
    g = c.g["yan"]
    yan_ciz = ciz_orneklem("yan", x, z, c.s_yan, lambda zz: c.yer_yan - 0.5 - zz / c.s_yan,
                           lambda xx: g.c0 - 0.5 + xx / c.s_yan)
    karsilastir(ax[0, 0], "yan", yan_model, yan_ciz, "Yan")
    ust_model = dolu.any(axis=2).T
    gu = c.g["ust"]
    ust_ciz = ciz_orneklem("ust", x, y, c.s_ust, lambda yy: c.rc_ust - yy / c.s_ust,
                           lambda xx: gu.c0 - 0.5 + xx / c.s_ust)
    karsilastir(ax[0, 1], "ust", ust_model, ust_ciz, "Üst")
    for axi, ad in ((ax[1, 0], "on"), (ax[1, 1], "arka")):
        if ad not in c.g:
            axi.axis("off")
            continue
        gv = c.g[ad]
        mdl = dolu.any(axis=0).T
        ciz = ciz_orneklem(ad, y, z, gv.s, lambda zz: gv.r1 + 0.5 - zz / gv.s, lambda yy: gv.cc + yy / gv.s)
        karsilastir(axi, ad, mdl, ciz, "Ön" if ad == "on" else "Arka")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "kontrol.png"), dpi=70)
    print(json.dumps({"IoU": sonuc}, indent=2, ensure_ascii=False))
    print("Görsel:", os.path.join(out, "kontrol.png"))


def main():
    ap = argparse.ArgumentParser(description="Araç teknik çiziminden 3D baskı modeli")
    ap.add_argument("komut", choices=["siluet", "model", "kontrol"])
    ap.add_argument("ayar")
    ap.add_argument("--h", type=float, default=None, help="baskı ölçeğinde ızgara (mm)")
    ap.add_argument("--parca", choices=["tek", "kit", "hepsi"], default=None)
    args = ap.parse_args()
    a, kok = ayar_oku(args.ayar)
    if args.komut == "siluet":
        komut_siluet(a, kok)
    elif args.komut == "model":
        komut_model(a, kok, args)
    else:
        komut_kontrol(a, kok)


if __name__ == "__main__":
    main()
