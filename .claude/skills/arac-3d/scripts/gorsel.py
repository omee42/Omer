#!/usr/bin/env python3
"""gorsel — model ağını gölgeli ve zebra (yansıma çizgisi) olarak dik izdüşümde çizer; önizleme paftaları.

Zebra: bakış ışınının yüzeyden yansıyan yönünün yükseklik açısına göre siyah/beyaz bant. Parlak boyadaki yansıma
gibidir: bantlar düzgün, geniş ve sürekli akıyorsa yüzey pürüzsüzdür; kıvrılan, kopan, halka yapan bant çukur/tümsek
ya da eğrilik sıçramasıdır. Göz kararı gölgeli görüntüde görünmeyen 1-2 mm'lik kusurlar zebrada hemen görünür.

Kullanım:
  python3 gorsel.py zebra  <ag.stl|bolge_onek> <cikis.png> cx cy cz az el genislik_mm [olcek]
  python3 gorsel.py pafta  <model.stl> <cikis.png> genel|detay [olcek]
Bölge öneki verilirse <onek>_L.npz ve <onek>_R.npz (bolge_ag.py çıktısı, gerçek mm) birleştirilir.
az: 0 sağ yandan (+y'den) bakış, -90 arkadan, 90 önden; el: yukarıdan bakış açısı.
"""
import sys
import numpy as np
from PIL import Image


def raster(uv, zc, V, F, W, H, C):
    x, y = uv[:, 0][F], uv[:, 1][F]
    dep = zc[F]
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    sg = np.sign(((V[F].mean(1) - C) * n).sum(1))
    n = -n * sg[:, None]
    keep = (x.max(1) >= 0) & (x.min(1) < W) & (y.max(1) >= 0) & (y.min(1) < H) & (dep.min(1) > 0)
    x, y, dep, n = x[keep], y[keep], dep[keep], n[keep]
    x0 = np.floor(x.min(1)).astype(int); y0 = np.floor(y.min(1)).astype(int)
    sp = np.ceil(np.maximum(x.max(1) - x0, y.max(1) - y0)).astype(int) + 1
    zb = np.full(W * H, np.inf); nb = np.zeros((W * H, 3))
    den = (y[:, 1] - y[:, 2]) * (x[:, 0] - x[:, 2]) + (x[:, 2] - x[:, 1]) * (y[:, 0] - y[:, 2])
    ok = np.abs(den) > 1e-12
    big = ok & (sp > 64)
    for i in np.nonzero(big)[0]:
        gx, gy = np.meshgrid(np.arange(max(x0[i], 0), min(x0[i] + sp[i], W)) + 0.5, np.arange(max(y0[i], 0), min(y0[i] + sp[i], H)) + 0.5)
        xs, ys = x[i], y[i]
        w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / den[i]
        w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / den[i]
        w2 = 1 - w0 - w1
        m = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        dd = w0[m] * dep[i, 0] + w1[m] * dep[i, 1] + w2[m] * dep[i, 2]
        pix = (gy[m] - 0.5).astype(int) * W + (gx[m] - 0.5).astype(int)
        upd = dd < zb[pix]
        zb[pix[upd]] = dd[upd]; nb[pix[upd]] = n[i]
    for smax_ in (2, 4, 8, 16, 32, 64):
        sel = ok & (sp <= smax_) & (sp > smax_ // 2 if smax_ > 2 else True)
        if not sel.any():
            continue
        xs, ys, ds, ns, xx0, yy0, dn = x[sel], y[sel], dep[sel], n[sel], x0[sel], y0[sel], den[sel]
        for oy in range(smax_):
            for ox in range(smax_):
                gx = xx0 + ox + 0.5; gy = yy0 + oy + 0.5
                w0 = ((ys[:, 1] - ys[:, 2]) * (gx - xs[:, 2]) + (xs[:, 2] - xs[:, 1]) * (gy - ys[:, 2])) / dn
                w1 = ((ys[:, 2] - ys[:, 0]) * (gx - xs[:, 2]) + (xs[:, 0] - xs[:, 2]) * (gy - ys[:, 2])) / dn
                w2 = 1 - w0 - w1
                m = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
                px = (xx0 + ox)[m]; py = (yy0 + oy)[m]
                ib = (px >= 0) & (px < W) & (py >= 0) & (py < H)
                idx = np.nonzero(m)[0][ib]
                dd = (np.stack([w0[m][ib], w1[m][ib], w2[m][ib]], 1) * ds[idx]).sum(1)
                pix = py[ib] * W + px[ib]
                o = np.lexsort((-dd, pix))
                pix, dd, idx = pix[o], dd[o], idx[o]
                upd = dd < zb[pix]
                zb[pix[upd]] = dd[upd]; nb[pix[upd]] = ns[idx[upd]]
    return zb.reshape(H, W), nb.reshape(H, W, 3)


def golge(zb, nb, fw, rt, dn):
    L1 = -fw * 0.6 - dn * 0.7 - rt * 0.35; L1 /= np.linalg.norm(L1)
    L2 = -fw * 0.5 + rt * 0.6 - dn * 0.2; L2 /= np.linalg.norm(L2)
    sh = 0.15 + 0.7 * np.clip(nb @ L1, 0, 1) + 0.3 * np.clip(nb @ L2, 0, 1)
    img = np.ones(zb.shape + (3,)) * 0.97; fin = np.isfinite(zb)
    img[fin] = np.clip(sh[fin][:, None] * np.array([0.78, 0.8, 0.84]), 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))


# ---------------- cameras: each returns (foto, W, H, projfn(P)->(uv,z), C, fw, rt, dn)


def kamera(az, el):
    a, e = np.radians(az), np.radians(el)
    w = np.array([-np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)])
    fw = -w
    rt = np.cross(fw, [0, 0, 1.0]); rt /= np.linalg.norm(rt)
    return fw, rt, np.cross(fw, rt)


def ciz(V, F, merkez, az, el, gen, W=1000, H=700, zebra=False, per=8):
    """Dik izdüşüm; gen: görüntü genişliğinin karşılığı (gerçek mm). per: 90°'deki bant çifti sayısı."""
    fw, rt, dn = kamera(az, el)
    c = np.asarray(merkez, float); s = W / gen; P = V - c
    uv = np.stack([W / 2 + s * (P @ rt), H / 2 + s * (P @ dn)], 1); z = P @ fw + 1e5
    U = uv[:, 0][F]; Vv = uv[:, 1][F]
    Fs = F[(U.max(1) > 0) & (U.min(1) < W) & (Vv.max(1) > 0) & (Vv.min(1) < H)]
    zb, nb = raster(uv, z, V, Fs, W, H, c - fw * 1e5)
    if not zebra:
        return golge(zb, nb, fw, rt, dn)
    r = fw - 2 * (nb @ fw)[..., None] * nb
    ang = np.degrees(np.arcsin(np.clip(r[..., 2], -1, 1)))
    st = np.floor(ang / (180 / (2 * per))) % 2
    img = np.ones(zb.shape + (3,)) * 0.97; fin = np.isfinite(zb)
    col = np.where(st[..., None] > 0, np.array([0.1, 0.1, 0.12]), np.array([0.95, 0.95, 0.97]))
    img[fin] = col[fin]
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def yukle(yol, olcek=24.0):
    """STL (baskı mm -> gerçek mm'ye çevrilir) ya da bölge öneki (_L/_R.npz, gerçek mm)."""
    if yol.endswith(".stl"):
        import trimesh
        t = trimesh.load(yol)
        return np.asarray(t.vertices, float) * olcek, np.asarray(t.faces, np.int64)
    a = np.load(yol + "_L.npz"); b = np.load(yol + "_R.npz")
    V = np.vstack([a["v"], b["v"]]).astype(float)
    F = np.vstack([a["f"], b["f"] + len(a["v"])]).astype(np.int64)
    return V, F





def rot(el, az):
    a, e = np.radians(az), np.radians(el)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Rz


def render(t, el, az, W, H, renk=(0.72, 0.74, 0.78), zoom=1.0, merkez=None):
    V = t.vertices - (t.bounds.mean(0) if merkez is None else merkez)
    F = t.faces
    VN = t.vertex_normals
    R = rot(el, az)
    P = V @ R.T
    N = VN @ R.T
    ext = np.ptp(P[:, [0, 2]], axis=0)
    s = zoom * min(W * 0.92 / ext[0], H * 0.92 / ext[1])
    sx = P[:, 0] * s + W / 2
    sy = H / 2 - P[:, 2] * s
    dep = P[:, 1]
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    keep = fn[:, 1] < 0
    F = F[keep]
    x = sx[F]; y = sy[F]; z = dep[F]
    x0 = np.floor(x.min(1)).astype(int); y0 = np.floor(y.min(1)).astype(int)
    span = int(np.ceil(max((x.max(1) - x0).max(), (y.max(1) - y0).max()))) + 1
    span = min(span, 12)
    den = (y[:, 1] - y[:, 2]) * (x[:, 0] - x[:, 2]) + (x[:, 2] - x[:, 1]) * (y[:, 0] - y[:, 2])
    ok = np.abs(den) > 1e-12
    F, x, y, z, x0, y0, den = F[ok], x[ok], y[ok], z[ok], x0[ok], y0[ok], den[ok]
    pix_all, dep_all, nrm_all = [], [], []
    for oy in range(span):
        for ox in range(span):
            gx = x0 + ox + 0.5; gy = y0 + oy + 0.5
            w0 = ((y[:, 1] - y[:, 2]) * (gx - x[:, 2]) + (x[:, 2] - x[:, 1]) * (gy - y[:, 2])) / den
            w1 = ((y[:, 2] - y[:, 0]) * (gx - x[:, 2]) + (x[:, 0] - x[:, 2]) * (gy - y[:, 2])) / den
            w2 = 1 - w0 - w1
            m = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            px = (x0 + ox)[m]; py = (y0 + oy)[m]
            inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            idx = np.nonzero(m)[0][inb]
            px, py = px[inb], py[inb]
            ww = np.stack([w0[m][inb], w1[m][inb], w2[m][inb]], 1)
            pix_all.append(py * W + px)
            dep_all.append((ww * z[idx]).sum(1))
            nn = (ww[:, :, None] * N[F[idx]]).sum(1)
            nrm_all.append(nn)
    pix = np.concatenate(pix_all); dp = np.concatenate(dep_all); nr = np.concatenate(nrm_all)
    o = np.lexsort((dp, pix))
    pix, dp, nr = pix[o], dp[o], nr[o]
    u, first = np.unique(pix, return_index=True)
    nr = nr[first]
    nr /= np.linalg.norm(nr, axis=1, keepdims=True) + 1e-12
    L1 = np.array([-0.35, -0.75, 0.55]); L1 /= np.linalg.norm(L1)
    L2 = np.array([0.6, -0.3, 0.4]); L2 /= np.linalg.norm(L2)
    vdir = np.array([0, -1.0, 0])
    dif = 0.75 * np.clip(nr @ L1, 0, 1) + 0.3 * np.clip(nr @ L2, 0, 1)
    hv = L1 + vdir; hv /= np.linalg.norm(hv)
    spec = 0.35 * np.clip(nr @ hv, 0, 1) ** 40
    col = np.clip((0.18 + dif)[:, None] * np.array(renk) + spec[:, None], 0, 1)
    img = np.ones((H * W, 3)) * np.array([0.96, 0.96, 0.97])
    img[u] = col
    return img.reshape(H, W, 3)




def pafta(stl, cikis, tur="genel"):
    import trimesh
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = trimesh.load(stl)
    v, f = trimesh.remesh.subdivide_to_size(t.vertices, t.faces, max_edge=(0.6 if tur == "genel" else 0.22), max_iter=14)
    t = trimesh.Trimesh(v, f, process=False)
    W, H = 1500, 860
    fig, axs = plt.subplots(2, 2, figsize=(20, 11.5))
    if tur == "genel":
        views = [(14, 35, "Ön-yan"), (14, -145, "Arka-yan"), (0, 0, "Yan"), (28, 140, "Arka-üst")]
        for ax, (el, az, ttl) in zip(axs.flat, views):
            ax.imshow(render(t, el, az, W, H, renk=(0.75, 0.77, 0.80))); ax.set_title(ttl, fontsize=16); ax.axis("off")
    else:
        c = t.bounds.mean(0); L = t.extents[0]
        views = [(10, 60, "Ön detay", 2.0, c + [-0.30 * L, 0, 0]), (10, -120, "Arka detay", 2.0, c + [0.30 * L, 0, 0]),
                 (5, 0, "Jant ve lastik", 4.5, np.array([t.bounds[0, 0] + 0.2 * L, 0, 0.23 * t.extents[2]])),
                 (12, 15, "Ayna ve camlar", 2.6, np.array([t.bounds[0, 0] + 0.43 * L, 0, 0.8 * t.extents[2]]))]
        for ax, (el, az, ttl, z, m) in zip(axs.flat, views):
            ax.imshow(render(t, el, az, W, H, renk=(0.75, 0.77, 0.80), zoom=z, merkez=m)); ax.set_title(ttl, fontsize=16); ax.axis("off")
    plt.tight_layout(); plt.savefig(cikis, dpi=75)


if __name__ == "__main__":
    kom = sys.argv[1]
    if kom == "zebra":
        yol, cikis = sys.argv[2], sys.argv[3]
        cx, cy, cz, az, el, gen = map(float, sys.argv[4:10])
        olc = float(sys.argv[10]) if len(sys.argv) > 10 else 24.0
        V, F = yukle(yol, olc)
        a = ciz(V, F, (cx, cy, cz), az, el, gen); b = ciz(V, F, (cx, cy, cz), az, el, gen, zebra=True)
        im = Image.new("RGB", (2000, 700), "white"); im.paste(a, (0, 0)); im.paste(b, (1000, 0)); im.save(cikis)
    elif kom == "pafta":
        pafta(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "genel")
