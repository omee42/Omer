"""kamera — fotoğraf kalibrasyonu ve bindirme. Koordinatlar gerçek mm (x ön->arka, +y sağ, z yukarı).

Fotoğraftaki bilinen noktalar (teker merkezleri, jant çemberi, kapı köşesi, ayna, far ucu...) ile 3B karşılıkları
eşleştirilip iğne deliği kamera oturtulur (fit / fit_sev). Sonra model fotoğrafın üstüne çizilir (bindir): kırmızı
siluet, sarı keskin kırımlar, camgöbeği iç derinlik sıçramaları. Farkları ızgaralı kırpmalarda (izgara) piksel piksel
okuyup geri3d ile modelin 3B noktasına çevirerek ölçüm tablolarını düzeltirsin.

par = [Cx, Cy, Cz, rotvec(3), f, du, dv, (k1)] ; kam json: {"par": [...], "W": .., "H": .., "foto": "..."}
"""
import sys, json, numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
from PIL import Image, ImageDraw, ImageFont

R0 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)


def R_of(par):
    return Rot.from_rotvec(par[3:6]).as_matrix() @ R0


def proj(par, P, W, H):
    """par: C(3), rotvec(3), f, [du, dv] (asal nokta kayması)."""
    par = np.asarray(par, float)
    C, f = par[:3], par[6]
    du, dv = (par[7], par[8]) if len(par) > 8 else (0.0, 0.0)
    Q = (np.asarray(P, float).reshape(-1, 3) - C) @ R_of(par).T
    a, b = Q[:, 0] / Q[:, 2], Q[:, 1] / Q[:, 2]
    if len(par) > 9 and par[9] != 0.0:     # radyal bozulma k1 (normalize yarıçap: f'e göre)
        k = 1.0 + par[9] * (a * a + b * b)
        a, b = a * k, b * k
    return np.stack([f * a + W / 2 + du, f * b + H / 2 + dv], 1), Q[:, 2]


def bakis(C, T, f, roll=0.0, du=0.0, dv=0.0):
    """C kamerasından T'ye bakan kamera parametreleri."""
    C = np.asarray(C, float); T = np.asarray(T, float)
    zc = T - C; zc /= np.linalg.norm(zc)
    xc = np.cross(zc, [0, 0, 1.0]); xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.stack([xc, yc, zc])
    if roll:
        cr, sr = np.cos(roll), np.sin(roll)
        R = np.stack([cr * xc + sr * yc, -sr * xc + cr * yc, zc])
    rv = Rot.from_matrix(R @ R0.T).as_rotvec()
    return np.r_[C, rv, f, du, dv]


def daire(xc, yc, zc, r, n=72):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([xc + r * np.cos(th), np.full_like(th, yc), zc + r * np.sin(th)], 1)


def fit(pts, W, H, x0, circ=(), serbest_pp=False, loss='soft_l1', fs=4.0):
    """pts: [(X,Y,Z,u,v), ...]; circ: [(xc,yc,zc,r, [(u,v),...]), ...] noktalar 3B çemberin izdüşümüne en yakın."""
    a = np.asarray(pts, float).reshape(-1, 5)
    x0 = np.asarray(x0, float)
    if not serbest_pp:
        x0 = x0[:7]

    def res(p):
        q = np.r_[p, 0, 0] if len(p) == 7 else p
        r = list(np.nan_to_num((proj(q, a[:, :3], W, H)[0] - a[:, 3:5])).ravel()) if len(a) else []
        for xc, yc, zc, rr, uvs in circ:
            cu, _ = proj(q, daire(xc, yc, zc, rr, 360), W, H)
            for u, v in uvs:
                dd = np.hypot(cu[:, 0] - u, cu[:, 1] - v)
                i = np.argmin(dd)
                r += [cu[i, 0] - u, cu[i, 1] - v]
        return np.array(r)
    s = least_squares(res, x0, loss=loss, f_scale=fs, x_scale='jac', max_nfev=4000)
    p = s.x if len(s.x) == 9 else np.r_[s.x, 0, 0]
    e = res(s.x).reshape(-1, 2)
    return p, np.hypot(e[:, 0], e[:, 1])


def rasterize(V, F, par, W, H):
    uv, z = proj(par, V, W, H)
    x, y = uv[:, 0][F], uv[:, 1][F]
    dep = z[F]
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    C = np.asarray(par[:3])
    sg = np.sign(((V[F].mean(1) - C) * n).sum(1))
    n = -n * sg[:, None]
    keep = (x.max(1) >= 0) & (x.min(1) < W) & (y.max(1) >= 0) & (y.min(1) < H) & (dep.min(1) > 10)
    x, y, dep, n, fid = x[keep], y[keep], dep[keep], n[keep], np.nonzero(keep)[0]
    x0 = np.floor(x.min(1)).astype(int); y0 = np.floor(y.min(1)).astype(int)
    sp = np.ceil(np.maximum(x.max(1) - x0, y.max(1) - y0)).astype(int) + 1
    zb = np.full(W * H, np.inf); nb = np.zeros((W * H, 3)); fb = np.full(W * H, -1)
    den = (y[:, 1] - y[:, 2]) * (x[:, 0] - x[:, 2]) + (x[:, 2] - x[:, 1]) * (y[:, 0] - y[:, 2])
    ok = np.abs(den) > 1e-12
    lo = 0
    for smax_ in (2, 4, 8, 16, 32, 64, 128, 256, 512):
        sel = ok & (sp <= smax_) & (sp > lo)
        lo = smax_
        if not sel.any():
            continue
        xs, ys, ds, ns, xx0, yy0, dn, fi = x[sel], y[sel], dep[sel], n[sel], x0[sel], y0[sel], den[sel], fid[sel]
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
                zb[pix[upd]] = dd[upd]
                nb[pix[upd]] = ns[idx[upd]]
                fb[pix[upd]] = fi[idx[upd]]
    return zb.reshape(H, W), nb.reshape(H, W, 3), fb.reshape(H, W)


def kenarlar(zb, nb, sicrama=40.0, kirim=0.80):
    fin = np.isfinite(zb)
    sil = np.zeros_like(fin)
    sil[1:-1, 1:-1] = fin[1:-1, 1:-1] & ~(fin[:-2, 1:-1] & fin[2:, 1:-1] & fin[1:-1, :-2] & fin[1:-1, 2:])
    z = np.where(fin, zb, 1e9)
    jump = np.zeros_like(fin)
    for dy, dx in ((0, 1), (1, 0)):
        a = z[: z.shape[0] - dy, : z.shape[1] - dx]; b = z[dy:, dx:]
        j = np.abs(a - b) > sicrama * np.minimum(a, b) / 8000.0
        jump[: z.shape[0] - dy, : z.shape[1] - dx] |= j & (a < b)
        jump[dy:, dx:] |= j & (b < a)
    cr = np.zeros_like(fin)
    for dy, dx in ((0, 2), (2, 0)):
        a = nb[: nb.shape[0] - dy, : nb.shape[1] - dx]; b = nb[dy:, dx:]
        c = (a * b).sum(-1) < kirim
        cr[: nb.shape[0] - dy, : nb.shape[1] - dx] |= c & fin[: nb.shape[0] - dy, : nb.shape[1] - dx] & fin[dy:, dx:]
    return sil, jump & fin & ~sil, cr & ~jump


def mesh_yukle(yol, olcek=24.0):
    """STL (baskı mm, gerçek mm'ye çevrilir) ya da bölge öneki (<onek>_L.npz + _R.npz, gerçek mm)."""
    if yol.endswith('.stl'):
        import trimesh
        t = trimesh.load(yol)
        return np.asarray(t.vertices, float) * olcek, np.asarray(t.faces, np.int64)
    a = np.load(f'{yol}_L.npz'); b = np.load(f'{yol}_R.npz')
    V = np.vstack([a['v'], b['v']]).astype(float)
    F = np.vstack([a['f'], b['f'] + len(a['v'])])
    return V, F


def bindir(foto, V, F, par, renk=(255, 40, 40), crease=True, ic=True, sc=1.0):
    im = Image.open(foto).convert('RGB')
    W, H = im.size
    zb, nb, _ = rasterize(V, F, par, W, H)
    sil, jump, cr = kenarlar(zb, nb)
    a = np.asarray(im).copy()
    if crease:
        a[cr] = (255, 230, 0)
    if ic:
        a[jump] = (0, 255, 255)
    from scipy.ndimage import binary_dilation
    s2 = binary_dilation(sil, iterations=1 if W < 1500 else 2)
    a[s2] = renk
    return Image.fromarray(a), zb


def izgara(foto, kutu, adim=50, olcek=None, cikis=None, noktalar=()):
    """foto kırpması + piksel ızgarası (orijinal piksel etiketleriyle)."""
    im = Image.open(foto).convert('RGB') if isinstance(foto, str) else foto
    x0, y0, x1, y1 = kutu
    c = im.crop(kutu)
    if olcek is None:
        olcek = min(1400 / (x1 - x0), 1000 / (y1 - y0))
    c = c.resize((int((x1 - x0) * olcek), int((y1 - y0) * olcek)), Image.LANCZOS)
    d = ImageDraw.Draw(c)
    for gx in range((x0 // adim + 1) * adim, x1, adim):
        X = (gx - x0) * olcek
        d.line([(X, 0), (X, c.size[1])], fill=(0, 255, 0) if gx % (adim * 2) == 0 else (0, 160, 0), width=1)
        d.text((X + 2, 2), str(gx), fill=(255, 255, 0))
        d.text((X + 2, c.size[1] - 12), str(gx), fill=(255, 255, 0))
    for gy in range((y0 // adim + 1) * adim, y1, adim):
        Y = (gy - y0) * olcek
        d.line([(0, Y), (c.size[0], Y)], fill=(0, 255, 0) if gy % (adim * 2) == 0 else (0, 160, 0), width=1)
        d.text((2, Y + 2), str(gy), fill=(255, 255, 0))
        d.text((c.size[0] - 34, Y + 2), str(gy), fill=(255, 255, 0))
    for u, v, *lab in noktalar:
        X, Y = (u - x0) * olcek, (v - y0) * olcek
        d.ellipse([X - 4, Y - 4, X + 4, Y + 4], outline=(255, 0, 255), width=2)
        if lab:
            d.text((X + 6, Y - 6), str(lab[0]), fill=(255, 0, 255))
    if cikis:
        c.save(cikis)
    return c


def golge_render(V, F, par, W, H, cikis):
    zb, nb, fb = rasterize(V, F, par, W, H)
    L = np.array([0.3, -0.5, 0.8]); L /= np.linalg.norm(L)
    sh = np.clip(0.2 + 0.8 * np.abs(nb @ L), 0, 1)
    sil, jump, cr = kenarlar(zb, nb)
    a = np.where(np.isfinite(zb)[..., None], (sh[..., None] * np.array([200, 205, 215])), 255).astype(np.uint8)
    a[cr] = (200, 120, 0); a[jump] = (0, 120, 255); a[sil] = (255, 0, 0)
    Image.fromarray(a).save(cikis)
    np.savez_compressed(cikis + '.npz', zb=zb.astype(np.float32), par=np.asarray(par))
    return zb


def geri3d(zb, par, u, v, W, H, ara=3):
    """render derinliğinden pikselin 3B noktası (en yakın sonlu komşu)."""
    par = np.asarray(par, float)
    ui, vi = int(round(u)), int(round(v))
    blk = zb[vi - ara:vi + ara + 1, ui - ara:ui + ara + 1]
    z = np.nanmin(np.where(np.isfinite(blk), blk, np.nan))
    du, dv = (par[7], par[8]) if len(par) > 8 else (0, 0)
    q = np.array([(u - W / 2 - du) / par[6] * z, (v - H / 2 - dv) / par[6] * z, z])
    return par[:3] + R_of(par).T @ q


_VY = None


def yuzey_y(x, z, mesh, isaret=-1, r=10.0):
    """(x,z)'de gövdenin dış yüzeyi |y| (mesh tepe noktalarından)."""
    global _VY
    if _VY is None:
        _VY = mesh_yukle(mesh)[0]
    V = _VY
    m = (np.abs(V[:, 0] - x) < r) & (np.abs(V[:, 2] - z) < r) & (np.sign(V[:, 1]) == isaret)
    return isaret * np.abs(V[m, 1]).max()


def yuzey_x(y, z, mesh, on=True, r=10.0):
    global _VY
    if _VY is None:
        _VY = mesh_yukle(mesh)[0]
    V = _VY
    m = (np.abs(V[:, 1] - y) < r) & (np.abs(V[:, 2] - z) < r)
    return V[m, 0].min() if on else V[m, 0].max()


def dik_par(q):
    """q: Cx, Cy, Cz, yaw, f, dv, [pitch, roll] -> tam par (bakış yatay/eğik, düşey kenarlar düşey)."""
    pitch = q[6] if len(q) > 6 else 0.0
    roll = q[7] if len(q) > 7 else 0.0
    C = np.asarray(q[:3]); d = np.array([np.cos(q[3]) * np.cos(pitch), np.sin(q[3]) * np.cos(pitch), np.sin(pitch)])
    return bakis(C, C + d, q[4], roll=roll, dv=q[5])


def fit_dik(pts, W, H, circ=(), baslar=None, serbest=('pitch',), loss='linear', fs=4.0, yaw0=None):
    a = np.asarray(pts, float).reshape(-1, 5)
    nek = len(serbest)

    def res(q):
        p = dik_par(np.r_[q[:6], q[6:6 + nek] if 'pitch' in serbest else [0.0], q[7:8] if 'roll' in serbest else [0.0]])
        r = list(((proj(p, a[:, :3], W, H)[0] - a[:, 3:5])).ravel()) if len(a) else []
        for xc, yc, zc, rr, uvs in circ:
            cu, _ = proj(p, daire(xc, yc, zc, rr, 360), W, H)
            for u, v in uvs:
                dd = np.hypot(cu[:, 0] - u, cu[:, 1] - v); i = np.argmin(dd)
                r += [cu[i, 0] - u, cu[i, 1] - v]
        return np.array(r)
    best = None
    for q0 in baslar:
        q0 = np.r_[q0, [0.0] * nek]
        try:
            s = least_squares(res, q0, loss=loss, f_scale=fs, x_scale='jac', max_nfev=3000)
        except Exception:
            continue
        if best is None or s.cost < best.cost:
            best = s
    q = best.x
    p = dik_par(np.r_[q[:6], q[6:7] if 'pitch' in serbest else [0.0], q[7:8] if 'roll' in serbest else [0.0]])
    e = res(q).reshape(-1, 2)
    return p, np.hypot(e[:, 0], e[:, 1]), q


def baslangiclar(hedef=(2300, 0, 500), yonler=(-150, -90, -30), uzak=(5000, 9000, 14000), zs=(1000, 1600), fpx=None, W=2000):
    out = []
    for az in yonler:
        for d in uzak:
            for zc in zs:
                a = np.radians(az)
                C = np.array([hedef[0] + d * np.cos(a), hedef[1] + d * np.sin(a), zc])
                yaw = np.arctan2(hedef[1] - C[1], hedef[0] - C[0])
                f = (fpx or W * 0.9) * d / 6000
                out.append([*C, yaw, f, 0.0])
    return out


def kalan(p, pts, circ, W, H):
    a = np.asarray(pts, float).reshape(-1, 5)
    r = list(np.nan_to_num((proj(p, a[:, :3], W, H)[0] - a[:, 3:5])).ravel()) if len(a) else []
    for xc, yc, zc, rr, uvs in circ:
        cu, _ = proj(p, daire(xc, yc, zc, rr, 360), W, H)
        for u, v in uvs:
            dd = np.hypot(cu[:, 0] - u, cu[:, 1] - v); i = np.argmin(dd)
            r += [cu[i, 0] - u, cu[i, 1] - v]
    return np.array(r)


def fit_sev(pts, circ, W, H, baslar, serbest=(), sabit=None):
    """düzey kamera: q = Cx Cy Cz yaw f + serbest{'dv','pitch','roll','du'}; sabit: dict varsayılanlar."""
    sabit = {**dict(dv=0.0, pitch=0.0, roll=0.0, du=0.0, k1=0.0), **(sabit or {})}
    def par(q):
        d = dict(sabit); d.update(zip(serbest, q[5:]))
        p = dik_par([q[0], q[1], q[2], q[3], q[4], d['dv'], d['pitch'], d['roll']])
        p[7] = d['du']
        if 'k1' in d or 'k1' in serbest:
            p = np.r_[p, d.get('k1', 0.0)]
        return p
    best = None
    for q0 in baslar:
        q0 = np.r_[q0[:5], [0.0] * len(serbest)]
        try:
            s = least_squares(lambda q: kalan(par(q), pts, circ, W, H), q0, x_scale='jac', max_nfev=3000)
        except Exception:
            continue
        if best is None or s.cost < best.cost:
            best = s
    p = par(best.x)
    e = kalan(p, pts, circ, W, H).reshape(-1, 2)
    return p, np.hypot(e[:, 0], e[:, 1]), dict(zip(serbest, best.x[5:]))


if __name__ == "__main__":
    # python3 kamera.py bindir <kam.json> <model.stl|bolge_onek> <cikis.jpg> [x0 y0 x1 y1]
    import json as _j
    if sys.argv[1] == "bindir":
        c = _j.load(open(sys.argv[2])); V, F = mesh_yukle(sys.argv[3])
        im, _ = bindir(c["foto"], V, F, np.array(c["par"]))
        if len(sys.argv) > 8:
            im = im.crop(tuple(map(int, sys.argv[5:9])))
        im.save(sys.argv[4], quality=90)
