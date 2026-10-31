import itertools
import torch

_CHUNK = 1 << 21


# ------------------------- Utility -------------------------
def _expand(counts):
    dev = counts.device
    total = int(counts.sum().item())
    owner = torch.repeat_interleave(torch.arange(counts.numel(), device=dev), counts)
    starts = torch.cumsum(counts, 0) - counts
    local = torch.arange(total, device=dev) - starts[owner]
    return owner, local


def _chunks(counts, limit=_CHUNK):
    cum = torch.cumsum(counts, 0).cpu()
    n, s = counts.numel(), 0
    while s < n:
        base = int(cum[s - 1]) if s > 0 else 0
        e = int(torch.searchsorted(cum, torch.tensor([base + limit]), right=True))
        e = min(max(e, s + 1), n)
        yield s, e
        s = e


def _encode(c, gmin, dims):
    return ((c[:, 0] - gmin[0]) * dims[1] + (c[:, 1] - gmin[1])) * dims[2] + (c[:, 2] - gmin[2])


def _plane_q(n, d):
    q = torch.cat([n, d[:, None]], 1)
    return (q[:, :, None] * q[:, None, :]).reshape(-1, 16)


def _psolve(M, rhs, xref):
    """Resolves M x = rhs (simmetric M, singular too) remaining close to xref.
    Minimum Ridge instead of pinv: same effect on null directions, but without eigh."""
    r = rhs - torch.einsum('...ij,...j->...i', M, xref)
    diag = torch.diagonal(M, dim1=-2, dim2=-1)
    lam = (diag.abs().amax(-1) * 1e-10).clamp(min=1e-14)
    Mr = M + lam[..., None, None] * torch.eye(3, device=M.device, dtype=M.dtype)
    delta = torch.linalg.solve(Mr, r.unsqueeze(-1)).squeeze(-1)
    delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
    return xref + delta


# ------------------------- 1) Angles Intersections -------------------------
def _edge_intersections(tri, gmin, gmax, dims):
    tri = tri.double() # precision
    keys, axes, tids, zs = [], [], [], []
    mn, mx = tri.min(1).values, tri.max(1).values
    for a in range(3):
        u, v = (a + 1) % 3, (a + 2) % 3
        lo_u = torch.ceil(mn[:, u]).long().clamp(gmin[u] + 1, gmax[u])
        lo_v = torch.ceil(mn[:, v]).long().clamp(gmin[v] + 1, gmax[v])
        hi_u = torch.floor(mx[:, u]).long().clamp(gmin[u] + 1, gmax[u])
        hi_v = torch.floor(mx[:, v]).long().clamp(gmin[v] + 1, gmax[v])
        nu = (hi_u - lo_u + 1).clamp(min=0)
        nv = (hi_v - lo_v + 1).clamp(min=0)
        cnt = nu * nv
        idx = torch.nonzero(cnt).squeeze(1)
        if idx.numel() == 0:
            continue
        cnt_i = cnt[idx]
        for s, e in _chunks(cnt_i):
            owner, local = _expand(cnt_i[s:e])
            t = idx[s:e][owner]
            nu_t = nu[t]
            iu = lo_u[t] + local % nu_t
            iv = lo_v[t] + local // nu_t
            P = tri[t]
            x, y, z = P[:, :, u], P[:, :, v], P[:, :, a]
            px, py = iu.double(), iv.double()
            det = (y[:, 1] - y[:, 2]) * (x[:, 0] - x[:, 2]) + (x[:, 2] - x[:, 1]) * (y[:, 0] - y[:, 2])
            ok = det.abs() > 1e-12
            det = torch.where(ok, det, torch.ones_like(det))
            l0 = ((y[:, 1] - y[:, 2]) * (px - x[:, 2]) + (x[:, 2] - x[:, 1]) * (py - y[:, 2])) / det
            l1 = ((y[:, 2] - y[:, 0]) * (px - x[:, 2]) + (x[:, 0] - x[:, 2]) * (py - y[:, 2])) / det
            l2 = 1.0 - l0 - l1
            eps = 1e-9
            zz = l0 * z[:, 0] + l1 * z[:, 1] + l2 * z[:, 2]
            k = torch.floor(zz).long()
            m = ok & (l0 >= -eps) & (l1 >= -eps) & (l2 >= -eps) & (k >= gmin[a]) & (k < gmax[a])
            if not m.any():
                continue
            c = torch.empty((int(m.sum()), 3), dtype=torch.long, device=tri.device)
            c[:, u], c[:, v], c[:, a] = iu[m] - 1, iv[m] - 1, k[m]
            keys.append(_encode(c, gmin, dims))
            axes.append(torch.full((c.shape[0],), a, dtype=torch.long, device=tri.device))
            tids.append(t[m])
            zs.append((zz - k.double())[m].float()) # local fraction in [0,1)
    if not keys:
        return None
    return torch.cat(keys), torch.cat(axes), torch.cat(tids), torch.cat(zs)


# ------------------------- 2) face QEF (SAT) -------------------------
def _tri_box_overlap(P, n):
    """P: (T,3,3) vertices relative to the center of the box (half side 0.5); n: normals (T,3)."""
    h, eps = 0.5, 1e-5
    ok = torch.ones(P.shape[0], dtype=torch.bool, device=P.device)
    ok &= ~((P.min(1).values > h + eps).any(-1) | (P.max(1).values < -h - eps).any(-1))
    ok &= (n * P[:, 0]).sum(-1).abs() <= h * n.abs().sum(-1) + eps
    E = torch.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 1], P[:, 0] - P[:, 2]], 1)
    z = torch.zeros_like(E[..., 0])
    axes = torch.cat([
        torch.stack([z, E[..., 2], -E[..., 1]], -1),
        torch.stack([-E[..., 2], z, E[..., 0]], -1),
        torch.stack([E[..., 1], -E[..., 0], z], -1)], 1)          # (T,9,3)
    p = torch.einsum('tak,tvk->tav', axes, P)
    r = h * axes.abs().sum(-1)
    ok &= ((p.min(-1).values <= r + eps) & (p.max(-1).values >= -r - eps)).all(-1)
    return ok


# ------------------------- 3) boundary QEF (DDA) -------------------------
def _segment_voxels(p0, p1):
    S, dev = p0.shape[0], p0.device
    d = p1 - p0
    segs = [torch.arange(S, device=dev)] * 2
    ts = [torch.zeros(S, device=dev), torch.ones(S, device=dev)]
    for a in range(3):
        k0 = torch.ceil(torch.minimum(p0[:, a], p1[:, a])).long()
        k1 = torch.floor(torch.maximum(p0[:, a], p1[:, a])).long()
        n = (k1 - k0 + 1).clamp(min=0)
        n = torch.where(d[:, a].abs() > 1e-9, n, torch.zeros_like(n))
        if int(n.sum()) == 0:
            continue
        owner, local = _expand(n)
        k = k0[owner] + local
        segs.append(owner)
        ts.append(((k.float() - p0[owner, a]) / d[owner, a]).clamp(0, 1))
    seg, t = torch.cat(segs), torch.cat(ts)
    o = torch.argsort(t); seg, t = seg[o], t[o]
    o = torch.argsort(seg, stable=True); seg, t = seg[o], t[o]
    m = (seg[1:] == seg[:-1]) & ((t[1:] - t[:-1]) > 1e-6)
    mid = 0.5 * (t[1:] + t[:-1])[m]
    sg = seg[1:][m]
    pt = p0[sg] + mid[:, None] * d[sg]
    return torch.floor(pt).long(), sg, pt


def _accumulate_boundary(V, F, uniq, cabs, qef, w, gmin, dims):
    N = uniq.numel()
    e = torch.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1).reshape(-1, 2)
    e = e.sort(1).values
    Vn = V.shape[0]
    uk, cn = torch.unique(e[:, 0] * Vn + e[:, 1], return_counts=True)
    bk = uk[cn == 1]
    if bk.numel() == 0:
        return
    p0, p1 = V[bk // Vn], V[bk % Vn]
    vox, sg, pt = _segment_voxels(p0, p1)
    key = _encode(vox, gmin, dims)
    ok = ((vox >= torch.tensor(gmin, device=V.device)).all(1) &
          (vox < torch.tensor([gmin[i] + dims[i] for i in range(3)], device=V.device)).all(1))
    vox, sg, pt, key = vox[ok], sg[ok], pt[ok], key[ok]
    pos = torch.searchsorted(uniq, key).clamp_(max=N - 1)
    m = uniq[pos] == key
    sg, pt, pos = sg[m], pt[m], pos[m]
    if pos.numel() == 0:
        return
    dvec = p1[sg] - p0[sg]
    dvec = dvec / dvec.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    q = pt - cabs[pos]
    A = torch.eye(3, device=V.device)[None] - dvec[:, :, None] * dvec[:, None, :]
    b = -torch.einsum('kij,kj->ki', A, q)
    c = (q * q).sum(-1) - (q * dvec).sum(-1) ** 2
    Q = torch.zeros(pos.numel(), 4, 4, device=V.device)
    Q[:, :3, :3], Q[:, :3, 3], Q[:, 3, :3], Q[:, 3, 3] = A, b, b, c
    qef.index_add_(0, pos, w * Q.reshape(-1, 16))


# ------------------------- 4) solve QEF -------------------------
def _patterns(dev):
    st = torch.tensor([s for s in itertools.product(range(3), repeat=3) if s != (0, 0, 0)],
                      device=dev) # 0 free, 1 = min, 2 = max
    return (st > 0).float(), (st == 2).float()


def _solve_vertices(Q, pref, chunk=50_000):
    Q, pref = Q.double(), pref.double()
    N, dev = Q.shape[0], Q.device
    out = torch.empty(N, 3, device=dev) # remains float32
    fixed, val = [p.double() for p in _patterns(dev)]
    eye = torch.eye(3, device=dev, dtype=torch.float64)
    tol = 1e-6
    for s in range(0, N, chunk):
        q, pr = Q[s:s + chunk], pref[s:s + chunk]
        A, b, c = q[:, :3, :3], q[:, :3, 3], q[:, 3, 3]
        x = _psolve(A, -b, pr) # free case
        inside = ((x >= -tol) & (x <= 1 + tol)).all(1)
        res = x.clamp(0, 1)
        o = torch.nonzero(~inside).squeeze(1)
        if o.numel():
            Ao, bo, co, po = A[o], b[o], c[o], pr[o]
            fm = (1 - fixed)[:, :, None] * (1 - fixed)[:, None, :]       # (P,3,3)
            M = Ao[:, None] * fm[None] + (fixed[:, :, None] * eye)[None] # (K,P,3,3)
            g = bo[:, None, :] + torch.einsum('kij,pj->kpi', Ao, val)
            rhs = (1 - fixed)[None] * (-g) + (fixed * val)[None]
            xref = (fixed * val)[None] + (1 - fixed)[None] * po[:, None, :]
            xs = _psolve(M, rhs, xref)                                   # (K,P,3)
            valid = ((xs >= -tol) & (xs <= 1 + tol)).all(-1)
            xs = xs.clamp(0, 1)
            err = (torch.einsum('kpi,kij,kpj->kp', xs, Ao, xs)
                   + 2 * torch.einsum('kpi,ki->kp', xs, bo) + co[:, None])
            err = torch.where(valid, err, torch.full_like(err, float('inf')))
            best = err.argmin(1)
            res[o] = xs[torch.arange(o.numel(), device=dev), best]
        out[s:s + chunk] = res
    return out


def _overlap_pairs(tri, nrm, gmin, gmax, dims, limit=1 << 19):
    dev = tri.device
    gmin_t = torch.tensor(gmin, device=dev)
    gmax_t = torch.tensor(gmax, device=dev)
    lo = torch.maximum(torch.floor(tri.min(1).values).long(), gmin_t)
    hi = torch.minimum(torch.floor(tri.max(1).values).long(), gmax_t - 1)
    nd = (hi - lo + 1).clamp(min=0)
    cnt = nd.prod(1)
    idx = torch.nonzero(cnt).squeeze(1)
    keys, tids = [], []
    if idx.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=dev), torch.zeros(0, dtype=torch.long, device=dev)
    cnt_i = cnt[idx]
    for s, e in _chunks(cnt_i, limit):
        owner, local = _expand(cnt_i[s:e])
        t = idx[s:e][owner]
        ndt = nd[t]
        lx = local % ndt[:, 0]
        ly = (local // ndt[:, 0]) % ndt[:, 1]
        lz = local // (ndt[:, 0] * ndt[:, 1])
        c = lo[t] + torch.stack([lx, ly, lz], 1)
        P = (tri[t].double() - (c.double() + 0.5)[:, None, :]).float() # subtraction in fp64
        m = _tri_box_overlap(P, nrm[t])
        keys.append(_encode(c[m], gmin, dims))
        tids.append(t[m])
    return torch.cat(keys), torch.cat(tids)


# Construct `uniq` from only the 4 neighbors of each intersection, as in C++, and filter `okeys` based on that set
def _neighbor_keys(ekeys, axes, dims, dev):
    dz, dy = dims[2], dims[1]
    c0 = torch.stack([ekeys // (dy * dz), (ekeys // dz) % dy, ekeys % dz], 1)
    ar = torch.arange(axes.numel(), device=dev)
    u, v = (axes + 1) % 3, (axes + 2) % 3
    dims_t = torch.tensor(dims, device=dev)
    out = []
    for du in (0, 1):
        for dv in (0, 1):
            c = c0.clone()
            c[ar, u] += du
            c[ar, v] += dv
            ok = ((c >= 0) & (c < dims_t)).all(1)
            out.append((c[ok, 0] * dy + c[ok, 1]) * dz + c[ok, 2])
    return torch.cat(out)


# ------------------------- Main Function -------------------------
@torch.no_grad()
def mesh_to_flexible_dual_grid(
    vertices, faces, 
    voxel_size, 
    grid_range,
    face_weight=1.0,
    boundary_weight=1.0,
    regularization_weight=0.1
):
    dev = vertices.device
    vs = voxel_size.to(dev, torch.float32).reshape(3)
    gr = grid_range.reshape(-1).tolist()
    gmin, gmax = [int(x) for x in gr[:3]], [int(x) for x in gr[3:]]
    dims = [gmax[i] - gmin[i] for i in range(3)]

    V = vertices.to(dev, torch.float32) / vs
    F = faces.to(dev).long()
    tri = V[F]
    n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    a2 = n.norm(dim=-1)
    good = a2 > 1e-12
    tri, nrm = tri[good], n[good] / a2[good, None]

    def empty():
        return (torch.zeros(0, 3, dtype=torch.int32, device=dev),
                torch.zeros(0, 3, dtype=torch.float32, device=dev),
                torch.zeros(0, 3, dtype=torch.bool, device=dev))

    r = _edge_intersections(tri, gmin, gmax, dims)
    if r is None:
        return empty()
    ekeys, axes, tids, zs = r

    # voxel = union of (intersected angles) and (voxels intersected by the surface)
    okeys, otids = _overlap_pairs(tri, nrm, gmin, gmax, dims)
    uniq = torch.unique(_neighbor_keys(ekeys, axes, dims, dev)) # only the close 4
    N = uniq.numel()

    # face QEF: only already present voxels (like C++ hash table)
    pos = torch.searchsorted(uniq, okeys).clamp_(max=N - 1)
    m = uniq[pos] == okeys
    okeys, otids = okeys[m], otids[m]

    inv = torch.searchsorted(uniq, ekeys)

    dz, dy = dims[2], dims[1]
    cz = uniq % dz
    cy = (uniq // dz) % dy
    cx = uniq // (dz * dy)
    coords = torch.stack([cx, cy, cz], 1) + torch.tensor(gmin, device=dev)
    cabs = coords.float()

    intersected = torch.zeros(N, 3, dtype=torch.bool, device=dev)
    inv = torch.searchsorted(uniq, ekeys)
    intersected[inv, axes] = True # flag only on "max angle" voxel

    qef = torch.zeros(N, 16, device=dev)
    mean = torch.zeros(N, 3, device=dev)
    cnt = torch.zeros(N, device=dev)
    dims_t = torch.tensor(dims, device=dev)

    for s in range(0, ekeys.numel(), _CHUNK):
        sl = slice(s, s + _CHUNK)
        k0, a, t, z = ekeys[sl], axes[sl], tids[sl], zs[sl]
        c0 = torch.stack([k0 // (dy * dz), (k0 // dz) % dy, k0 % dz], 1) # relative to gmin
        nn = nrm[t]
        ar = torch.arange(a.numel(), device=dev)
        u, v = (a + 1) % 3, (a + 2) % 3
        for du in (0, 1):
            for dv in (0, 1):
                c = c0.clone()
                c[ar, u] += du
                c[ar, v] += dv
                valid = ((c >= 0) & (c < dims_t)).all(1)
                key = (c[:, 0] * dy + c[:, 1]) * dz + c[:, 2]
                pos = torch.searchsorted(uniq, key).clamp_(max=N - 1)
                valid &= uniq[pos] == key
                pl = torch.empty(a.numel(), 3, device=dev)
                pl[ar, u] = 1.0 - du # local point coordinate
                pl[ar, v] = 1.0 - dv
                pl[ar, a] = z
                Qf = _plane_q(nn, -(nn * pl).sum(-1))
                p_ = pos[valid]
                qef.index_add_(0, p_, Qf[valid])
                mean.index_add_(0, p_, pl[valid])
                cnt.index_add_(0, p_, torch.ones(p_.numel(), device=dev))

    pref = torch.where(cnt[:, None] > 0, mean / cnt.clamp(min=1)[:, None],
                       torch.full_like(mean, 0.5))

    if face_weight > 0:
        for s in range(0, okeys.numel(), _CHUNK):
            k, t = okeys[s:s + _CHUNK], otids[s:s + _CHUNK]
            pos = torch.searchsorted(uniq, k)
            nn = nrm[t]
            d = -(nn * (tri[t, 0].double() - cabs[pos].double()).float()).sum(-1)
            qef.index_add_(0, pos, face_weight * _plane_q(nn, d))
    if boundary_weight > 0:
        _accumulate_boundary(V, F, uniq, cabs, qef, boundary_weight, gmin, dims)

    Q = qef.view(N, 4, 4)
    if regularization_weight > 0:
        Qr = torch.zeros(N, 4, 4, device=dev)
        Qr[:, :3, :3] = torch.eye(3, device=dev)
        Qr[:, :3, 3] = -pref
        Qr[:, 3, :3] = -pref
        Qr[:, 3, 3] = (pref * pref).sum(-1)
        Q = Q + regularization_weight * cnt[:, None, None] * Qr # cnt=0 -> no regularization

    mesh_to_flexible_dual_grid._LAST_Q = Q.detach().clone() # DEBUG

    u = _solve_vertices(Q, pref)
    dual = (cabs + u) * vs
    return coords.int(), dual.float(), intersected
