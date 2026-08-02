"""Probe: can we reconstruct real per-palm plantation geometry from ds_B filenames?

Tile filenames encode absolute ortho pixel offsets:
    {orthoX}_{orthoY}_{tileX}_{tileY}_jpg.rf.<hash>.jpg
so global crown position = (tileX + bbox_cx, tileY + bbox_cy) within its ortho.

This is a READ-ONLY probe. It prints geometry stats and writes nothing except a
crowns_{ortho}.npy point cloud for inspection.
"""
import os, re, sys, json, collections
import numpy as np
from grids import load_coco, BASE, OUT

FNAME = re.compile(r"^(\d+)_(\d+)_(\d+)_(\d+)_jpg\.rf\.")


def parse(fn):
    """-> (ortho_id, tile_x, tile_y) or None."""
    m = FNAME.match(os.path.basename(fn))
    if not m:
        return None
    ox, oy, tx, ty = m.groups()
    return f"{ox}_{oy}", int(tx), int(ty)


def crowns(tag="B"):
    """Global crown records per ortho: (x, y, w, h, label)."""
    cats, anns = load_coco(tag)
    by = collections.defaultdict(list)
    unparsed = 0
    for a in anns:
        p = parse(a["path"])
        if p is None:
            unparsed += 1
            continue
        ortho, tx, ty = p
        x, y, w, h = a["bbox"]
        by[ortho].append((tx + x + w / 2.0, ty + y + h / 2.0, w, h, a["cat"]))
    return by, unparsed, len(anns)


def dedup(pts, radius):
    """Greedy merge of detections closer than `radius` px (tile overlap duplicates).
    Returns kept-index mask. O(n^2) on a grid-bucketed neighbourhood."""
    n = len(pts)
    xy = pts[:, :2].astype(float)
    order = np.argsort(xy[:, 0])
    keep = np.ones(n, bool)
    for ii in range(n):
        i = order[ii]
        if not keep[i]:
            continue
        for jj in range(ii + 1, n):
            j = order[jj]
            if xy[j, 0] - xy[i, 0] > radius:
                break
            if keep[j] and (xy[j, 0] - xy[i, 0]) ** 2 + (xy[j, 1] - xy[i, 1]) ** 2 < radius ** 2:
                keep[j] = False
    return keep


def nn_dist(xy):
    """Nearest-neighbour distance per point (brute force, chunked)."""
    n = len(xy)
    out = np.empty(n)
    CH = 2000
    for s in range(0, n, CH):
        e = min(s + CH, n)
        d = np.sqrt(((xy[s:e, None, :] - xy[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d[:, s:e], np.inf)
        out[s:e] = d.min(1)
    return out


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "B"
    by, unparsed, total = crowns(tag)
    print(f"[ds_{tag}] annotations={total}  unparsed_filenames={unparsed}  orthos={len(by)}")

    for ortho in sorted(by):
        rec = by[ortho]
        arr = np.array([(x, y, w, h) for x, y, w, h, _ in rec], float)
        lab = np.array([c for *_, c in rec])
        med_box = float(np.median(np.sqrt(arr[:, 2] * arr[:, 3])))

        # dedup at 0.5 * median crown size (tile overlap -> same palm twice)
        keep = dedup(arr, radius=0.5 * med_box)
        A, L = arr[keep], lab[keep]
        xy = A[:, :2]

        nn = nn_dist(xy)
        med_nn = float(np.median(nn))
        gsd = 9.0 / med_nn                       # ASUMSI: real palm spacing = 9 m
        ext_x = (xy[:, 0].max() - xy[:, 0].min()) * gsd
        ext_y = (xy[:, 1].max() - xy[:, 1].min()) * gsd

        unh = int((L == "Unhealthy").sum())
        print(f"\n  ortho {ortho}")
        print(f"    raw={len(arr)}  after_dedup={len(A)}  removed={len(arr)-len(A)}"
              f" ({100*(len(arr)-len(A))/len(arr):.1f}%)")
        print(f"    median crown box  = {med_box:.1f} px")
        print(f"    median NN spacing = {med_nn:.1f} px  -> GSD ~ {gsd*100:.1f} cm/px (if 9 m planting)")
        print(f"    extent            = {ext_x:.0f} m x {ext_y:.0f} m  ({ext_x*ext_y/1e4:.1f} ha)")
        print(f"    labels            = Healthy {len(A)-unh}  Unhealthy {unh}"
              f"  ({100*unh/len(A):.2f}%)")

        # degree under the paper's root radius (13 m) on REAL geometry
        m_xy = xy * gsd
        deg = []
        CH = 2000
        for s in range(0, len(m_xy), CH):
            e = min(s + CH, len(m_xy))
            d = np.sqrt(((m_xy[s:e, None, :] - m_xy[None, :, :]) ** 2).sum(-1))
            deg.append(((d <= 13.0).sum(1) - 1))
        deg = np.concatenate(deg)
        print(f"    mean root degree @13 m = {deg.mean():.2f}  (synthetic Moore-8 = 7.70)")

        np.save(os.path.join(OUT, f"crowns_{tag}_{ortho}.npy"),
                dict(xy_px=xy, wh=A[:, 2:], label=L, gsd=gsd), allow_pickle=True)
    print(f"\nwrote crowns_*.npy to {OUT}")
