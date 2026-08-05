"""Pins the depth-sort invariant that makes a knot look knotted.

mplot3d is a painter's-algorithm renderer with no z-buffer: it asks each artist
for ONE depth and draws whole artists back to front. So a Poly3DCollection is
atomic, and two linked tubes drawn as two collections put one wholly in front of
the other at BOTH crossings -- the link reads as two adjacent rings. The fix, and
the reason viz.add_parts exists, is to merge every triangle into a single
collection and let matplotlib sort face by face.

This was the recurring rendering failure in the upstream repo, so it is pinned
here rather than left as a comment. The measurement is camera-agnostic: render
each tube ALONE to get its pixel mask, intersect the masks to find the pixels
where the two tubes project onto one another, and ask who wins there.

    two collections -> one tube wins essentially every contested pixel
    one collection  -> both win large shares, because each is in front somewhere

A regression that reverts add_parts to one-collection-per-part turns the second
number into the first, and this test fails.
"""
from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("skimage")

from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from soliton_playground.viz import (add_parts, bbox_of, dark_3d, fit_axes,
                                    iso_parts, shade_faces, write_gif)

N, L, R, r = 96, 8.0, 2.0, 0.62
DX = L / N
ELEV, AZIM = 24.0, -58.0
RED = (0.90, 0.16, 0.16, 1.0)
BLUE = (0.16, 0.42, 0.92, 1.0)


def _grid():
    g = np.linspace(-L / 2, L / 2, N, endpoint=False)
    return np.meshgrid(g, g, g, indexing="ij")


def _hopf_link():
    """Two interlocked tori. A's core circle is x^2+y^2=R^2 at z=0; B's is
    (x-R)^2+z^2=R^2 at y=0, so B threads A's hole at the origin and escapes at
    x=2R. Genuinely linked, not merely adjacent -- which is what makes "who is in
    front" differ between the two crossings.
    """
    X, Y, Z = _grid()
    sa = np.sqrt((np.sqrt(X ** 2 + Y ** 2) - R) ** 2 + Z ** 2) - r
    sb = np.sqrt((np.sqrt((X - R) ** 2 + Z ** 2) - R) ** 2 + Y ** 2) - r
    a = iso_parts(sa, 0.0, DX, RED)
    b = iso_parts(sb, 0.0, DX, BLUE)
    assert a is not None and b is not None, "SDF isosurface came back empty"
    return a, b


def _render(collections, center, half):
    """collections = list of parts; ONE Poly3DCollection per entry. Passing a
    single-entry list of concatenated faces is the merged case.
    """
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    for faces, cols in collections:
        pc = Poly3DCollection(list(faces), facecolors=cols, linewidths=0,
                              shade=False)
        pc.set_zsort("average")
        ax.add_collection3d(pc)
    fit_axes(ax, center, half)
    ax.view_init(elev=ELEV, azim=AZIM)
    dark_3d(ax)
    fig.patch.set_facecolor("black")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def _is_red(img):
    return img[..., 0].astype(int) - img[..., 2].astype(int) > 20


def _is_blue(img):
    return img[..., 2].astype(int) - img[..., 0].astype(int) > 20


@pytest.fixture(scope="module")
def link():
    a, b = _hopf_link()
    center, half = bbox_of([a, b])
    only_a = _render([a], center, half)
    only_b = _render([b], center, half)
    contested = _is_red(only_a) & _is_blue(only_b)
    assert contested.sum() > 500, "tubes barely overlap; the probe proves nothing"
    return a, b, center, half, contested


def _minority_share(img, contested):
    """Fraction of contested pixels won by whichever tube wins fewer of them.
    0 means one tube occludes the other everywhere; 0.5 is a perfect split.
    """
    n = int(contested.sum())
    red = int(_is_red(img)[contested].sum())
    blue = int(_is_blue(img)[contested].sum())
    return min(red, blue) / n


def test_separate_collections_do_not_weave(link):
    """The bug, pinned: one collection per tube means uniform occlusion."""
    a, b, center, half, contested = link
    share = _minority_share(_render([a, b], center, half), contested)
    assert share < 0.02, (
        f"expected uniform occlusion from separate collections, got {share:.3f}; "
        "if mplot3d gained a real z-buffer this test is obsolete, not wrong")


def test_merged_collection_weaves(link):
    """The fix: merged faces depth-sort against each other, so each tube is in
    front somewhere and the link actually crosses over and under.
    """
    a, b, center, half, contested = link
    merged = (np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]]))
    share = _minority_share(_render([merged], center, half), contested)
    assert share > 0.20, (
        f"merged collection did not weave: minority share {share:.3f}. The knot "
        "will render as separate rings.")


def test_add_parts_merges_into_one_collection(link):
    """add_parts must produce exactly ONE collection however many parts it gets.
    This is the invariant; the two tests above are what it buys.
    """
    a, b, center, half, _ = link
    fig = plt.figure(figsize=(3, 3), dpi=60)
    ax = fig.add_subplot(111, projection="3d")
    nf = add_parts(ax, [a, b, None])
    assert len(ax.collections) == 1, (
        f"add_parts made {len(ax.collections)} collections; must be 1 or the "
        "link stops weaving")
    assert nf == len(a[0]) + len(b[0])
    plt.close(fig)


def test_insertion_order_is_not_a_workaround(link):
    """Reordering cannot fix it: mplot3d sorts collections by their own average
    depth, so both orders give the same picture. Recorded because it is the first
    thing one tries.
    """
    a, b, center, half, _ = link
    assert np.array_equal(_render([a, b], center, half),
                          _render([b, a], center, half))


def test_shade_faces_is_two_sided(link):
    """Marching-cubes winding is not reliably outward, so shading must use |n.l|.
    A one-sided version leaves black patches where normals flip.
    """
    a, _, _, _, _ = link
    faces = a[0]
    cols = shade_faces(faces, RED)
    flipped = shade_faces(faces[:, ::-1, :], RED)
    assert np.allclose(cols, flipped), "shading changed when winding reversed"
    assert cols[:, :3].max() > 0.05, "every face came out black"


def test_write_gif_shares_one_palette(tmp_path):
    """Frames are quantised against a single shared palette. Per-frame ADAPTIVE
    palettes make static regions shift hue between frames.
    """
    from PIL import Image

    frames = []
    for k in range(6):
        img = np.zeros((32, 32, 3), np.uint8)
        img[:, :, 0] = 40 * k
        img[8:24, 8:24, 2] = 200
        frames.append(img)
    out = write_gif(frames, tmp_path / "a.gif", fps=10)
    assert out.exists() and out.stat().st_size > 0

    with Image.open(out) as im:
        assert getattr(im, "n_frames", 1) == 6
        first = im.convert("RGB").getpixel((16, 16))
        im.seek(5)
        last = im.convert("RGB").getpixel((16, 16))
    assert first[2] > 100 and last[2] > 100, "the blue square did not survive"


def test_write_gif_rejects_empty():
    with pytest.raises(ValueError):
        write_gif([], "unused.gif")
