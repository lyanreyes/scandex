"""
ScanDex visual feature extractor.

Every function here is mirrored line-for-line in web/js/features.js so that the
fingerprints computed in the browser at scan time are identical to the ones
stored in the index built by 03_build_index.py. Keep both files in sync.

Pipeline for one card image (RGB uint8 array, any size):
  1. trim  - drop a fixed margin on every side (border / alignment noise)
  2. base  - area-resize to a 64 x 88 RGB "base" image (card aspect 63:88)
  3. gray  - luma (ITU-R BT.601)
  4. aHash - 8x8 mean threshold                     ->  64 bits
  5. dHash - 9x8 horizontal gradient sign           ->  64 bits
  6. pHash - 2D DCT-II of a 32x32 / 64x64 image,
             low-frequency 8x8 / 16x16 block vs. median -> 64 / 256 bits
  7. color - 4x6 RGB layout, zero-mean / unit-norm  ->  72 floats
  8. hist  - HSV colour histogram (8 hue x 2 sat x 3 val + 3 grey bins),
             square-rooted (Hellinger) so it is unit-norm   ->  51 floats
"""
import numpy as np

BASE_W, BASE_H = 64, 88
TRIM = 0.05          # fraction removed from each side before resizing
COLOR_W, COLOR_H = 4, 6


def area_resize(img, tw, th):
    """Box-filter resize: every source pixel is added to the target cell its
    centre falls into. Deterministic and trivial to reproduce in JavaScript."""
    img = np.asarray(img, dtype=np.float64)
    sh, sw = img.shape[:2]
    ys = (np.arange(sh) * th) // sh
    xs = (np.arange(sw) * tw) // sw
    chans = 1 if img.ndim == 2 else img.shape[2]
    flat = img.reshape(sh, sw, chans)
    out = np.zeros((th, tw, chans))
    cnt = np.zeros((th, tw, 1))
    np.add.at(out, (ys[:, None], xs[None, :]), flat)
    np.add.at(cnt, (ys[:, None], xs[None, :]), 1)
    out /= np.maximum(cnt, 1)
    return out if img.ndim == 3 else out[:, :, 0]


def trim(img, frac=TRIM):
    h, w = img.shape[:2]
    dy, dx = int(np.floor(h * frac + .5)), int(np.floor(w * frac + .5))  # = Math.round
    return img[dy:h - dy, dx:w - dx]


def to_gray(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def bits_to_hex(bits):
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    out = []
    for i in range(0, len(bits), 4):
        nib = bits[i] << 3 | bits[i + 1] << 2 | bits[i + 2] << 1 | bits[i + 3]
        out.append("0123456789abcdef"[nib])
    return "".join(out)


def dct_matrix(n):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    m[0] *= 1 / np.sqrt(2)
    return m * np.sqrt(2 / n)


_DCT = {32: dct_matrix(32), 64: dct_matrix(64)}


def ahash(gray):
    g = area_resize(gray, 8, 8)
    return g > g.mean()


def dhash(gray):
    g = area_resize(gray, 9, 8)
    return g[:, 1:] > g[:, :-1]


def phash(gray, size=8):
    n = size * 4
    g = area_resize(gray, n, n)
    d = _DCT[n] @ g @ _DCT[n].T
    low = d[:size, :size]
    return low > np.median(low)


def color_layout(rgb):
    c = area_resize(rgb, COLOR_W, COLOR_H).ravel()
    c = c - c.mean()
    return c / (np.linalg.norm(c) + 1e-9)


HIST_BINS = 51


def hsv_hist(rgb):
    """Framing- and rotation-invariant colour signature of the base image.
    (vectorised; same arithmetic as hsvHist() in features.js)"""
    px = rgb.reshape(-1, 3)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    mx, mn = px.max(1), px.min(1)
    v = mx / 255
    s = np.where(mx == 0, 0.0, (mx - mn) / np.where(mx == 0, 1, mx))
    vb = np.minimum((v * 3).astype(int), 2)
    d = np.where(mx == mn, 1, mx - mn)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4))
    hb = np.minimum((h / 6 * 8).astype(int), 7)
    sb = (s >= 0.6).astype(int)
    bins = np.where(s < 0.2, 48 + vb, (hb * 2 + sb) * 3 + vb)
    hist = np.bincount(bins, minlength=HIST_BINS).astype(np.float64)
    return np.sqrt(hist / len(px))


def extract(rgb):
    """rgb: HxWx3 uint8 card crop -> dict of fingerprints."""
    base = area_resize(trim(np.asarray(rgb)[..., :3]), BASE_W, BASE_H)
    gray = to_gray(base)
    return {
        "a": bits_to_hex(ahash(gray)),
        "d": bits_to_hex(dhash(gray)),
        "p": bits_to_hex(phash(gray, 8)),
        "P": bits_to_hex(phash(gray, 16)),
        "c": np.round(color_layout(base) * 127).astype(int).tolist(),
        "h": np.round(hsv_hist(base) * 127).astype(int).tolist(),
    }


# ---------- distances (used by evaluation; mirrored in web/js/matcher.js) ----------

def hex_to_bits(h):
    return np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8))


def stack_bits(hex_list):
    return np.stack([hex_to_bits(h) for h in hex_list]).astype(np.uint8)


def hamming_all(q_bits, gallery_bits):
    """normalised Hamming distance of one query to every gallery row (0..1)."""
    return (gallery_bits != q_bits[None, :]).mean(axis=1)


def color_dist_all(q_c, gallery_c):
    """1 - cosine similarity, rescaled to 0..1."""
    q = np.asarray(q_c, dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-9
    g = gallery_c / (np.linalg.norm(gallery_c, axis=1, keepdims=True) + 1e-9)
    return (1 - g @ q) / 2
