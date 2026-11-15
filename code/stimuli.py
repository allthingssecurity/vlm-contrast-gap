"""Stimulus generation. Contrast is Weber luminance contrast C = (Y_bg - Y_text)/Y_bg with Y the sRGB relative
luminance (IEC 61966-2-1 transfer curve); text is composited in linear light as Y = Y_bg * (1 - C * alpha) with the
anti-aliased glyph coverage alpha, then re-encoded to 8-bit sRGB. Achieved contrast (after quantisation) is reported."""
import os, numpy as np, matplotlib
from PIL import Image, ImageDraw, ImageFont

FONT = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")
SLOAN = list("CDHKNORSVZ")   # the ten Sloan letters used on Pelli-Robson / ETDRS charts

def srgb_to_lin(v):
    v = np.asarray(v, dtype=np.float64) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)

def lin_to_srgb8(L):
    L = np.clip(L, 0, 1)
    v = np.where(L <= 0.0031308, 12.92 * L, 1.055 * L ** (1 / 2.4) - 0.055)
    return np.clip(np.round(v * 255.0), 0, 255).astype(np.uint8)

def font_for_cap_height(cap_px):
    """Return a font whose capital 'H' is cap_px pixels tall."""
    size = cap_px
    for _ in range(30):
        f = ImageFont.truetype(FONT, size); b = f.getbbox("H"); h = b[3] - b[1]
        if abs(h - cap_px) <= 0.5: return f
        size = max(4, int(round(size * cap_px / max(h, 1))))
    return f

def text_alpha(size_hw, lines, font, xy):
    """Anti-aliased coverage mask in [0,1] for the given text lines."""
    m = Image.new("L", (size_hw[1], size_hw[0]), 0); d = ImageDraw.Draw(m)
    x, y = xy
    for ln in lines:
        d.text((x, y), ln, fill=255, font=font)
        b = font.getbbox("Hg"); y += int((b[3] - b[1]) * 1.35)
    return np.asarray(m, dtype=np.float64) / 255.0

def composite(img8, alpha, C):
    """Darken text pixels multiplicatively in linear light (exact Weber contrast C w.r.t. the local background)."""
    L = srgb_to_lin(img8)
    out = lin_to_srgb8(L * (1.0 - C * alpha[..., None]))
    full = alpha >= 0.999
    if full.any():
        Yb = (srgb_to_lin(img8) @ np.array([0.2126, 0.7152, 0.0722]))[full]
        Yt = (srgb_to_lin(out) @ np.array([0.2126, 0.7152, 0.0722]))[full]
        ok = Yb > 1e-6
        achieved = float(np.median(1 - Yt[ok] / Yb[ok])); changed = float((out[full] != img8[full]).any(-1).mean())
    else:
        achieved, changed = float("nan"), 0.0
    return out, achieved, changed

def letter_chart(letters, cap_px, C, bg=255, size=512):
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    f = font_for_cap_height(cap_px)
    widths = [f.getbbox(ch)[2] - f.getbbox(ch)[0] for ch in letters]
    gap = int(cap_px * 0.6); total = sum(widths) + gap * (len(letters) - 1)
    x = (size - total) // 2; top = f.getbbox("H")[1]; y = (size - cap_px) // 2 - top
    m = Image.new("L", (size, size), 0); d = ImageDraw.Draw(m)
    for ch, w in zip(letters, widths):
        d.text((x - f.getbbox(ch)[0], y), ch, fill=255, font=f); x += w + gap
    alpha = np.asarray(m, dtype=np.float64) / 255.0
    return composite(img, alpha, C)

def min_contrast_8bit(bg):
    """Smallest non-zero Weber contrast representable by darkening an 8-bit grey level bg by one step."""
    return float(1 - srgb_to_lin(bg - 1) / srgb_to_lin(bg))

if __name__ == "__main__":
    os.makedirs("samples", exist_ok=True)
    for C in (1.0, 0.1, 0.03, 0.0089):
        im, a, ch = letter_chart(list("KDR"), 48, C)
        Image.fromarray(im).save(f"samples/chart_C{C}.png"); print(C, "achieved", a, "changed", ch)
    print("8-bit floor at bg=255:", min_contrast_8bit(255), " bg=128:", min_contrast_8bit(128))
