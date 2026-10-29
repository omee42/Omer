"""Pofi the owl: a procedurally drawn mascot with talking beak, blinking and waving."""
import math
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFilter

from .gfx import darken, lighten, rgb, shaded

BODY = "#8E7CF8"
BELLY = "#FFE6C4"
DISC = "#C3B8FF"
BEAK = "#FFA53A"
PUPIL = (45, 37, 72)

W0, H0 = 520, 480  # design grid (owl is drawn around x=200, shifted by OX)
OX = 60


def _ellipse_mask(size, box, k):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).ellipse([v * k for v in box], fill=255)
    return m


def _wing(size, k, side, angle):
    """Wing layer rotated around the shoulder; side=-1 left, +1 right."""
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    cx = 200 + side * 128 + OX
    box = (cx - 46, 225, cx + 46, 420)
    m = _ellipse_mask(size, box, k)
    layer.alpha_composite(shaded(m, darken(BODY, 0.12), 0.3, 0.3))
    pivot = ((200 + side * 112 + OX) * k, 250 * k)
    return layer.rotate(-side * angle, resample=Image.BICUBIC, center=pivot)


@lru_cache(maxsize=256)
def owl(height: int, mouth: int = 0, eyes: str = "open", wing: int = 0) -> Image.Image:
    """mouth 0..5, eyes in {'open','half','happy'}, wing 0=rest 1/2=waving frames."""
    ss = 2
    k = height * ss / H0
    size = (int(W0 * k), int(H0 * k))
    img = Image.new("RGBA", size, (0, 0, 0, 0))

    wing_angles = {0: (-12, -12), 1: (-12, -125), 2: (-12, -95)}
    la, ra = wing_angles[wing]
    img.alpha_composite(_wing(size, k, -1, la))
    img.alpha_composite(_wing(size, k, 1, ra))

    bsize = (int(400 * k), size[1])
    b = Image.new("RGBA", bsize, (0, 0, 0, 0))
    d = ImageDraw.Draw(b)
    tuft = darken(BODY, 0.18)
    d.polygon([(v[0] * k, v[1] * k) for v in [(92, 205), (104, 92), (172, 158)]], fill=tuft)
    d.polygon([(v[0] * k, v[1] * k) for v in [(308, 205), (296, 92), (228, 158)]], fill=tuft)

    b.alpha_composite(shaded(_ellipse_mask(bsize, (55, 132, 345, 468), k), BODY, 0.42, 0.32))
    b.alpha_composite(shaded(_ellipse_mask(bsize, (112, 288, 288, 456), k), BELLY, 0.5, 0.12))
    d = ImageDraw.Draw(b)
    fc = darken(BELLY, 0.1)
    for row, y in enumerate((342, 378, 414)):
        xs = (165, 200, 235) if row != 1 else (182, 218)
        for x in xs:
            d.arc(((x - 14) * k, (y - 10) * k, (x + 14) * k, (y + 10) * k), 20, 160, fill=fc, width=int(4 * k))

    disc = Image.new("L", bsize, 0)
    dd = ImageDraw.Draw(disc)
    for cx in (143, 257):
        dd.ellipse(((cx - 74) * k, 176 * k, (cx + 74) * k, 324 * k), fill=255)
    disc = disc.filter(ImageFilter.GaussianBlur(2 * k))
    b.alpha_composite(shaded(disc, DISC, 0.35, 0.08))
    d = ImageDraw.Draw(b)

    for cx in (143, 257):
        cy = 250
        if eyes == "happy":
            d.arc(((cx - 34) * k, (cy - 18) * k, (cx + 34) * k, (cy + 36) * k), 200, 340,
                  fill=PUPIL, width=int(10 * k))
            continue
        d.ellipse(((cx - 52) * k, (cy - 52) * k, (cx + 52) * k, (cy + 52) * k), fill=(255, 255, 255))
        px = cx + (8 if cx < 200 else -8)
        d.ellipse(((px - 31) * k, (cy - 27) * k, (px + 31) * k, (cy + 35) * k), fill=PUPIL)
        d.ellipse(((px - 18) * k, (cy - 18) * k, (px - 2) * k, (cy - 2) * k), fill=(255, 255, 255))
        d.ellipse(((px + 8) * k, (cy + 10) * k, (px + 16) * k, (cy + 18) * k), fill=(255, 255, 255))
        if eyes == "half":
            d.chord(((cx - 54) * k, (cy - 54) * k, (cx + 54) * k, (cy + 54) * k), 180, 360, fill=rgb(DISC))
            d.line(((cx - 52) * k, cy * k, (cx + 52) * k, cy * k), fill=darken(DISC, 0.25), width=int(5 * k))

    # beak: upper part stays, lower part drops to show the mouth
    gap = mouth * 5.0
    top = 284
    if gap > 0.5:
        d.ellipse(((200 - 21) * k, (top + 12) * k, (200 + 21) * k, (top + 24 + gap) * k), fill=(140, 30, 60))
        d.ellipse(((200 - 11) * k, (top + 14 + gap * 0.65) * k, (200 + 11) * k, (top + 24 + gap) * k),
                  fill=(255, 120, 140))
    d.polygon([((200 - 28) * k, top * k), ((200 + 28) * k, top * k), (200 * k, (top + 34 - min(gap, 12)) * k)],
              fill=rgb(BEAK))
    d.polygon([((200 - 18) * k, (top + 4) * k), ((200 - 3) * k, (top + 4) * k), ((200 - 7) * k, (top + 15) * k)],
              fill=lighten(BEAK, 0.45))
    if gap > 0.5:
        ly = top + 21 + gap
        d.polygon([((200 - 17) * k, ly * k), ((200 + 17) * k, ly * k), (200 * k, (ly + 15) * k)],
                  fill=darken(BEAK, 0.12))

    cheeks = Image.new("RGBA", bsize, (0, 0, 0, 0))
    cd = ImageDraw.Draw(cheeks)
    for cx in (92, 308):
        cd.ellipse(((cx - 24) * k, 296 * k, (cx + 24) * k, 326 * k), fill=(255, 120, 160, 120))
    b.alpha_composite(cheeks.filter(ImageFilter.GaussianBlur(3 * k)))

    d = ImageDraw.Draw(b)
    for cx in (160, 240):
        for off in (-14, 0, 14):
            d.ellipse(((cx + off - 11) * k, 448 * k, (cx + off + 11) * k, 474 * k), fill=rgb(BEAK))

    img.alpha_composite(b, (int(OX * k), 0))
    return img.resize((size[0] // ss, size[1] // ss), Image.LANCZOS)


class MascotAnimator:
    """Chooses the owl pose for a time t, given the narration loudness envelope."""

    def __init__(self, mouth_env, fps):
        self.env = mouth_env  # per-frame 0..1
        self.fps = fps

    def mouth(self, t):
        i = int(t * self.fps)
        if i < 0 or i >= len(self.env):
            return 0
        return int(round(min(1.0, self.env[i] * 1.25) * 5))

    @staticmethod
    def eyes(t, happy=False):
        if happy:
            return "happy"
        # deterministic blink every ~3.4 s with a little variation
        cycle = 3.4 + 0.9 * math.sin(int(t / 3.4) * 1.7)
        phase = t % cycle
        if phase < 0.07 or 0.17 < phase < 0.22:
            return "half"
        if phase < 0.17:
            return "happy"
        return "open"

    def pose(self, t, height, waving=False, happy=False):
        wing = 0
        if waving:
            wing = 1 if int(t * 4) % 2 == 0 else 2
        return owl(height, self.mouth(t), self.eyes(t, happy), wing)
