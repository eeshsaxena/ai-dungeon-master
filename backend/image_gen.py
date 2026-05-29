"""
Scene image generation for AI Dungeon Master.

Providers (IMAGE_PROVIDER env var):
  - "procedural" (default): renders an atmospheric PNG server-side with Pillow —
    per-location silhouettes, mood-driven particles, gradient skies. No model,
    no GPU, no network; the game ships with real generated scene art out of the box.
  - "stable_diffusion": calls an AUTOMATIC1111 txt2img API for photoreal art
    (Phase 3 upgrade, needs SD_API_URL running).
  - "placeholder": legacy colour/description payload (no image).
"""
import os
import io
import base64
import hashlib
import random
from typing import Optional, Dict, Tuple

IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "procedural")
SD_API_URL = os.getenv("SD_API_URL", "http://127.0.0.1:7860")


# ── Scene Prompt Templates ─────────────────────────────────────────────────────

LOCATION_PROMPTS = {
    "loc_001": "a warm cozy wizarding pub interior at night, butterbeer on tables, candles floating, dark wood, magical atmosphere, Harry Potter style, detailed fantasy art, golden lighting",
    "loc_002": "Hogwarts castle at night, Gothic architecture, moonlit towers, magical aurora, torches flickering in windows, misty Scottish highlands below, epic fantasy digital art",
    "loc_003": "dark cobblestone alley in a wizarding city, shadowy shop windows filled with cursed objects, dim green lanterns, fog, sinister atmosphere, gothic fantasy art",
    "loc_004": "ancient forbidden forest at night, massive dark trees, bioluminescent fungi, mist, shadows with red eyes, mysterious and dangerous, dark fantasy art",
    "loc_005": "magical government building interior, gleaming marble floors, enchanted paper airplanes flying overhead, grand architecture, Ministry of Magic style",
    "loc_006": "grim island prison surrounded by stormy sea, stone walls, isolated tower, dark storm clouds, Azkaban prison, dark gothic fantasy",
    "loc_007": "haunted village at dusk, old cemetery, stone war memorial, crumbling cottage ruins, magical residue in the air, melancholy and eerie, British countryside"
}

ATMOSPHERE_OVERLAYS = {
    "warm": "warm golden tones, cozy lighting",
    "cold": "cold blue tones, harsh shadows",
    "mysterious": "purple mist, ethereal glow, mysterious atmosphere",
    "dangerous": "red warning colors, ominous shadows",
    "ancient": "sepia tones, aged textures, historic atmosphere",
    "magical": "sparkles, magical particles, iridescent colors"
}

ENEMY_PROMPTS = {
    "Dementor": "a dementor wraith floating in darkness, tattered robes, soul-draining creature, cold mist, Harry Potter universe",
    "Death Eater": "a Death Eater in dark robes and white mask, wand raised, dark magic sparks, threatening stance",
    "Acromantula": "a massive magical spider with eight glowing eyes, forest setting, dangerous and enormous",
    "Troll": "a mountain troll, enormous, green-grey skin, holding a club, dimly lit stone cave",
    "Werewolf": "a werewolf mid-transformation, wild eyes, teeth bared, moonlight, dramatic dark fantasy art",
    "Hollow Mage": "a dark mage whose form absorbs light, void where eyes should be, crackling dark energy, final boss aesthetic"
}

# Per-location palettes for the procedural renderer: (sky top, sky bottom, silhouette, accent/glow)
SCENE_PALETTES = {
    "loc_001": ("#3A2416", "#160C06", "#0E0804", "#FFB347"),  # warm pub
    "loc_002": ("#0A1628", "#020610", "#05080F", "#6FA8FF"),  # Hogwarts night
    "loc_003": ("#1A0A2E", "#070312", "#05030A", "#3BE08A"),  # Knockturn (green lanterns)
    "loc_004": ("#0D1F12", "#020A05", "#01060B", "#39E07A"),  # Forbidden Forest
    "loc_005": ("#1C1C2E", "#080810", "#0A0A14", "#8FB7FF"),  # Ministry
    "loc_006": ("#0F1423", "#02040A", "#04060C", "#7A86C8"),  # Azkaban storm
    "loc_007": ("#241F1A", "#0A0806", "#070605", "#C9B79A"),  # Godric's Hollow
}

MOOD_PARTICLES = {
    "warm": "embers", "dangerous": "embers", "cold": "snow",
    "mysterious": "stars", "magical": "stars", "ancient": "mist", "melancholic": "mist",
}


def build_scene_prompt(
    location_id: str,
    situation: str = "",
    mood: str = "mysterious",
    enemy: Optional[str] = None
) -> str:
    """Build a Stable Diffusion prompt for the current scene."""
    base = LOCATION_PROMPTS.get(location_id, "a magical wizarding world location, fantasy art")
    if enemy and enemy in ENEMY_PROMPTS:
        base += f", {ENEMY_PROMPTS[enemy]}"
    overlay = ATMOSPHERE_OVERLAYS.get(mood, "")
    if overlay:
        base += f", {overlay}"
    base += ", highly detailed, cinematic, 8k resolution, concept art style"
    if situation:
        base += f", scene shows: {situation[:100]}"
    return base


async def generate_scene_image(
    prompt: str,
    location_id: str = "loc_001",
    width: int = 768,
    height: int = 432,
    mood: str = "mysterious",
) -> Dict:
    """Generate a scene image using the configured provider."""
    if IMAGE_PROVIDER == "stable_diffusion":
        return await _generate_with_sd(prompt, width, height)
    if IMAGE_PROVIDER == "placeholder":
        return _generate_placeholder(location_id, prompt)
    # Default: procedural render (falls back to placeholder if Pillow is missing)
    return _generate_procedural(location_id, prompt, mood, width, height) \
        or _generate_placeholder(location_id, prompt)


# ── Procedural renderer ─────────────────────────────────────────────────────────

def _hex(c: str) -> Tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _lerp(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _generate_procedural(location_id: str, prompt: str, mood: str, w: int, h: int) -> Optional[Dict]:
    """Render an atmospheric scene PNG with Pillow. Deterministic per (scene, prompt)."""
    try:
        from PIL import Image, ImageDraw, ImageFilter
    except Exception:
        return None

    top, bottom, sil, accent = (
        _hex(x) for x in SCENE_PALETTES.get(location_id, ("#1A1A2E", "#05050A", "#08080F", "#FFD700"))
    )
    rng = random.Random(int(hashlib.sha256((location_id + prompt).encode()).hexdigest()[:8], 16))

    # Sky gradient (build a 1px column, then stretch — fast)
    col = Image.new("RGB", (1, h))
    cpx = col.load()
    for y in range(h):
        cpx[0, y] = _lerp(top, bottom, y / max(h - 1, 1))
    img = col.resize((w, h)).convert("RGBA")

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    horizon = int(h * 0.68)

    _draw_silhouette(location_id, draw, w, h, horizon, sil, accent, rng)
    _draw_particles(draw, w, h, horizon, mood, accent, rng)

    # Soft vignette for depth
    vign = Image.new("L", (w, h), 0)
    ImageDraw.Draw(vign).ellipse([-w * 0.2, -h * 0.2, w * 1.2, h * 1.2], fill=120)
    vign = vign.filter(ImageFilter.GaussianBlur(w * 0.15))
    dark = Image.new("RGBA", (w, h), (0, 0, 0, 110))
    img = Image.composite(img, Image.alpha_composite(img, dark), vign)

    img = Image.alpha_composite(img, overlay).convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {
        "type": "generated",
        "provider": "procedural",
        "location_id": location_id,
        "image_base64": f"data:image/png;base64,{b64}",
        "prompt": prompt,
        "sd_ready": False,
    }


def _draw_silhouette(loc, d, w, h, hy, sil, accent, rng):
    """Draw a recognizable foreground silhouette per location."""
    solid = sil + (255,)
    glow = accent + (210,)

    if loc == "loc_002":  # Hogwarts — castle towers + moon
        d.ellipse([w * 0.72, h * 0.12, w * 0.86, h * 0.30], fill=accent + (120,))
        x = 0
        while x < w:
            tw = rng.randint(int(w * 0.05), int(w * 0.11))
            th = rng.randint(int(h * 0.18), int(h * 0.42))
            ty = h - th
            d.rectangle([x, ty, x + tw, h], fill=solid)
            d.polygon([(x - 3, ty), (x + tw + 3, ty), (x + tw / 2, ty - th * 0.25)], fill=solid)
            for _ in range(rng.randint(1, 3)):  # lit windows
                wx = x + rng.randint(4, max(5, tw - 6)); wy = ty + rng.randint(8, max(9, int(th * 0.6)))
                d.rectangle([wx, wy, wx + 3, wy + 5], fill=glow)
            x += tw + rng.randint(int(w * 0.01), int(w * 0.04))

    elif loc == "loc_004":  # Forbidden Forest — trees + eyes
        for _ in range(rng.randint(12, 18)):
            tx = rng.randint(0, w); tw = rng.randint(int(w * 0.02), int(w * 0.05))
            th = rng.randint(int(h * 0.25), int(h * 0.55))
            d.polygon([(tx - tw, h), (tx + tw, h), (tx, h - th)], fill=solid)
        for _ in range(2):  # glowing eyes
            ex = rng.randint(int(w * 0.2), int(w * 0.8)); ey = rng.randint(hy, h - 10)
            d.ellipse([ex, ey, ex + 4, ey + 4], fill=(220, 40, 40, 230))
            d.ellipse([ex + 8, ey, ex + 12, ey + 4], fill=(220, 40, 40, 230))

    elif loc == "loc_003":  # Knockturn Alley — narrow buildings + green lanterns
        x = 0
        while x < w:
            bw = rng.randint(int(w * 0.08), int(w * 0.16)); bh = rng.randint(int(h * 0.4), int(h * 0.75))
            d.rectangle([x, h - bh, x + bw, h], fill=solid)
            lx = x + bw * 0.5; ly = h - bh * rng.uniform(0.3, 0.7)
            d.ellipse([lx - 4, ly - 4, lx + 4, ly + 4], fill=glow)
            x += bw + rng.randint(2, 8)

    elif loc == "loc_006":  # Azkaban — lone tower on island, storm
        d.line([(rng.randint(0, w), 0), (rng.randint(0, w), int(h * 0.5))], fill=accent + (180,), width=2)
        cx = w * 0.5
        d.ellipse([cx - w * 0.18, h - 18, cx + w * 0.18, h + 30], fill=solid)  # island
        d.rectangle([cx - w * 0.05, h * 0.35, cx + w * 0.05, h], fill=solid)   # tower
        d.polygon([(cx - w * 0.06, h * 0.35), (cx + w * 0.06, h * 0.35), (cx, h * 0.27)], fill=solid)
        d.rectangle([cx - 3, h * 0.45, cx + 3, h * 0.5], fill=glow)

    elif loc == "loc_001":  # Three Broomsticks — warm windows + floating candles
        for i in range(3):
            wx = w * (0.12 + i * 0.3)
            d.rounded_rectangle([wx, h * 0.3, wx + w * 0.16, h * 0.7], radius=12, fill=accent + (90,))
        for _ in range(rng.randint(8, 14)):
            cx = rng.randint(0, w); cy = rng.randint(int(h * 0.15), int(h * 0.6))
            d.ellipse([cx, cy, cx + 3, cy + 3], fill=glow)

    elif loc == "loc_005":  # Ministry — columns + flying memos
        for i in range(6):
            cx = w * (0.08 + i * 0.16)
            d.rectangle([cx, h * 0.2, cx + w * 0.04, h], fill=solid)
        for _ in range(rng.randint(6, 10)):
            px = rng.randint(0, w); py = rng.randint(10, int(h * 0.5))
            d.polygon([(px, py), (px + 10, py + 3), (px, py + 6)], fill=glow)

    elif loc == "loc_007":  # Godric's Hollow — gravestones + moon
        d.ellipse([w * 0.1, h * 0.12, w * 0.22, h * 0.30], fill=accent + (110,))
        for _ in range(rng.randint(6, 10)):
            gx = rng.randint(0, w); gh = rng.randint(int(h * 0.1), int(h * 0.22))
            d.rounded_rectangle([gx, h - gh, gx + w * 0.04, h], radius=8, fill=solid)

    else:  # generic rolling hills
        d.polygon([(0, h), (0, hy), (w * 0.5, hy - 20), (w, hy), (w, h)], fill=solid)


def _draw_particles(d, w, h, hy, mood, accent, rng):
    """Sprinkle mood-appropriate particles across the sky."""
    kind = MOOD_PARTICLES.get(mood, "stars")
    n = 60
    if kind == "stars":
        for _ in range(n):
            x, y = rng.randint(0, w), rng.randint(0, hy)
            a = rng.randint(60, 200); s = rng.choice([1, 1, 2])
            d.ellipse([x, y, x + s, y + s], fill=(255, 255, 255, a))
    elif kind == "embers":
        for _ in range(n):
            x, y = rng.randint(0, w), rng.randint(0, h)
            a = rng.randint(60, 180)
            d.ellipse([x, y, x + 2, y + 2], fill=accent + (a,))
    elif kind == "snow":
        for _ in range(n):
            x, y = rng.randint(0, w), rng.randint(0, h)
            d.ellipse([x, y, x + 2, y + 2], fill=(210, 230, 255, rng.randint(80, 180)))
    else:  # mist bands
        for _ in range(5):
            y = rng.randint(int(h * 0.4), h)
            d.rectangle([0, y, w, y + rng.randint(6, 16)], fill=(200, 200, 210, 22))


def _generate_placeholder(location_id: str, prompt: str) -> Dict:
    """Return placeholder scene data (colour + description)."""
    location_colors = {
        "loc_001": {"bg": "#2C1810", "accent": "#D4A017", "name": "The Three Broomsticks"},
        "loc_002": {"bg": "#0A1628", "accent": "#4A7BC8", "name": "Hogwarts Castle"},
        "loc_003": {"bg": "#1A0A2E", "accent": "#6B21A8", "name": "Knockturn Alley"},
        "loc_004": {"bg": "#0D1F12", "accent": "#22C55E", "name": "The Forbidden Forest"},
        "loc_005": {"bg": "#1C1C2E", "accent": "#60A5FA", "name": "Ministry of Magic"},
        "loc_006": {"bg": "#0F0F23", "accent": "#6366F1", "name": "Azkaban"},
        "loc_007": {"bg": "#1F1B18", "accent": "#78716C", "name": "Godric's Hollow"},
    }
    colors = location_colors.get(location_id, {"bg": "#1A1A2E", "accent": "#FFD700", "name": "Unknown Location"})
    return {
        "type": "placeholder",
        "location_id": location_id,
        "location_name": colors["name"],
        "bg_color": colors["bg"],
        "accent_color": colors["accent"],
        "prompt": prompt,
        "sd_ready": False,
        "message": "Scene rendering powered by imagination",
    }


async def _generate_with_sd(prompt: str, width: int, height: int) -> Dict:
    """Call Stable Diffusion AUTOMATIC1111 API."""
    try:
        import httpx
        payload = {
            "prompt": prompt,
            "negative_prompt": "ugly, deformed, blurry, low quality, watermark, text, nsfw",
            "width": width,
            "height": height,
            "steps": 25,
            "cfg_scale": 7.5,
            "sampler_name": "DPM++ 2M Karras"
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{SD_API_URL}/sdapi/v1/txt2img", json=payload)
            data = response.json()
            image_b64 = data["images"][0]
            return {
                "type": "generated",
                "provider": "stable_diffusion",
                "image_base64": f"data:image/png;base64,{image_b64}",
                "prompt": prompt,
                "sd_ready": True
            }
    except Exception as e:
        return {
            "type": "error",
            "error": str(e),
            "message": "Stable Diffusion unavailable, using placeholder",
            "sd_ready": False
        }
