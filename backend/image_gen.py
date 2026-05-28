"""
Image generation hook for AI Dungeon Master.
Phase 1: Returns placeholder styled scene descriptions + placeholder images.
Phase 3: Hook into Stable Diffusion API (AUTOMATIC1111 or ComfyUI).
"""
import os
import hashlib
import json
from typing import Optional, Dict

IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "placeholder")
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
        # Add situation context
        situation_short = situation[:100]
        base += f", scene shows: {situation_short}"

    return base


async def generate_scene_image(
    prompt: str,
    location_id: str = "loc_001",
    width: int = 768,
    height: int = 432
) -> Dict:
    """
    Generate a scene image.
    Phase 1: Returns placeholder data URL.
    Phase 3: Calls Stable Diffusion API.
    """
    if IMAGE_PROVIDER == "stable_diffusion":
        return await _generate_with_sd(prompt, width, height)
    else:
        return _generate_placeholder(location_id, prompt)


def _generate_placeholder(location_id: str, prompt: str) -> Dict:
    """Return placeholder scene data (gradient + description)."""
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
        "message": "Scene rendering powered by imagination (SD integration coming in Phase 3)"
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
                "image_base64": image_b64,
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
