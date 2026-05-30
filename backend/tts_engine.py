"""
Text-to-Speech engine for AI Dungeon Master (Phase 9).

The DM narrates aloud. Providers (TTS_PROVIDER env var):
  "coqui"      — Coqui TTS (GPU-accelerated, ~high quality British voice)
  "pyttsx3"    — OS built-in TTS (zero install, lower quality)
  "elevenlabs" — ElevenLabs API (highest quality, requires API key)
  "disabled"   — No TTS (default)
  "auto"       — Try coqui → pyttsx3 → disabled

All providers expose the same interface:
    audio_b64 = await synthesize(text)   → base64-encoded WAV or MP3 string
                                            (suitable for data:audio/... URLs)

Run directly to test:
    python tts_engine.py "Your wand tip blazes with cold white light."
"""
import asyncio
import base64
import io
import os
import re
import threading
from typing import Any, Optional

TTS_PROVIDER    = os.getenv("TTS_PROVIDER",    "disabled")
TTS_VOICE       = os.getenv("TTS_VOICE",       "")  # provider-specific voice name
COQUI_MODEL     = os.getenv("COQUI_MODEL",     "tts_models/en/vctk/vits")
ELEVENLABS_KEY  = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE = os.getenv("ELEVENLABS_VOICE", "Rachel")

# ── Text pre-processing ───────────────────────────────────────────────────────

_MARKUP_RE  = re.compile(r"\[(?:SPELL_CAST|LOCATION_CHANGE|COMBAT_START|DIFFICULTY|PLAYER STATE):[^\]]*\]")
_MD_BOLD    = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC  = re.compile(r"\*(.+?)\*")
_MD_HEADER  = re.compile(r"^#{1,4}\s+", re.MULTILINE)
_STAGE_DIR  = re.compile(r"\*[^*]+\*")   # *stage directions*


def preprocess(text: str, max_chars: int = 600) -> str:
    """
    Strip markup, markdown, and stage directions; truncate to max_chars.
    Returns clean plain text suitable for TTS.
    """
    t = _MARKUP_RE.sub("", text)
    t = _MD_HEADER.sub("", t)
    t = _MD_BOLD.sub(r"\1", t)
    t = _STAGE_DIR.sub("", t)          # remove *stage directions* entirely
    t = _MD_ITALIC.sub(r"\1", t)
    t = re.sub(r"\n+", " ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    # Truncate at sentence boundary near max_chars
    if len(t) > max_chars:
        cut = t.rfind(".", 0, max_chars)
        t = t[: cut + 1] if cut > max_chars // 2 else t[:max_chars]
    return t


# ── Coqui TTS ─────────────────────────────────────────────────────────────────

_coqui_tts: Any = None
_coqui_lock = threading.Lock()
_coqui_error: Optional[str] = None

# Default speaker for VCTK multi-speaker model — good British male voice
_VCTK_SPEAKER = TTS_VOICE or "p273"


def _load_coqui() -> None:
    global _coqui_tts, _coqui_error
    try:
        from TTS.api import TTS as CoquiAPI
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TTS/Coqui] Loading {COQUI_MODEL} on {device}…")
        tts = CoquiAPI(COQUI_MODEL).to(device)
        with _coqui_lock:
            _coqui_tts = tts
        print(f"[TTS/Coqui] Ready ({device})")
    except Exception as e:
        with _coqui_lock:
            _coqui_error = str(e)
        print(f"[TTS/Coqui] Failed: {e}")


def warmup_coqui() -> None:
    """Start Coqui model loading in a background thread (called at startup)."""
    if TTS_PROVIDER not in ("coqui", "auto"):
        return
    t = threading.Thread(target=_load_coqui, daemon=True)
    t.start()


def _get_coqui():
    with _coqui_lock:
        if _coqui_tts is not None:
            return _coqui_tts
        if _coqui_error is not None:
            return None
    # Still loading — wait up to 120s
    import time
    for _ in range(120):
        time.sleep(1)
        with _coqui_lock:
            if _coqui_tts is not None:
                return _coqui_tts
            if _coqui_error is not None:
                return None
    return None


def _synth_coqui(text: str) -> Optional[bytes]:
    tts = _get_coqui()
    if tts is None:
        return None
    try:
        buf = io.BytesIO()
        # VCTK is multi-speaker; ljspeech/glow-tts are single speaker
        is_multispeaker = hasattr(tts, "speakers") and tts.speakers
        if is_multispeaker:
            wav = tts.tts(text=text, speaker=_VCTK_SPEAKER)
        else:
            wav = tts.tts(text=text)
        # wav is a list of floats; save as WAV via soundfile
        import soundfile as sf
        import numpy as np
        sample_rate = tts.synthesizer.output_sample_rate
        wav_np = np.array(wav, dtype=np.float32)
        sf.write(buf, wav_np, sample_rate, format="WAV")
        return buf.getvalue()
    except Exception as e:
        print(f"[TTS/Coqui] Synthesis error: {e}")
        return None


# ── pyttsx3 (OS TTS) ──────────────────────────────────────────────────────────

def _synth_pyttsx3(text: str) -> Optional[bytes]:
    try:
        import pyttsx3
        import tempfile, os as _os
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 0.9)
        # Pick a British voice if available
        voices = engine.getProperty("voices")
        for v in voices:
            if "english" in v.name.lower() or "uk" in v.id.lower():
                engine.setProperty("voice", v.id)
                break
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        engine.save_to_file(text, tmp)
        engine.runAndWait()
        with open(tmp, "rb") as f:
            data = f.read()
        _os.unlink(tmp)
        return data if len(data) > 100 else None
    except Exception as e:
        print(f"[TTS/pyttsx3] Error: {e}")
        return None


# ── ElevenLabs ────────────────────────────────────────────────────────────────

async def _synth_elevenlabs(text: str) -> Optional[bytes]:
    if not ELEVENLABS_KEY:
        return None
    try:
        import httpx
        # Resolve voice ID
        voice = TTS_VOICE or ELEVENLABS_VOICE
        headers = {"xi-api-key": ELEVENLABS_KEY, "Content-Type": "application/json"}
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.6, "similarity_boost": 0.8},
        }
        async with httpx.AsyncClient(timeout=30.0) as c:
            resp = await c.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
                headers=headers, json=payload
            )
            if resp.status_code == 200:
                return resp.content  # MP3 bytes
    except Exception as e:
        print(f"[TTS/ElevenLabs] Error: {e}")
    return None


# ── Main synthesize API ────────────────────────────────────────────────────────

async def synthesize(text: str) -> Optional[str]:
    """
    Convert text to speech. Returns base64-encoded WAV/MP3 data URI string,
    or None if TTS is disabled or all providers fail.
    """
    clean = preprocess(text)
    if not clean or TTS_PROVIDER == "disabled":
        return None

    raw: Optional[bytes] = None
    mime = "audio/wav"

    if TTS_PROVIDER in ("coqui", "auto"):
        raw = await asyncio.get_event_loop().run_in_executor(None, _synth_coqui, clean)

    if raw is None and TTS_PROVIDER in ("pyttsx3", "auto"):
        raw = await asyncio.get_event_loop().run_in_executor(None, _synth_pyttsx3, clean)

    if raw is None and TTS_PROVIDER == "elevenlabs":
        raw = await _synth_elevenlabs(clean)
        mime = "audio/mpeg"

    if raw is None:
        return None
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def tts_status() -> dict:
    provider = TTS_PROVIDER
    if provider == "coqui":
        return {
            "provider": "coqui",
            "ready": _coqui_tts is not None,
            "loading": _coqui_tts is None and _coqui_error is None,
            "error": _coqui_error,
            "model": COQUI_MODEL,
        }
    if provider == "elevenlabs":
        return {"provider": "elevenlabs", "ready": bool(ELEVENLABS_KEY), "key_set": bool(ELEVENLABS_KEY)}
    if provider == "pyttsx3":
        return {"provider": "pyttsx3", "ready": True}
    return {"provider": "disabled", "ready": False}


# ── CLI test ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "Your wand tip blazes with cold white light, illuminating the ancient stone corridor."
    print(f"[TTS] Provider: {TTS_PROVIDER}")
    print(f"[TTS] Text    : {text[:80]}")

    async def _test():
        result = await synthesize(text)
        if result:
            # Save to file for manual listening
            b64 = result.split(",", 1)[1]
            raw = base64.b64decode(b64)
            out = "tts_test.wav"
            with open(out, "wb") as f:
                f.write(raw)
            print(f"[TTS] Saved {len(raw)} bytes → {out}")
        else:
            print("[TTS] No audio returned (check TTS_PROVIDER env var)")

    asyncio.run(_test())
