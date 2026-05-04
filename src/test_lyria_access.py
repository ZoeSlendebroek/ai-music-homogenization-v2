#!/usr/bin/env python3
"""
test_lyria_access.py

Tests access to Lyria 3 Clip API (lyria-3-clip-preview).
Generates one 30s clip from a sample metal prompt and saves it.

Usage:
    python src/test_lyria_access.py
"""

import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from google import genai
except ImportError:
    sys.exit("pip install google-genai")

ROOT = Path(__file__).resolve().parent.parent


def _load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def test():
    _load_env()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Add GEMINI_API_KEY=your_key to your .env file")

    prompt = (
        "Instrumental heavy metal at 142.0 BPM (Allegro), driving, "
        "syncopated, groove-forward. Strongly melodic, harmonic-forward "
        "(harmonic ratio 0.81). Structured with repeating sections. "
        "Contemporary heavy metal (modern production, post-metal elements). "
        "No vocals."
    )

    print(f"Prompt: {prompt}\n")
    print("Sending request to Lyria 3 Clip API...")

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model="lyria-3-clip-preview",
            contents=prompt,
        )
    except Exception as e:
        sys.exit(f"\n[FAILED] {e}")

    # Find audio in response
    audio_data = None
    try:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                audio_data = part.inline_data.data
                mime = part.inline_data.mime_type
                print(f"Received audio — mime: {mime}, size: {len(audio_data)} bytes")
                break
    except Exception as e:
        print(f"[parse error] {e}")
        print(f"Raw response: {response}")
        sys.exit(1)

    if not audio_data:
        print(f"No audio found in response. Full response:\n{response}")
        sys.exit(1)

    out_dir = ROOT / "data" / "audio" / "lyria"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_output.mp3"

    # audio_data may be bytes or base64 string depending on SDK version
    if isinstance(audio_data, str):
        audio_data = base64.b64decode(audio_data)

    out_path.write_bytes(audio_data)
    print(f"Saved to {out_path}")
    print("\nAccess confirmed. Lyria 3 Clip is working.")


if __name__ == "__main__":
    test()
