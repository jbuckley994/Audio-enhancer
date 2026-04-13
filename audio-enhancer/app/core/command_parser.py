"""
CommandParser — translates natural language instructions into PipelineConfig.

Uses an LLM (via Anthropic or OpenAI) with a JSON schema prompt.
Falls back to rule-based parsing if no API key is configured.
"""

import json
import logging
import os
import re
from typing import Optional

from app.models.schemas import PipelineConfig
from app.utils.model_loader import ModelLoader

logger = logging.getLogger("audio_enhancer.command_parser")


# ─── Keyword rules (fallback) ─────────────────────────────────────────────────

KEYWORD_RULES = {
    # Voice boost
    r"distant|far[\s-]?away|behind.*(wall|door)|faint|weak|low.?volume": {
        "voice_boost": True, "voice_boost_db": 9.0, "dynamic_gain": True,
        "spectral_subtraction": True, "noise_reduction_strength": 0.6,
    },
    r"boost|amplif|louder|enhance.*voice": {
        "voice_boost": True, "voice_boost_db": 6.0,
    },
    # Noise reduction
    r"remov.*noise|clean.*audio|noise.?reduc|background.?noise": {
        "noise_reduction": True, "noise_reduction_strength": 0.85,
    },
    r"heavy noise|extreme noise|very noisy": {
        "noise_reduction": True, "noise_reduction_strength": 0.95,
        "spectral_subtraction": True, "wiener_filter": True,
    },
    r"gentle|light|soft|preserve": {
        "noise_reduction_strength": 0.4, "preserve_weak_signals": True,
    },
    # Source separation
    r"separat|isolat|focus on|extract voice": {
        "source_separation": True, "separation_model": "demucs",
    },
    # No source separation
    r"no.?separat|skip.?separat|fast": {
        "source_separation": False,
    },
    # Transcription
    r"transcri|speech.?to.?text|text|words|what.*said": {
        "transcribe": True,
    },
    r"no.?transcri|skip.?transcri|audio only": {
        "transcribe": False,
    },
    # Whisper model size
    r"accura|best quality|slow.*process": {
        "whisper_model": "medium",
    },
    r"fast.*transcri|quick": {
        "whisper_model": "tiny",
    },
}


class CommandParser:
    """
    Parses a natural-language command into a PipelineConfig.

    Strategy:
    1. Try LLM (Anthropic claude-haiku or OpenAI gpt-3.5) if API key present
    2. Fall back to keyword/regex rule matching
    3. Fall back to PipelineConfig defaults
    """

    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
        self._anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self._openai_key = os.getenv("OPENAI_API_KEY")

    async def parse(self, command: str) -> PipelineConfig:
        """Parse a command string → PipelineConfig."""
        logger.info(f"Parsing command: {command!r}")

        # 1. Try LLM
        if self._anthropic_key:
            try:
                return await self._parse_with_anthropic(command)
            except Exception as e:
                logger.warning(f"Anthropic parse failed: {e}")

        if self._openai_key:
            try:
                return await self._parse_with_openai(command)
            except Exception as e:
                logger.warning(f"OpenAI parse failed: {e}")

        # 2. Rule-based fallback
        return self._parse_with_rules(command)

    # ── LLM parsing ──────────────────────────────────────────────────────────

    def _build_llm_prompt(self, command: str) -> str:
        schema_example = {
            "description": "Enhance distant voices with heavy noise reduction",
            "noise_reduction": True,
            "noise_reduction_strength": 0.85,
            "preserve_weak_signals": True,
            "voice_boost": True,
            "voice_boost_db": 9.0,
            "source_separation": True,
            "spectral_subtraction": True,
            "wiener_filter": True,
            "transcribe": True,
            "whisper_model": "base",
            "dynamic_gain": True,
        }
        return f"""You are an audio processing configuration assistant.

Given this user command: "{command}"

Return ONLY valid JSON matching these exact keys. Do not add any text before or after the JSON.
Values that are not applicable should use their defaults.

JSON schema (with defaults):
{{
  "description": "string — 1-sentence summary of what will be done",
  "noise_reduction": true,
  "noise_reduction_strength": 0.75,  // 0.0–1.0: higher = more aggressive
  "preserve_weak_signals": true,
  "voice_boost": true,
  "voice_boost_db": 6.0,  // 0–24 dB
  "speech_freq_boost": true,
  "source_separation": true,
  "separation_model": "demucs",  // "demucs" | "none"
  "spectral_subtraction": true,
  "wiener_filter": true,
  "transcribe": true,
  "whisper_model": "base",  // "tiny"|"base"|"small"|"medium"|"large"
  "dynamic_gain": true,
  "dynamic_gain_ratio": 3.0,
  "vad_enabled": true,
  "vad_aggressiveness": 2  // 0–3
}}

Example output for "recover muffled voices behind a door with heavy noise":
{json.dumps(schema_example, indent=2)}

Now return JSON for: "{command}"
"""

    async def _parse_with_anthropic(self, command: str) -> PipelineConfig:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._anthropic_key)
        msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": self._build_llm_prompt(command)}],
        )
        raw = msg.content[0].text.strip()
        return self._json_to_config(raw, command)

    async def _parse_with_openai(self, command: str) -> PipelineConfig:
        import openai

        client = openai.AsyncOpenAI(api_key=self._openai_key)
        resp = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=512,
            messages=[{"role": "user", "content": self._build_llm_prompt(command)}],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        return self._json_to_config(raw, command)

    def _json_to_config(self, raw: str, command: str) -> PipelineConfig:
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(raw)
        # Merge with defaults (Pydantic ignores unknown fields)
        cfg = PipelineConfig(**{k: v for k, v in data.items() if hasattr(PipelineConfig, k) or k in PipelineConfig.__fields__})
        logger.info(f"LLM parsed config: {cfg.description}")
        return cfg

    # ── Rule-based fallback ───────────────────────────────────────────────────

    def _parse_with_rules(self, command: str) -> PipelineConfig:
        cmd_lower = command.lower()
        overrides: dict = {}
        descriptions = []

        for pattern, settings in KEYWORD_RULES.items():
            if re.search(pattern, cmd_lower):
                overrides.update(settings)

        # Build human description
        parts = []
        if overrides.get("voice_boost"):
            parts.append("enhance distant/faint voices")
        if overrides.get("noise_reduction"):
            strength = overrides.get("noise_reduction_strength", 0.75)
            parts.append(f"reduce background noise ({int(strength*100)}% strength)")
        if overrides.get("source_separation"):
            parts.append("separate audio sources")
        if overrides.get("transcribe") is False:
            parts.append("skip transcription")
        elif overrides.get("transcribe") or True:
            parts.append("transcribe speech")

        description = "; ".join(parts) if parts else "Enhance audio with default settings"
        overrides["description"] = description.capitalize()

        cfg = PipelineConfig(**overrides)
        logger.info(f"Rule-based config: {cfg.description}")
        return cfg
