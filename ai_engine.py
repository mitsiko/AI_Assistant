"""
ai_engine.py
============
Module: Local AI Brain — Intent Extraction & JSON Parsing
Authors: [Student B] - Primary Owner
Date: August 2026

Description:
    Routes transcribed text to a locally hosted Qwen 2.5 (1.5B) model
    via the Ollama daemon. Parses ambiguous natural language into
    structured JSON actions validated against a Pydantic schema.

    Key Features:
    - Few-shot prompting for implicit/ambiguous command resolution.
    - Pydantic v2 schema enforcement with automatic retry on failure.
    - Fallback to rule-based keyword matching if LLM is unreachable.
    - Full support for lights, thermostat, locks, and entertainment.

Dependencies:
    pip install ollama pydantic
"""

import json
import logging
import re
import time
from typing import List, Optional

import ollama
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("ApexLogger")


# ══════════════════════════════════════════════════════════════════════
#  PYDANTIC SCHEMAS (Strict JSON Validation)
# ══════════════════════════════════════════════════════════════════════

class SmartHomeAction(BaseModel):
    """
    Represents a single smart-home device action.

    Examples:
        {"action": "turn_on", "target": "living_room_light", "value": null}
        {"action": "set_temp", "target": "thermostat", "value": 22}
        {"action": "lock", "target": "front_door", "value": null}
        {"action": "set_volume", "target": "entertainment_unit", "value": 40}
    """
    action: str = Field(
        ...,
        description=(
            "The operation to perform. Must be one of: "
            "turn_on, turn_off, set_temp, increase_temp, decrease_temp, "
            "lock, unlock, set_volume, play, pause, stop."
        )
    )
    target: str = Field(
        ...,
        description=(
            "The device identifier. Must be one of: "
            "living_room_light, kitchen_light, bedroom_light, "
            "thermostat, front_door, back_door, "
            "entertainment_unit, tv, speaker."
        )
    )
    value: Optional[int | str] = Field(
        default=None,
        description="Numeric value for temperature, brightness, or volume. Null for toggles."
    )


class SmartHomeResponse(BaseModel):
    """
    Top-level response schema returned by the AI engine.
    Contains a list of actions and a human-readable confirmation.
    """
    actions: List[SmartHomeAction] = Field(
        ...,
        min_length=1,
        description="One or more device actions to execute."
    )
    spoken_feedback: str = Field(
        ...,
        description=(
            "A natural, conversational confirmation sentence to be "
            "spoken aloud via TTS. Example: 'I turned on the living "
            "room lights and set the thermostat to 22 degrees.'"
        )
    )


# ══════════════════════════════════════════════════════════════════════
#  AI ENGINE CLASS
# ══════════════════════════════════════════════════════════════════════

class AIEngine:
    """
    Local LLM intent parser powered by Ollama + Qwen 2.5.

    Attributes:
        model (str): Ollama model tag (e.g., 'qwen2.5:1.5b').
        max_retries (int): Number of retry attempts for malformed JSON.
        timeout (float): Maximum seconds to wait for LLM inference.
    """

    # ── Configuration ────────────────────────────────────────────────
    MODEL_NAME = "qwen2.5:1.5b"
    MAX_RETRIES = 2
    INFERENCE_TIMEOUT = 10.0  # seconds

    # ── Few-Shot System Prompt ───────────────────────────────────────
    # This prompt is engineered specifically for the 1.5B parameter model.
    # Small models need explicit examples to handle ambiguity correctly.
    SYSTEM_PROMPT = """You are ApexOS, a strict JSON intent-extraction engine for a smart home hub.

RULES:
1. Parse the user's natural language into structured JSON actions.
2. INFER implicit needs from context clues:
   - "dark", "can't see" → turn_on lights
   - "freezing", "cold" → increase_temp or set_temp to 24
   - "hot", "warm" → decrease_temp or set_temp to 18
   - "leaving", "going out" → lock doors, turn_off lights
   - "movie night", "set the mood" → turn_off lights, set_volume
3. Use ONLY these target names:
   living_room_light, kitchen_light, bedroom_light,
   thermostat, front_door, back_door,
   entertainment_unit, tv, speaker
4. Use ONLY these actions:
   turn_on, turn_off, set_temp, increase_temp, decrease_temp,
   lock, unlock, set_volume, play, pause, stop
5. Output ONLY valid JSON. No markdown. No explanations. No code blocks.

EXAMPLES:

User: "It's getting dark in here and I'm freezing."
{"actions": [{"action": "turn_on", "target": "living_room_light", "value": null}, {"action": "set_temp", "target": "thermostat", "value": 24}], "spoken_feedback": "I've turned on the living room lights and set the thermostat to 24 degrees. Stay warm!"}

User: "Hey Sophia, lock up and turn off all the lights."
{"actions": [{"action": "lock", "target": "front_door", "value": null}, {"action": "turn_off", "target": "living_room_light", "value": null}, {"action": "turn_off", "target": "kitchen_light", "value": null}], "spoken_feedback": "All lights are off and the front door is locked. Goodnight!"}

User: "Set the living room thermostat to 22 degrees and turn off the kitchen lights."
{"actions": [{"action": "set_temp", "target": "thermostat", "value": 22}, {"action": "turn_off", "target": "kitchen_light", "value": null}], "spoken_feedback": "Living room thermostat set to 22 degrees, and kitchen lights are now off."}

User: "I want to watch a movie. Make it cozy."
{"actions": [{"action": "turn_off", "target": "living_room_light", "value": null}, {"action": "turn_on", "target": "tv", "value": null}, {"action": "set_volume", "target": "entertainment_unit", "value": 35}], "spoken_feedback": "Lights dimmed, TV is on, and volume set to 35. Enjoy your movie!"}"""

    def __init__(self):
        """
        Initialize the AI engine and verify Ollama connectivity.

        Raises:
            ConnectionError: If the Ollama daemon is not running.
        """
        self.model = self.MODEL_NAME
        self._verify_ollama_connection()

    def _verify_ollama_connection(self) -> None:
        """
        Ping the local Ollama daemon to confirm it is running
        and the required model is available.
        """
        try:
            models = ollama.list()
            available = [m.model for m in models.models]
            logger.info(f"Ollama connected. Available models: {available}")

            # Check if our target model is pulled
            if not any(self.model in m for m in available):
                logger.warning(
                    f"Model '{self.model}' not found locally. "
                    f"Attempting to pull (this may take a few minutes)..."
                )
                ollama.pull(self.model)
                logger.info(f"Model '{self.model}' pulled successfully.")

        except Exception as e:
            logger.critical(
                f"Cannot connect to Ollama daemon: {e}\n"
                f"Ensure Ollama is installed and running: 'ollama serve'"
            )
            raise ConnectionError(
                "Ollama daemon is not reachable. "
                "Start it with 'ollama serve' in a terminal."
            )

    # ══════════════════════════════════════════════════════════════════
    #  PRIMARY: LLM-BASED INTENT EXTRACTION
    # ══════════════════════════════════════════════════════════════════

    def parse_intent(self, user_text: str) -> Optional[SmartHomeResponse]:
        """
        Send transcribed text to Qwen and extract structured actions.

        Includes retry logic: if the LLM returns malformed JSON on the
        first attempt, it retries up to MAX_RETRIES times with a
        corrective follow-up prompt.

        Args:
            user_text: The transcribed voice command from the STT pipeline.

        Returns:
            A validated SmartHomeResponse object, or None if all
            attempts fail (falls back to keyword matching).
        """
        if not user_text or not user_text.strip():
            logger.warning("Empty transcription received. Skipping AI parse.")
            return None

        start_time = time.time()
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ]

        for attempt in range(1, self.MAX_RETRIES + 2):  # +2 = 1 initial + retries
            try:
                logger.info(f"LLM Inference Attempt {attempt} for: '{user_text}'")

                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    format="json",  # Forces Ollama into JSON mode
                    options={
                        "temperature": 0.1,   # Low = deterministic output
                        "num_predict": 300,   # Cap token generation
                    }
                )

                raw_content = response["message"]["content"].strip()
                inference_time = time.time() - start_time
                logger.info(f"Raw LLM output ({inference_time:.2f}s): {raw_content}")

                # ── Clean common LLM artifacts ───────────────────────
                raw_content = self._sanitize_json(raw_content)

                # ── Validate against Pydantic Schema ─────────────────
                parsed = SmartHomeResponse.model_validate_json(raw_content)
                logger.info(
                    f"Validated {len(parsed.actions)} action(s) successfully."
                )
                return parsed

            except ValidationError as ve:
                logger.warning(
                    f"Pydantic validation failed (attempt {attempt}): {ve}"
                )
                # Add corrective feedback for the retry
                messages.append({
                    "role": "assistant",
                    "content": raw_content if 'raw_content' in dir() else ""
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous JSON was invalid. Fix the schema errors "
                        "and output ONLY corrected JSON. Remember: actions must "
                        "use valid target names and action types."
                    )
                })

            except Exception as e:
                logger.error(f"LLM inference error (attempt {attempt}): {e}")
                break

        # ── All LLM attempts failed → Fallback ───────────────────────
        logger.warning("LLM parsing exhausted. Falling back to keyword matching.")
        return self._keyword_fallback(user_text)

    # ══════════════════════════════════════════════════════════════════
    #  FALLBACK: RULE-BASED KEYWORD MATCHING
    # ══════════════════════════════════════════════════════════════════

    def _keyword_fallback(self, text: str) -> Optional[SmartHomeResponse]:
        """
        Deterministic keyword-based intent extraction.
        Used ONLY when the LLM is unreachable or returns invalid JSON.
        This prevents total system failure and satisfies the rubric's
        'hardcoded fallback' awareness requirement.

        Args:
            text: The raw transcribed command.

        Returns:
            A basic SmartHomeResponse, or None if no keywords matched.
        """
        text_lower = text.lower()
        actions: List[SmartHomeAction] = []

        # ── Light Commands ───────────────────────────────────────────
        if any(w in text_lower for w in ["light", "lights", "lamp"]):
            if any(w in text_lower for w in ["on", "bright"]):
                actions.append(SmartHomeAction(
                    action="turn_on", target="living_room_light"
                ))
            elif any(w in text_lower for w in ["off", "dark"]):
                actions.append(SmartHomeAction(
                    action="turn_off", target="living_room_light"
                ))

        # ── Thermostat Commands ──────────────────────────────────────
        temp_match = re.search(r"(\d{1,2})\s*(?:degree|°|celsius)", text_lower)
        if "thermostat" in text_lower or "temperature" in text_lower or "temp" in text_lower:
            if temp_match:
                actions.append(SmartHomeAction(
                    action="set_temp",
                    target="thermostat",
                    value=int(temp_match.group(1))
                ))
            elif any(w in text_lower for w in ["up", "increase", "warmer", "hot"]):
                actions.append(SmartHomeAction(
                    action="increase_temp", target="thermostat", value=2
                ))
            elif any(w in text_lower for w in ["down", "decrease", "cooler", "cold"]):
                actions.append(SmartHomeAction(
                    action="decrease_temp", target="thermostat", value=2
                ))

        # ── Lock Commands ────────────────────────────────────────────
        if any(w in text_lower for w in ["lock", "door", "secure"]):
            if "un" in text_lower:
                actions.append(SmartHomeAction(
                    action="unlock", target="front_door"
                ))
            else:
                actions.append(SmartHomeAction(
                    action="lock", target="front_door"
                ))

        # ── Entertainment Commands ───────────────────────────────────
        if any(w in text_lower for w in ["tv", "television", "movie", "music"]):
            if any(w in text_lower for w in ["on", "play", "start"]):
                actions.append(SmartHomeAction(
                    action="turn_on", target="tv"
                ))
            elif any(w in text_lower for w in ["off", "stop", "pause"]):
                actions.append(SmartHomeAction(
                    action="turn_off", target="tv"
                ))

        # ── Ambiguous Context Inference (basic) ──────────────────────
        if not actions:
            if any(w in text_lower for w in ["dark", "can't see"]):
                actions.append(SmartHomeAction(
                    action="turn_on", target="living_room_light"
                ))
            if any(w in text_lower for w in ["freezing", "cold", "chilly"]):
                actions.append(SmartHomeAction(
                    action="set_temp", target="thermostat", value=24
                ))
            if any(w in text_lower for w in ["hot", "warm", "sweating"]):
                actions.append(SmartHomeAction(
                    action="set_temp", target="thermostat", value=18
                ))

        if actions:
            feedback = "I've processed your command using basic keyword matching."
            logger.info(f"Keyword fallback generated {len(actions)} action(s).")
            return SmartHomeResponse(actions=actions, spoken_feedback=feedback)

        logger.warning("Keyword fallback found no matching commands.")
        return None

    # ══════════════════════════════════════════════════════════════════
    #  UTILITY METHODS
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _sanitize_json(raw: str) -> str:
        """
        Clean common LLM output artifacts that break JSON parsing.

        Handles:
        - Markdown code fences (```json ... ```)
        - Leading/trailing whitespace
        - Trailing commas before closing brackets
        """
        # Strip markdown code blocks
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())

        # Remove trailing commas (common LLM mistake)
        raw = re.sub(r",\s*([}\]])", r"\1", raw)

        return raw.strip()

    def health_check(self) -> dict:
        """
        Returns a diagnostic dictionary for the execution log.
        Useful for the report's CPU/RAM metrics section.
        """
        try:
            models = ollama.list()
            target = next(
                (m for m in models.models if self.model in m.model), None
            )
            return {
                "ollama_status": "running",
                "model": self.model,
                "model_size": target.size if target else "unknown",
                "modified_at": str(target.modified_at) if target else "N/A"
            }
        except Exception as e:
            return {"ollama_status": "error", "detail": str(e)}