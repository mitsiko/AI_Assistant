"""
ai_engine.py
============
Module: Local AI Brain — Intent Extraction & JSON Parsing
Authors: [Student B] - Primary Owner, [Your Name] - Refactored for Integration
Date: August 2026

Description:
    Routes transcribed text to a locally hosted Qwen 2.5 (1.5B) model
    via the Ollama daemon. Parses ambiguous natural language into
    structured JSON actions validated against a Pydantic schema.

    Key Features:
    - Enhanced few-shot prompting for implicit/ambiguous command resolution.
    - Pydantic v2 schema enforcement with automatic retry on failure.
    - Fallback to rule-based keyword matching if LLM is unreachable.
    - Command normalization for common speech recognition errors.
    - Full support for lights, thermostat, locks, and entertainment.
    - Handles "all devices", "except", and contextual commands.

Dependencies:
    pip install ollama pydantic
"""

import json
import logging
import re
import time
from typing import List, Optional, Dict, Any

import ollama
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("ApexLogger")

# ══════════════════════════════════════════════════════════════════════
#  PYDANTIC SCHEMAS (Strict JSON Validation)
# ══════════════════════════════════════════════════════════════════════

class SmartHomeAction(BaseModel):
    """
    Represents a single smart-home device action.
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
            "tv, speaker, entertainment_unit."
        )
    )
    value: Optional[int] = Field(
        default=None,
        description="Numeric value for temperature or volume. Null for toggles."
    )

class SmartHomeResponse(BaseModel):
    """
    Top-level response schema returned by the AI engine.
    """
    actions: List[SmartHomeAction] = Field(
        default_factory=list,
        description="One or more device actions to execute. Empty for clarification."
    )
    spoken_feedback: str = Field(
        ...,
        description=(
            "A natural, conversational confirmation sentence to be "
            "spoken aloud via TTS."
        )
    )

# ══════════════════════════════════════════════════════════════════════
#  AI ENGINE CLASS
# ══════════════════════════════════════════════════════════════════════

class AIEngine:
    """
    Local LLM intent parser powered by Ollama + Qwen 2.5.
    """
    
    MODEL_NAME = "qwen2.5:1.5b"
    MAX_RETRIES = 2
    INFERENCE_TIMEOUT = 10.0
    
    SYSTEM_PROMPT = """You are Sophia, a smart home assistant. Parse user commands into structured JSON ONLY.

CRITICAL RULES - READ CAREFULLY:

1. ONLY respond to DIRECT commands or CLEAR contextual requests. Do NOT add extra actions that weren't requested.

2. TEMPERATURE RULES:
   - "hot", "warm", "too hot" → ONLY change thermostat (decrease_temp or set_temp to 18-20)
   - "cold", "freezing", "chilly" → ONLY change thermostat (increase_temp or set_temp to 23-24)
   - Do NOT touch lights when user mentions temperature
   - Do NOT set thermostat when user only mentions TV or lights

3. LIGHT RULES:
   - Only control lights when user mentions "light", "dark", "bright", or specific room names
   - "turn off all lights" = ALL 3 lights off
   - "turn on the living room light" = ONLY living room

4. TV/ENTERTAINMENT RULES:
   - "watch TV", "play TV", "TV on" → ONLY turn on TV (action: turn_on, target: tv)
   - Do NOT turn on lights or change temperature for TV commands
   - "pause" alone → pause tv (if TV is on)
   - Do NOT use "entertainment_unit" - use "tv" or "speaker" instead

5. VALID TARGETS (use EXACTLY these):
   - Lights: "living_room_light", "kitchen_light", "bedroom_light"
   - Thermostat: "thermostat"
   - Doors: "front_door", "back_door"
   - Entertainment: "tv", "speaker"

6. VALID ACTIONS (use EXACTLY these):
   - Lights: turn_on, turn_off
   - Thermostat: set_temp, increase_temp, decrease_temp
   - Doors: lock, unlock
   - TV/Speaker: turn_on, turn_off, play, pause, stop, set_volume, increase_volume, decrease_volume

7. NEVER:
   - Use "turn_on" for thermostat (use set_temp or increase_temp)
   - Turn off lights when user says "hot" or "warm"
   - Turn on lights when user says "cold" or "freezing"
   - Add extra actions beyond what user requested
   - Invent new device names
   - Guess which device when user says "one light" or "a door" without specifics

8. AMBIGUITY HANDLING:
   - If user says "one light", "a light", "one door", "a door" WITHOUT specifying which one:
     * Return empty actions array
     * spoken_feedback should ask for clarification
     * Example: "Which light would you like me to control? Please specify living room, kitchen, or bedroom."
   - If user says "door" without "front" or "back":
     * Return empty actions array
     * spoken_feedback should ask which door
   - NEVER randomly pick a device when the user is ambiguous

9. If command is unclear, respond with empty actions and polite feedback asking for clarification.

OUTPUT FORMAT (JSON ONLY, no markdown):
{
  "actions": [
    {"action": "action_name", "target": "device_name", "value": null_or_number}
  ],
  "spoken_feedback": "brief confirmation"
}

10. VOLUME CONTROL:
    - "volume up" or "turn volume up" → increase_volume by 10 (NOT set to arbitrary value)
    - "volume down" or "turn volume down" → decrease_volume by 10
    - "volume to 80" or "set volume to 80%" → set_volume to 80
    - "too loud" → decrease_volume by 10
    - "can't hear" or "too quiet" → increase_volume by 10
    - Do NOT jump from 50% to 5% - use decrease_volume for relative changes

11. TEMPERATURE LIMITS:
    - Valid range is 10-35°C
    - If user asks for temperature above 35, set to 35 and mention the limit
    - If user asks for temperature below 10, set to 10 and mention the limit
    - "increase temperature to 36" means set_temp to 36, but simulator caps at 35
    - Use set_temp when user gives specific number
    - Use increase_temp/decrease_temp when user says "warmer" or "cooler" without number


EXAMPLES:

User: "It's hot in here"
Output: {"actions": [{"action": "decrease_temp", "target": "thermostat", "value": 2}], "spoken_feedback": "I've decreased the temperature by 2 degrees."}

User: "It's cold in here"
Output: {"actions": [{"action": "increase_temp", "target": "thermostat", "value": 2}], "spoken_feedback": "I've increased the temperature by 2 degrees."}

User: "Turn the volume up"
Output: {"actions": [{"action": "increase_volume", "target": "speaker", "value": 10}], "spoken_feedback": "Volume increased by 10%."}

User: "Turn the volume down"
Output: {"actions": [{"action": "decrease_volume", "target": "speaker", "value": 10}], "spoken_feedback": "Volume decreased by 10%."}

User: "Set volume to 75 percent"
Output: {"actions": [{"action": "set_volume", "target": "speaker", "value": 75}], "spoken_feedback": "Volume set to 75%."}

User: "Increase temperature to 36 degrees"
Output: {"actions": [{"action": "set_temp", "target": "thermostat", "value": 35}], "spoken_feedback": "I've set the temperature to 35 degrees, which is the maximum."}

User: "I'm going to sleep"
Output: {"actions": [{"action": "turn_off", "target": "living_room_light", "value": null}, {"action": "turn_off", "target": "kitchen_light", "value": null}, {"action": "turn_off", "target": "bedroom_light", "value": null}, {"action": "lock", "target": "front_door", "value": null}, {"action": "lock", "target": "back_door", "value": null}, {"action": "turn_off", "target": "tv", "value": null}], "spoken_feedback": "All lights off, doors locked, and TV turned off. Goodnight!"}"""

    def __init__(self):
        """Initialize the AI engine and verify Ollama connectivity."""
        self.model = self.MODEL_NAME
        self._verify_ollama_connection()

    def _detect_ambiguity(self, text: str) -> Optional[str]:
        """
        Detect ambiguous commands that need clarification.
        
        Returns:
            Clarification message if command is ambiguous, None otherwise.
        """
        text_lower = text.lower()
        
        # ── Check for "one" without specific target ─────────────────
        # Pattern: "one light", "a light", "one door", "a door" without room specification
        if re.search(r'\b(one|a|an)\s+(light|door)\b', text_lower):
            # Check if specific room is mentioned
            rooms = ["living", "kitchen", "bedroom", "front", "back"]
            if not any(room in text_lower for room in rooms):
                if "door" in text_lower:
                    return "Which door would you like me to control? Please specify front door or back door."
                elif "light" in text_lower:
                    return "Which light would you like me to control? Please specify living room, kitchen, or bedroom."
        
        # ── Check for "lights" (plural) with "one" ──────────────────
        if re.search(r'\b(one|a single)\s+(of\s+)?(the\s+)?lights?\b', text_lower):
            if not any(room in text_lower for room in rooms):
                return "Which light would you like me to control? Please specify living room, kitchen, or bedroom."
        
        # ── Check for "door" without front/back specification ───────
        if re.search(r'\b(unlock|lock|open|close)\s+(the\s+)?door\b', text_lower):
            if not ("front" in text_lower or "back" in text_lower or "all" in text_lower):
                return "Which door would you like me to control? Please specify front door or back door."
        
        # ── Check for "light" without room specification ────────────
        if re.search(r'\b(turn\s+)?(on|off)\s+(the\s+)?light\b', text_lower):
            if not any(room in text_lower for room in ["living", "kitchen", "bedroom", "all"]):
                return "Which light would you like me to control? Please specify living room, kitchen, or bedroom."
        
        return None
    
    def _verify_ollama_connection(self) -> None:
        """Ping the local Ollama daemon to confirm it is running."""
        try:
            models = ollama.list()
            available = [m.model for m in models.models]
            logger.info(f"Ollama connected. Available models: {available}")
            
            if not any(self.model in m for m in available):
                logger.warning(f"Model '{self.model}' not found. Pulling...")
                ollama.pull(self.model)
                logger.info(f"Model '{self.model}' pulled successfully.")
                
        except Exception as e:
            logger.critical(f"Cannot connect to Ollama daemon: {e}")
            raise ConnectionError(
                "Ollama daemon is not reachable. Start it with 'ollama serve'"
            )
    
    def parse_intent(self, user_text: str) -> Optional[SmartHomeResponse]:
        """
        Send transcribed text to Qwen and extract structured actions.
        
        Args:
            user_text: The transcribed voice command
            
        Returns:
            A validated SmartHomeResponse object, or None if parsing fails
        """
        if not user_text or not user_text.strip():
            logger.warning("Empty transcription received. Skipping AI parse.")
            return None
        
        # Normalize common speech recognition errors
        normalized_text = self._normalize_command(user_text)
        
        # Check for ambiguous commands that need clarification
        clarification = self._detect_ambiguity(normalized_text)
        if clarification:
            logger.info(f"Ambiguous command detected: '{normalized_text}'")
            return SmartHomeResponse(
                actions=[],
                spoken_feedback=clarification
            )
        
        start_time = time.time()
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": normalized_text}
        ]
        
        for attempt in range(1, self.MAX_RETRIES + 2):
            try:
                logger.info(f"LLM Inference Attempt {attempt}: '{normalized_text}'")
                
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    format="json",
                    options={
                        "temperature": 0.1,
                        "num_predict": 300,
                    }
                )
                
                raw_content = response["message"]["content"].strip()
                inference_time = time.time() - start_time
                logger.info(f"Raw LLM output ({inference_time:.2f}s): {raw_content}")
                
                raw_content = self._sanitize_json(raw_content)
                
                # Check for empty actions (unsupported command)
                parsed_data = json.loads(raw_content)
                if "actions" in parsed_data and len(parsed_data["actions"]) == 0:
                    # Unsupported command - return valid response with no actions
                    return SmartHomeResponse(
                        actions=[],
                        spoken_feedback=parsed_data.get("spoken_feedback", 
                            "I'm sorry, I can only control lights, temperature, doors, and entertainment devices.")
                    )
                
                parsed = SmartHomeResponse.model_validate_json(raw_content)
                logger.info(f"Validated {len(parsed.actions)} action(s) successfully.")
                return parsed
                
            except ValidationError as ve:
                logger.warning(f"Pydantic validation failed (attempt {attempt}): {ve}")
                messages.append({
                    "role": "assistant",
                    "content": raw_content if 'raw_content' in locals() else ""
                })
                messages.append({
                    "role": "user",
                    "content": "Your previous JSON was invalid. Fix the schema errors and output ONLY corrected JSON."
                })
                
            except Exception as e:
                logger.error(f"LLM inference error (attempt {attempt}): {e}")
                break
        
        # Fallback to keyword matching
        logger.warning("LLM parsing exhausted. Falling back to keyword matching.")
        return self._keyword_fallback(normalized_text)
    
    def _normalize_command(self, text: str) -> str:
        """
        Normalize common speech recognition errors and command variations.
        Uses word-boundary matching to avoid corrupting words.
        """
        text_lower = text.lower().strip()
        
        # Remove wake word if present
        text_lower = re.sub(r"^(hey|ok|okay)?\s*sophia[,\s]*", "", text_lower)

        # Handle "open/close" for lights and doors (common speech pattern)
        text_lower = re.sub(r'\bopen\s+(the\s+)?(.*?)\s+lights?\b', r'turn on \2light', text_lower)
        text_lower = re.sub(r'\bclose\s+(the\s+)?(.*?)\s+lights?\b', r'turn off \2light', text_lower)
        text_lower = re.sub(r'\bopen\s+(the\s+)?door\b', r'unlock door', text_lower)
        text_lower = re.sub(r'\bclose\s+(the\s+)?door\b', r'lock door', text_lower)
        
        # Common STT corrections (word-boundary only)
        corrections = {
            "boss": "pause",
            "pos": "pause",
            "paws": "pause",
            "lite": "light",
            "lites": "lights",
            "temp": "temperature",
            "open the light": "turn on the light",
            "open the lights": "turn on the lights",
            "open all the lights": "turn on all the lights",
            "close the light": "turn off the light",
            "close the lights": "turn off the lights",
            "open the door": "unlock the door",
            "close the door": "lock the door",
        }
        
        # Use word boundaries to avoid corrupting words like "temperature"
        for wrong, right in corrections.items():
            pattern = r'\b' + re.escape(wrong) + r'\b'
            if re.search(pattern, text_lower):
                text_lower = re.sub(pattern, right, text_lower)
                logger.info(f"Normalized: '{wrong}' -> '{right}'")
                break  # Only apply first correction to avoid over-correction
        
        return text_lower
    
    def _keyword_fallback(self, text: str) -> Optional[SmartHomeResponse]:
        """
        Deterministic keyword-based intent extraction.
        Used ONLY when the LLM is unreachable or returns invalid JSON.
        """
        text_lower = text.lower()
        actions: List[SmartHomeAction] = []
        
        # Helper function to add action if not already present
        def add_action(action, target, value=None):
            if not any(a.action == action and a.target == target for a in actions):
                actions.append(SmartHomeAction(action=action, target=target, value=value))
        
        # Lights
        all_lights = ["living_room_light", "kitchen_light", "bedroom_light"]
        
        # "all lights" handling
        if any(w in text_lower for w in ["all lights", "every light", "all the lights"]):
            if "except" in text_lower:
                # Find which light to exclude
                exclude = None
                if "living" in text_lower:
                    exclude = "living_room_light"
                elif "kitchen" in text_lower:
                    exclude = "kitchen_light"
                elif "bedroom" in text_lower:
                    exclude = "bedroom_light"
                
                for light in all_lights:
                    if light != exclude:
                        if "off" in text_lower:
                            add_action("turn_off", light)
                        else:
                            add_action("turn_on", light)
            else:
                for light in all_lights:
                    if "off" in text_lower:
                        add_action("turn_off", light)
                    else:
                        add_action("turn_on", light)
        
        # Individual lights
        light_mapping = {
            "living": "living_room_light",
            "kitchen": "kitchen_light",
            "bedroom": "bedroom_light"
        }
        for room, light_id in light_mapping.items():
            if room in text_lower and "light" in text_lower:
                if "off" in text_lower:
                    add_action("turn_off", light_id)
                elif "on" in text_lower:
                    add_action("turn_on", light_id)
        
        # Thermostat
        if "thermostat" in text_lower or "temperature" in text_lower:
            temp_match = re.search(r"(\d{1,2})\s*(?:degree|°|celsius|c)?", text_lower)
            if temp_match:
                temp = int(temp_match.group(1))
                if 10 <= temp <= 35:
                    add_action("set_temp", "thermostat", temp)
            elif any(w in text_lower for w in ["increase", "warmer", "up", "hot"]):
                add_action("increase_temp", "thermostat", 2)
            elif any(w in text_lower for w in ["decrease", "cooler", "down", "cold"]):
                add_action("decrease_temp", "thermostat", 2)
        
        # Contextual temperature
        if not actions and any(w in text_lower for w in ["freezing", "cold", "chilly"]):
            add_action("set_temp", "thermostat", 24)
        if not actions and any(w in text_lower for w in ["hot", "warm", "sweating"]):
            add_action("set_temp", "thermostat", 18)
        
        # Doors
        doors = ["front_door", "back_door"]
        if any(w in text_lower for w in ["all doors", "lock up", "lock everything"]):
            for door in doors:
                if "un" in text_lower:
                    add_action("unlock", door)
                else:
                    add_action("lock", door)
        else:
            if "front" in text_lower and "door" in text_lower:
                if "un" in text_lower:
                    add_action("unlock", "front_door")
                else:
                    add_action("lock", "front_door")
            if "back" in text_lower and "door" in text_lower:
                if "un" in text_lower:
                    add_action("unlock", "back_door")
                else:
                    add_action("lock", "back_door")
        
        # Entertainment
        if "tv" in text_lower or "television" in text_lower:
            if "pause" in text_lower:
                add_action("pause", "tv")
            elif "play" in text_lower or "resume" in text_lower:
                add_action("play", "tv")
            elif "stop" in text_lower:
                add_action("stop", "tv")
            elif "off" in text_lower:
                add_action("turn_off", "tv")
            elif "on" in text_lower:
                add_action("turn_on", "tv")
        
        # Volume
        vol_match = re.search(r"(\d{1,3})\s*(?:percent|%)", text_lower)
        if vol_match and any(w in text_lower for w in ["volume", "speaker", "sound"]):
            vol = int(vol_match.group(1))
            if 0 <= vol <= 100:
                add_action("set_volume", "speaker", vol)
        
        # Context: "going to sleep"
        if "sleep" in text_lower or "bedtime" in text_lower or "goodnight" in text_lower:
            for light in all_lights:
                add_action("turn_off", light)
            for door in doors:
                add_action("lock", door)
            add_action("turn_off", "tv")
            add_action("turn_off", "speaker")
        
        # Context: "going out"
        if "going out" in text_lower or "leaving" in text_lower:
            for door in doors:
                add_action("lock", door)
        
        if actions:
            feedback = self._generate_feedback(actions)
            logger.info(f"Keyword fallback generated {len(actions)} action(s).")
            return SmartHomeResponse(actions=actions, spoken_feedback=feedback)
        
        logger.warning("No matching commands found.")
        return None
    
    def _generate_feedback(self, actions: List[SmartHomeAction]) -> str:
        """Generate simple feedback for fallback responses."""
        if len(actions) == 1:
            action = actions[0]
            target = action.target.replace("_", " ")
            return f"I've {action.action.replace('_', ' ')} the {target}."
        else:
            return f"I've completed {len(actions)} actions as requested."
    
    @staticmethod
    def _sanitize_json(raw: str) -> str:
        """Clean common LLM output artifacts that break JSON parsing."""
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        raw = re.sub(r",\s*([}\]])", r"\1", raw)
        return raw.strip()
    
    def health_check(self) -> dict:
        """Returns a diagnostic dictionary."""
        try:
            models = ollama.list()
            target = next((m for m in models.models if self.model in m.model), None)
            return {
                "ollama_status": "running",
                "model": self.model,
                "model_size": target.size if target else "unknown",
            }
        except Exception as e:
            return {"ollama_status": "error", "detail": str(e)}