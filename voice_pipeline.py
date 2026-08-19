"""
voice_pipeline.py
=================
Module: Audio Input (STT) & Audio Output (TTS)
Authors: [Student A] - Primary Owner
Date: August 2026

Description:
    Handles the full audio lifecycle for the Apex Home Assistant.
    - STT: Offline speech-to-text via Vosk + PyAudio streaming.
    - Wake Word: Continuous listening for "hey sophia" trigger.
    - Silence Detection: Auto-captures end-of-utterance.
    - TTS: Offline spoken feedback via pyttsx3.

Dependencies:
    pip install vosk pyaudio pyttsx3
"""

import json
import logging
import queue
import time
from typing import Optional, Callable

import pyaudio
from vosk import Model, KaldiRecognizer, SetLogLevel
import pyttsx3

# Suppress Vosk's verbose internal logging (keeps console clean)
SetLogLevel(-1)

logger = logging.getLogger("ApexLogger")


class VoicePipeline:
    """
    Manages offline speech recognition and text-to-speech synthesis.

    Attributes:
        sample_rate (int): Audio sampling rate in Hz (16kHz for Vosk).
        chunk_size (int): Audio buffer size per read cycle.
        wake_word (str): Trigger phrase to activate command capture.
        silence_threshold (float): Seconds of silence before cutting off.
    """

    # ── Configuration Constants ──────────────────────────────────────
    SAMPLE_RATE = 16000          # Vosk expects 16kHz mono PCM
    CHUNK_SIZE = 4000            # ~250ms of audio per buffer
    CHANNELS = 1                 # Mono microphone input
    SILENCE_LIMIT = 2.0          # Seconds of silence = end of command
    WAKE_WORD = "hey sophia"

    def __init__(self, model_path: str = "models/vosk-model-small-en-us-0.15"):
        """
        Initialize the voice pipeline with Vosk STT and pyttsx3 TTS.

        Args:
            model_path: Filesystem path to the extracted Vosk model directory.

        Raises:
            FileNotFoundError: If the Vosk model directory does not exist.
            OSError: If no microphone device is detected.
        """
        # ── 1. Initialize Vosk STT Model ─────────────────────────────
        try:
            logger.info(f"Loading Vosk model from: {model_path}")
            self._vosk_model = Model(model_path)
            self._recognizer = KaldiRecognizer(self._vosk_model, self.SAMPLE_RATE)
            logger.info("Vosk model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load Vosk model: {e}")
            raise FileNotFoundError(
                f"Vosk model not found at '{model_path}'. "
                f"Download from https://alphacephei.com/vosk/models "
                f"and extract to the /models directory."
            )

        # ── 2. Initialize PyAudio Stream ─────────────────────────────
        try:
            self._pyaudio = pyaudio.PyAudio()
            # Verify at least one input device exists
            input_device_count = 0
            for i in range(self._pyaudio.get_device_count()):
                if self._pyaudio.get_device_info_by_index(i)["maxInputChannels"] > 0:
                    input_device_count += 1
            if input_device_count == 0:
                raise OSError("No microphone detected on this system.")
            logger.info(f"PyAudio initialized. {input_device_count} input device(s) found.")
        except Exception as e:
            logger.critical(f"PyAudio initialization failed: {e}")
            raise

        # ── 3. Initialize pyttsx3 TTS Engine ─────────────────────────
        try:
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty("rate", 175)       # Words per minute
            self._tts_engine.setProperty("volume", 0.9)     # 0.0 to 1.0

            # Attempt to select a female voice for "Sophia" persona
            voices = self._tts_engine.getProperty("voices")
            for voice in voices:
                if "female" in voice.name.lower() or "zira" in voice.id.lower():
                    self._tts_engine.setProperty("voice", voice.id)
                    break
            logger.info("TTS engine (pyttsx3) initialized.")
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}. Voice feedback disabled.")
            self._tts_engine = None

        # ── 4. Internal State ────────────────────────────────────────
        self._audio_queue: queue.Queue = queue.Queue()
        self._is_listening = False

    # ══════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════════

    def listen_for_command(self, wake_word: Optional[str] = None) -> Optional[str]:
        """
        Blocking call that listens continuously for the wake word,
        then captures the full spoken command until silence is detected.

        Args:
            wake_word: Override the default wake phrase.

        Returns:
            The transcribed command string (without the wake word),
            or None if capture failed.
        """
        trigger = (wake_word or self.WAKE_WORD).lower()
        self._is_listening = True

        stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=self.CHANNELS,
            rate=self.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.CHUNK_SIZE,
        )

        logger.info(f"Microphone stream opened. Waiting for wake word: '{trigger}'")

        try:
            # ── Phase 1: Wait for Wake Word ──────────────────────────
            wake_detected = False
            partial_buffer = ""

            while self._is_listening and not wake_detected:
                raw_audio = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)

                if self._recognizer.AcceptWaveform(raw_audio):
                    # Full utterance recognized
                    result = json.loads(self._recognizer.Result())
                    text = result.get("text", "").lower().strip()
                    if trigger in text:
                        wake_detected = True
                        # Capture anything spoken AFTER the wake word
                        partial_buffer = text.split(trigger, 1)[-1].strip()
                        logger.info(f"Wake word detected in full result: '{text}'")
                else:
                    # Partial recognition (live streaming)
                    partial = json.loads(self._recognizer.PartialResult())
                    partial_text = partial.get("partial", "").lower()
                    if trigger in partial_text:
                        wake_detected = True
                        partial_buffer = partial_text.split(trigger, 1)[-1].strip()
                        logger.info(f"Wake word detected in partial: '{partial_text}'")

            if not wake_detected:
                return None

            # ── Phase 2: Capture Full Command Until Silence ──────────
            command_text = partial_buffer
            silence_start = time.time()
            last_partial = ""

            logger.info("Capturing command...")

            while self._is_listening:
                raw_audio = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)

                if self._recognizer.AcceptWaveform(raw_audio):
                    result = json.loads(self._recognizer.Result())
                    new_text = result.get("text", "").strip()
                    if new_text:
                        command_text += " " + new_text
                        command_text = command_text.strip()
                        silence_start = time.time()  # Reset silence timer
                        last_partial = ""
                else:
                    partial = json.loads(self._recognizer.PartialResult())
                    current_partial = partial.get("partial", "").strip()
                    if current_partial and current_partial != last_partial:
                        silence_start = time.time()  # User is still talking
                        last_partial = current_partial

                # Check if user has stopped speaking
                if time.time() - silence_start > self.SILENCE_LIMIT:
                    # Grab any remaining partial text
                    final_partial = json.loads(self._recognizer.PartialResult())
                    remaining = final_partial.get("partial", "").strip()
                    if remaining:
                        command_text += " " + remaining
                    break

            command_text = command_text.strip()

            if command_text:
                logger.info(f"Command captured: '{command_text}'")
                return command_text
            else:
                logger.warning("Wake word triggered but no command followed.")
                return None

        except IOError as e:
            logger.error(f"Audio stream read error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in listen_for_command: {e}")
            return None
        finally:
            stream.stop_stream()
            stream.close()

    def speak(self, text: str) -> bool:
        """
        Synthesize and play spoken audio feedback offline.

        Args:
            text: The confirmation message to speak aloud.

        Returns:
            True if speech was played successfully, False otherwise.
        """
        if not self._tts_engine:
            logger.warning("TTS engine unavailable. Skipping speech output.")
            return False

        try:
            logger.info(f"TTS Output: '{text}'")
            self._tts_engine.say(text)
            self._tts_engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"TTS playback failed: {e}")
            return False

    def stop(self):
        """Gracefully shut down all audio resources."""
        self._is_listening = False
        if self._tts_engine:
            self._tts_engine.stop()
        self._pyaudio.terminate()
        logger.info("Voice pipeline shut down.")

    # ══════════════════════════════════════════════════════════════════
    #  FALLBACK: Manual Text Input (for testing without microphone)
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def manual_input() -> str:
        """
        Fallback text input for environments without a working microphone.
        Satisfies the 'manual text backup' mentioned in the Basic rubric tier.
        """
        return input("[Manual Input] Enter command: ").strip()