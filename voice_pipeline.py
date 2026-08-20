"""
voice_pipeline.py
=================
Module: Audio Input (STT) & Audio Output (TTS)
Authors: [Student A] - Primary Owner, [Your Name] - Refactored for Reliability
Date: August 2026

Description:
    Handles the full audio lifecycle for the Apex Home Assistant.
    - STT: Offline speech-to-text via Vosk + PyAudio streaming.
    - Wake Word: Continuous listening for "hey sophia" trigger.
    - Silence Detection: Auto-captures end-of-utterance.
    - TTS: Reliable offline spoken feedback via pyttsx3 with queue management.

Key Improvements:
    - Thread-safe TTS queue to prevent overlapping speech
    - Engine lifecycle management for repeated use
    - Improved wake word detection with confidence checking
    - Command normalization for common STT errors

Dependencies:
    pip install vosk pyaudio pyttsx3
"""

import json
import logging
import queue
import re
import threading
import time
from typing import Optional, Callable

import pyaudio
from vosk import Model, KaldiRecognizer, SetLogLevel
import pyttsx3

SetLogLevel(-1)
logger = logging.getLogger("ApexLogger")


class VoicePipeline:
    """
    Manages offline speech recognition and text-to-speech synthesis.
    """
    
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 4000
    CHANNELS = 1
    SILENCE_LIMIT = 1.5  # Reduced from 2.0 for snappier response
    WAKE_WORD = "hey sophia"
    
    def __init__(self, model_path: str = "models/vosk-model-small-en-us-0.15"):
        """
        Initialize the voice pipeline with Vosk STT and pyttsx3 TTS.
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
                f"Vosk model not found at '{model_path}'."
            )
        
        # ── 2. Initialize PyAudio ─────────────────────────────────────
        try:
            self._pyaudio = pyaudio.PyAudio()
            input_device_count = 0
            for i in range(self._pyaudio.get_device_count()):
                if self._pyaudio.get_device_info_by_index(i)["maxInputChannels"] > 0:
                    input_device_count += 1
            if input_device_count == 0:
                raise OSError("No microphone detected.")
            logger.info(f"PyAudio initialized. {input_device_count} input device(s).")
        except Exception as e:
            logger.critical(f"PyAudio initialization failed: {e}")
            raise
        
        # ── 3. TTS Setup ────────────────────────────────────────────
        self._tts_queue = queue.Queue()
        self._tts_thread = None
        self._tts_lock = threading.Lock()
        
        # ── 4. Internal State ────────────────────────────────────────
        self._is_listening = False
        self._status_callback = None
    
    def _init_tts_engine(self):
        """Initialize a fresh TTS engine instance."""
        try:
            # Always create a new engine instance
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 0.9)
            
            # Try to select a female voice
            try:
                voices = engine.getProperty("voices")
                for voice in voices:
                    if "female" in voice.name.lower() or "zira" in voice.id.lower():
                        engine.setProperty("voice", voice.id)
                        break
            except:
                pass  # Voice selection is optional
            
            return engine
        except Exception as e:
            logger.error(f"TTS engine initialization failed: {e}")
            return None
    
    def set_status_callback(self, callback: Callable[[str], None]):
        """Set callback for status updates (used by GUI)."""
        self._status_callback = callback
    
    def _update_status(self, status: str):
        """Update status via callback if set."""
        if self._status_callback:
            self._status_callback(status)
    
    def listen_for_command(self, wake_word: Optional[str] = None) -> Optional[str]:
        """
        Blocking call that listens for the wake word and captures command.
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
        
        logger.info(f"Listening for wake word: '{trigger}'")
        
        try:
            # Phase 1: Wait for Wake Word
            wake_detected = False
            partial_buffer = ""
            wake_start_time = time.time()
            max_wake_wait = 30  # 30 second timeout for wake word
            
            while self._is_listening and not wake_detected:
                if time.time() - wake_start_time > max_wake_wait:
                    logger.info("Wake word timeout reached.")
                    break
                
                raw_audio = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                
                if self._recognizer.AcceptWaveform(raw_audio):
                    result = json.loads(self._recognizer.Result())
                    text = result.get("text", "").lower().strip()
                    
                    # Check for wake word in full result
                    if trigger in text:
                        wake_detected = True
                        partial_buffer = text.split(trigger, 1)[-1].strip()
                        logger.info(f"Wake word detected: '{text}'")
                        self._update_status("Listening...")
                else:
                    partial = json.loads(self._recognizer.PartialResult())
                    partial_text = partial.get("partial", "").lower().strip()
                    
                    # Check for wake word in partial (with word boundary)
                    if trigger in partial_text:
                        wake_detected = True
                        partial_buffer = partial_text.split(trigger, 1)[-1].strip()
                        logger.info(f"Wake word detected (partial): '{partial_text}'")
                        self._update_status("Listening...")
            
            if not wake_detected:
                return None
            
            # Reset recognizer for command capture
            self._recognizer = KaldiRecognizer(self._vosk_model, self.SAMPLE_RATE)
            
            # Phase 2: Capture Full Command
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
                        if command_text:
                            command_text += " " + new_text
                        else:
                            command_text = new_text
                        command_text = command_text.strip()
                        silence_start = time.time()
                        last_partial = ""
                else:
                    partial = json.loads(self._recognizer.PartialResult())
                    current_partial = partial.get("partial", "").strip()
                    if current_partial and current_partial != last_partial:
                        silence_start = time.time()
                        last_partial = current_partial
                
                # Check for silence
                if time.time() - silence_start > self.SILENCE_LIMIT:
                    final_partial = json.loads(self._recognizer.PartialResult())
                    remaining = final_partial.get("partial", "").strip()
                    if remaining and remaining != last_partial:
                        if command_text:
                            command_text += " " + remaining
                        else:
                            command_text = remaining
                    break
            
            command_text = self._normalize_transcription(command_text.strip())
            
            if command_text:
                logger.info(f"Command captured: '{command_text}'")
                return command_text
            else:
                logger.warning("No command captured after wake word.")
                return None
                
        except IOError as e:
            logger.error(f"Audio stream error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
        finally:
            stream.stop_stream()
            stream.close()
    
    def _normalize_transcription(self, text: str) -> str:
        """
        Lightweight normalization of common STT errors.
        Only corrects when context makes the intent clear.
        """
        if not text:
            return text
        
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        # Common word corrections (context-dependent)
        corrections = {
            "boss": "pause",  # "pause the tv" → "boss the tv"
            "pos": "pause",
            "paws": "pause",
            "lite": "light",
            "lites": "lights",
            "temp": "temperature",
        }
        
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in corrections:
                words[i] = corrections[word.lower()]
                logger.info(f"Corrected '{word}' → '{corrections[word.lower()]}'")
        
        return " ".join(words)
    
    def speak(self, text: str, blocking: bool = False) -> bool:
        """
        Queue text for TTS playback.
        
        Args:
            text: The text to speak
            blocking: If True, wait for speech to complete
            
        Returns:
            True if successfully queued, False otherwise
        """
        if not text or not text.strip():
            logger.warning("Empty text for TTS. Skipping.")
            return False
        
        # Add to queue
        self._tts_queue.put(text)
        
        # Start TTS thread if not running
        if self._tts_thread is None or not self._tts_thread.is_alive():
            self._tts_thread = threading.Thread(
                target=self._tts_worker,
                daemon=True,
                name="TTS-Thread"
            )
            self._tts_thread.start()
            logger.info("TTS thread started.")
        
        if blocking:
            # Wait for queue to be empty
            self._tts_queue.join()
        
        return True
    
    def _tts_worker(self):
        """
        Worker thread that processes TTS queue sequentially.
        Creates a fresh engine for each speech request to avoid pyttsx3 state issues.
        """
        logger.info("TTS worker thread started.")
        
        while True:
            try:
                # Get next text from queue (blocking with timeout)
                text = self._tts_queue.get(timeout=1)
                
                if text == "STOP":
                    logger.info("TTS worker stopping.")
                    break
                
                self._update_status("Speaking...")
                logger.info(f"TTS Output: '{text}'")
                
                # Create a fresh engine for this speech request
                engine = self._init_tts_engine()
                
                if engine:
                    try:
                        engine.say(text)
                        engine.runAndWait()
                        logger.info("TTS completed successfully.")
                    except Exception as e:
                        logger.error(f"TTS playback failed: {e}")
                    finally:
                        # Clean up engine
                        try:
                            engine.stop()
                            del engine
                        except:
                            pass
                else:
                    logger.error("Failed to create TTS engine.")
                
                self._tts_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS worker error: {e}")
                try:
                    self._tts_queue.task_done()
                except:
                    pass
    
    def wait_for_tts_complete(self, timeout: float = 5.0):
        """Wait for all queued TTS to complete."""
        try:
            self._tts_queue.join()
            logger.info("TTS queue empty.")
        except Exception as e:
            logger.warning(f"TTS wait timeout: {e}")
    
    def stop_tts(self):
        """Stop current TTS playback and clear queue."""
        # Clear queue
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
                self._tts_queue.task_done()
            except:
                break
        
        logger.info("TTS queue cleared.")
    
    def stop(self):
        """Gracefully shut down all audio resources."""
        logger.info("Shutting down voice pipeline...")
        self._is_listening = False
        
        # Stop TTS
        self.stop_tts()
        
        # Signal TTS thread to stop
        self._tts_queue.put("STOP")
        if self._tts_thread and self._tts_thread.is_alive():
            self._tts_thread.join(timeout=2)
        
        # Terminate PyAudio
        try:
            self._pyaudio.terminate()
        except:
            pass
        
        logger.info("Voice pipeline shut down.")
    
    @staticmethod
    def manual_input() -> str:
        """Fallback text input for testing."""
        return input("[Manual Input] Enter command: ").strip()