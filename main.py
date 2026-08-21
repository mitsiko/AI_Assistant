"""
main.py
=======
Module: Application Controller & GUI
Author: [Your Name] - Partner 2 (Interface & Integration Engineer)
Date: August 2026

Description:
    Main entry point for the Apex Home Virtual Assistant.
    Coordinates the voice pipeline, AI engine, and home simulator.
    Provides the Tkinter GUI dashboard.

Workflow:
    Voice → Speech-to-Text → AI Intent Processing → 
    Structured Command → Home Simulator → GUI Update → TTS
"""

import json
import logging
import queue
import re
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from typing import Optional, Dict, Any

from voice_pipeline import VoicePipeline
from ai_engine import AIEngine
from home_simulator import HomeSimulator

# ══════════════════════════════════════════════════════════════════════
#  LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════

def setup_logging():
    """Configure logging to file and console."""
    logger = logging.getLogger("ApexLogger")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler("assistant_execution.log")
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()

# ══════════════════════════════════════════════════════════════════════
#  GUI COLORS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════

BG_COLOR = "#f4eddb"
BORDER_COLOR = "#000000"
ACTIVE_COLOR = "#eb3d1b"
INACTIVE_COLOR = "#767062"
FONT_FAMILY = "MS Gothic"  # Monospace

# ══════════════════════════════════════════════════════════════════════
#  CUSTOM WIDGETS
# ══════════════════════════════════════════════════════════════════════

class StatusSwitch(tk.Canvas):
    """Custom switch-style widget showing ON/OFF or LOCKED/UNLOCKED state."""
    
    def __init__(self, parent, on_text="ON", off_text="OFF", 
                 initial_state=False, command=None, **kwargs):
        super().__init__(parent, height=30, width=100, 
                        bg=BG_COLOR, highlightthickness=0, **kwargs)
        
        self.on_text = on_text
        self.off_text = off_text
        self.is_on = initial_state
        self.command = command
        
        self._draw()
        self.bind("<Button-1>", self._on_click)
    
    def _draw(self):
        """Draw the switch based on current state."""
        self.delete("all")
        
        # Background
        color = ACTIVE_COLOR if self.is_on else INACTIVE_COLOR
        self.create_rectangle(0, 5, 100, 25, fill=color, outline="")
        
        # Text
        text = self.on_text if self.is_on else self.off_text
        self.create_text(50, 15, text=text, fill="white", 
                        font=(FONT_FAMILY, 10, "bold"))
    
    def set_state(self, state: bool):
        """Update switch state."""
        self.is_on = state
        self._draw()
    
    def get_state(self) -> bool:
        """Get current switch state."""
        return self.is_on
    
    def _on_click(self, event):
        """Handle click for manual testing."""
        if self.command:
            self.command()
    
    def toggle(self):
        """Toggle switch state."""
        self.is_on = not self.is_on
        self._draw()
        return self.is_on


class BentoCard(tk.Frame):
    """Base bento-style card with border."""
    
    def __init__(self, parent, title: str, **kwargs):
        super().__init__(parent, bg=BG_COLOR, **kwargs)
        
        # Card frame with border
        self.card_frame = tk.Frame(
            self, bg=BG_COLOR, 
            highlightbackground=BORDER_COLOR,
            highlightcolor=BORDER_COLOR,
            highlightthickness=1,
            bd=0
        )
        self.card_frame.pack(fill="both", expand=True, padx=2.5, pady=2.5)
        
        # Title
        self.title_label = tk.Label(
            self.card_frame,
            text=title,
            font=(FONT_FAMILY, 12, "bold"),
            bg=BORDER_COLOR,
            fg=BG_COLOR,
            anchor="w"
        )
        self.title_label.pack(pady=(10, 5), padx=10, anchor="w")
        
        # Content area
        self.content_frame = tk.Frame(self.card_frame, bg=BG_COLOR)
        self.content_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))


# ══════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION CLASS
# ══════════════════════════════════════════════════════════════════════

class ApexAssistantApp:
    """Main application controller and GUI."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Apex Home Assistant - Sophia")
        self.root.geometry("800x600")
        self.root.configure(bg=BG_COLOR)
        self.root.minsize(600, 500)
        
        # ── Initialize Components ───────────────────────────────────
        logger.info("Application started")
        logger.info("Initializing components...")
        
        try:
            self.simulator = HomeSimulator()
            logger.info("HomeSimulator initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize HomeSimulator: {e}")
            messagebox.showerror("Error", f"Failed to initialize simulator: {e}")
            raise
        
        try:
            self.ai_engine = AIEngine()
            logger.info("AIEngine initialized.")
        except ConnectionError as e:
            logger.error(f"Failed to initialize AIEngine: {e}")
            self.ai_engine = None
            messagebox.showwarning(
                "Ollama Not Available",
                "Ollama daemon is not running. Please start it with 'ollama serve'.\n\n"
                "The application will continue with limited functionality."
            )
        except Exception as e:
            logger.error(f"Unexpected AIEngine error: {e}")
            self.ai_engine = None
        
        try:
            self.voice_pipeline = VoicePipeline()
            logger.info("VoicePipeline initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize VoicePipeline: {e}")
            self.voice_pipeline = None
            messagebox.showwarning(
                "Audio Not Available",
                f"Voice pipeline failed to initialize: {e}\n\n"
                "The application will continue with manual testing only."
            )
        
        # ── Threading Setup ─────────────────────────────────────────
        self.command_queue = queue.Queue()
        self.is_processing = False
        self.voice_thread = None
        
        # ── Build GUI ───────────────────────────────────────────────
        self._build_gui()
        
        # ── Start Voice Processing Thread ───────────────────────────
        if self.voice_pipeline:
            self.voice_pipeline.set_status_callback(self._update_status)
            self._start_voice_thread()
        
        # ── Bind Cleanup ────────────────────────────────────────────
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        logger.info("Application ready.")
        self._update_status("Ready")
    
    # ══════════════════════════════════════════════════════════════════
    #  GUI BUILDING
    # ══════════════════════════════════════════════════════════════════
    
    def _build_gui(self):
        """Build the complete GUI."""
        
        # ── Scrollable Container ────────────────────────────────────
        self.canvas = tk.Canvas(self.root, bg=BG_COLOR, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(
            self.root, orient="vertical", command=self.canvas.yview
        )
        self.scrollable_frame = tk.Frame(self.canvas, bg=BG_COLOR)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # ── Top Row: Status, Activity, Thermostat ───────────────────
        top_row = tk.Frame(self.scrollable_frame, bg=BG_COLOR)
        top_row.pack(fill="x", padx=10, pady=(10, 5))
        
        # Assistant Status Card
        self.status_card = BentoCard(top_row, "ASSISTANT STATUS")
        self.status_card.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Sophia icon (placeholder circle)
        self.icon_canvas = tk.Canvas(
            self.status_card.content_frame,
            width=60, height=60,
            bg=BG_COLOR, highlightthickness=0
        )
        self.icon_canvas.pack(pady=(0, 5))
        self.icon_canvas.create_oval(5, 5, 55, 55, fill=ACTIVE_COLOR, outline="")
        self.icon_canvas.create_text(30, 30, text="S", fill="white", 
                                    font=(FONT_FAMILY, 24, "bold"))
        
        self.assistant_name = tk.Label(
            self.status_card.content_frame,
            text="Sophia",
            font=(FONT_FAMILY, 14, "bold"),
            bg=BG_COLOR, fg=BORDER_COLOR
        )
        self.assistant_name.pack()
        
        self.status_label = tk.Label(
            self.status_card.content_frame,
            text="Ready",
            font=(FONT_FAMILY, 11),
            bg=BG_COLOR, fg=BORDER_COLOR
        )
        self.status_label.pack()
        
        self.instruction_label = tk.Label(
            self.status_card.content_frame,
            text='Say "Hey Sophia"',
            font=(FONT_FAMILY, 9),
            bg=BG_COLOR, fg=INACTIVE_COLOR
        )
        self.instruction_label.pack(pady=(0, 5))
        
        # Latest Activity Card
        self.activity_card = BentoCard(top_row, "LATEST ACTIVITY")
        self.activity_card.pack(side="left", fill="both", expand=True, padx=5)
        
        self.you_said_label = tk.Label(
            self.activity_card.content_frame,
            text='YOU SAID:\n"..."',
            font=(FONT_FAMILY, 9),
            bg=BG_COLOR, fg=BORDER_COLOR,
            justify="left", wraplength=200
        )
        self.you_said_label.pack(anchor="w", pady=(0, 5))
        
        self.understood_label = tk.Label(
            self.activity_card.content_frame,
            text="UNDERSTOOD:\n...",
            font=(FONT_FAMILY, 9),
            bg=BG_COLOR, fg=BORDER_COLOR,
            justify="left", wraplength=200
        )
        self.understood_label.pack(anchor="w", pady=(0, 5))
        
        self.sophia_label = tk.Label(
            self.activity_card.content_frame,
            text='SOPHIA:\n"..."',
            font=(FONT_FAMILY, 9),
            bg=BG_COLOR, fg=BORDER_COLOR,
            justify="left", wraplength=200
        )
        self.sophia_label.pack(anchor="w")
        
        # Thermostat Card
        self.thermostat_card = BentoCard(top_row, "THERMOSTAT")
        self.thermostat_card.pack(side="left", fill="both", expand=True, padx=(5, 0))
        
        self.temp_display = tk.Label(
            self.thermostat_card.content_frame,
            text="22°C",
            font=(FONT_FAMILY, 32, "bold"),
            bg=BG_COLOR, fg=ACTIVE_COLOR
        )
        self.temp_display.pack(pady=(10, 0))
        
        self.temp_label = tk.Label(
            self.thermostat_card.content_frame,
            text="Temperature",
            font=(FONT_FAMILY, 10),
            bg=BG_COLOR, fg=BORDER_COLOR
        )
        self.temp_label.pack(pady=(0, 10))
        
        # ── Middle Row: Lights, Security ────────────────────────────
        middle_row = tk.Frame(self.scrollable_frame, bg=BG_COLOR)
        middle_row.pack(fill="x", padx=10, pady=5)
        
        # Lights Card
        self.lights_card = BentoCard(middle_row, "LIGHTS")
        self.lights_card.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self.light_switches = {}
        light_rooms = [
            ("living_room_light", "Living Room"),
            ("kitchen_light", "Kitchen"),
            ("bedroom_light", "Bedroom")
        ]
        
        for device_id, room_name in light_rooms:
            row = tk.Frame(self.lights_card.content_frame, bg=BG_COLOR)
            row.pack(fill="x", pady=2)
            
            label = tk.Label(
                row, text=room_name,
                font=(FONT_FAMILY, 10),
                bg=BG_COLOR, fg=BORDER_COLOR,
                anchor="w", width=15
            )
            label.pack(side="left")
            
            switch = StatusSwitch(
                row,
                on_text="ON", off_text="OFF",
                initial_state=(self.simulator.get_device_state(device_id).state == "on"),
                command=lambda id=device_id: self._manual_toggle(id)
            )
            switch.pack(side="right", padx=5)
            self.light_switches[device_id] = switch
        
        # Security Card
        self.security_card = BentoCard(middle_row, "SECURITY")
        self.security_card.pack(side="left", fill="both", expand=True, padx=(5, 0))
        
        self.door_switches = {}
        door_rooms = [
            ("front_door", "Front Door", "LOCKED", "UNLOCKED"),
            ("back_door", "Back Door", "LOCKED", "UNLOCKED")
        ]
        
        for device_id, door_name, locked_text, unlocked_text in door_rooms:
            row = tk.Frame(self.security_card.content_frame, bg=BG_COLOR)
            row.pack(fill="x", pady=2)
            
            label = tk.Label(
                row, text=door_name,
                font=(FONT_FAMILY, 10),
                bg=BG_COLOR, fg=BORDER_COLOR,
                anchor="w", width=15
            )
            label.pack(side="left")
            
            switch = StatusSwitch(
                row,
                on_text=locked_text, off_text=unlocked_text,
                initial_state=(self.simulator.get_device_state(device_id).state == "locked"),
                command=lambda id=device_id: self._manual_toggle(id)
            )
            switch.pack(side="right", padx=5)
            self.door_switches[device_id] = switch
        
        # ── Bottom Row: Entertainment ───────────────────────────────
        bottom_row = tk.Frame(self.scrollable_frame, bg=BG_COLOR)
        bottom_row.pack(fill="x", padx=10, pady=(5, 10))
        
        self.entertainment_card = BentoCard(bottom_row, "ENTERTAINMENT")
        self.entertainment_card.pack(fill="both", expand=True)
        
        # TV controls
        tv_row = tk.Frame(self.entertainment_card.content_frame, bg=BG_COLOR)
        tv_row.pack(fill="x", pady=2)
        
        tv_label = tk.Label(
            tv_row, text="TV",
            font=(FONT_FAMILY, 10),
            bg=BG_COLOR, fg=BORDER_COLOR,
            anchor="w", width=15
        )
        tv_label.pack(side="left")
        
        self.tv_switch = StatusSwitch(
            tv_row,
            on_text="ON", off_text="OFF",
            initial_state=(self.simulator.get_device_state("tv").state in ["on", "playing"]),
            command=lambda: self._manual_toggle("tv")
        )
        self.tv_switch.pack(side="right", padx=5)
        
        # TV status
        self.tv_status_label = tk.Label(
            self.entertainment_card.content_frame,
            text="Status: Off",
            font=(FONT_FAMILY, 9),
            bg=BG_COLOR, fg=INACTIVE_COLOR,
            anchor="w"
        )
        self.tv_status_label.pack(anchor="w", padx=10)
        
        # Speaker controls
        speaker_row = tk.Frame(self.entertainment_card.content_frame, bg=BG_COLOR)
        speaker_row.pack(fill="x", pady=(10, 2))
        
        speaker_label = tk.Label(
            speaker_row, text="Speaker",
            font=(FONT_FAMILY, 10),
            bg=BG_COLOR, fg=BORDER_COLOR,
            anchor="w", width=15
        )
        speaker_label.pack(side="left")
        
        self.speaker_switch = StatusSwitch(
            speaker_row,
            on_text="ON", off_text="OFF",
            initial_state=(self.simulator.get_device_state("speaker").state == "on"),
            command=lambda: self._manual_toggle("speaker")
        )
        self.speaker_switch.pack(side="right", padx=5)
        
        # Volume display
        self.volume_label = tk.Label(
            self.entertainment_card.content_frame,
            text="Volume: 50%",
            font=(FONT_FAMILY, 9),
            bg=BG_COLOR, fg=INACTIVE_COLOR,
            anchor="w"
        )
        self.volume_label.pack(anchor="w", padx=10)
    
    # ══════════════════════════════════════════════════════════════════
    #  GUI UPDATE METHODS
    # ══════════════════════════════════════════════════════════════════
    
    def _update_status(self, status: str):
        """Update assistant status in GUI (thread-safe)."""
        self.root.after(0, self._set_status_text, status)
    
    def _set_status_text(self, status: str):
        """Set status label text."""
        self.status_label.config(text=status)
        
        if status == "Ready":
            self.instruction_label.config(text='Say "Hey Sophia"')
        else:
            self.instruction_label.config(text="")
        
        logger.info(f"Assistant status changed: {status}")
    
    def _update_activity(self, command: str, understood: str, response: str):
        """Update latest activity card."""
        self.you_said_label.config(text=f'YOU SAID:\n"{command}"')
        self.understood_label.config(text=f"UNDERSTOOD:\n{understood}")
        self.sophia_label.config(text=f'SOPHIA:\n"{response}"')
    
    def _update_all_devices(self):
        """Update all device switches based on simulator state."""
        # Lights
        for device_id, switch in self.light_switches.items():
            state = self.simulator.get_device_state(device_id)
            switch.set_state(state.state == "on")
        
        # Doors
        for device_id, switch in self.door_switches.items():
            state = self.simulator.get_device_state(device_id)
            switch.set_state(state.state == "locked")
        
        # TV
        tv_state = self.simulator.get_device_state("tv")
        self.tv_switch.set_state(tv_state.state in ["on", "playing"])
        self.tv_status_label.config(text=f"Status: {tv_state.state.title()}")
        
        # Speaker
        speaker_state = self.simulator.get_device_state("speaker")
        self.speaker_switch.set_state(speaker_state.state == "on")
        self.volume_label.config(text=f"Volume: {speaker_state.value}%")
        
        # Thermostat
        thermostat_state = self.simulator.get_device_state("thermostat")
        self.temp_display.config(text=f"{thermostat_state.value}°C")
    
    def _manual_toggle(self, device_id: str):
        """Handle manual switch clicks for testing."""
        device = self.simulator.get_device_state(device_id)
        
        if device_id in self.simulator.LIGHT_DEVICES:
            action = "turn_on" if device.state != "on" else "turn_off"
            self.simulator.apply_command(action, device_id)
        elif device_id in self.simulator.DOOR_DEVICES:
            action = "lock" if device.state != "locked" else "unlock"
            self.simulator.apply_command(action, device_id)
        elif device_id == "tv":
            action = "turn_on" if device.state in ["off", "stopped"] else "turn_off"
            self.simulator.apply_command(action, device_id)
        elif device_id == "speaker":
            action = "turn_on" if device.state != "on" else "turn_off"
            self.simulator.apply_command(action, device_id)
        
        self._update_all_devices()
        logger.info(f"Manual toggle: {device_id} -> {self.simulator.get_device_state(device_id).state}")
    
    # ══════════════════════════════════════════════════════════════════
    #  VOICE PROCESSING
    # ══════════════════════════════════════════════════════════════════
    
    def _start_voice_thread(self):
        """Start background thread for voice processing."""
        self.voice_thread = threading.Thread(
            target=self._voice_worker,
            daemon=True,
            name="Voice-Thread"
        )
        self.voice_thread.start()
        logger.info("Voice thread started.")
    
    def _voice_worker(self):
        """Main voice processing loop."""
        while True:
            try:
                # Listen for command
                self._update_status("Listening...")
                command = self.voice_pipeline.listen_for_command()
                
                if command:
                    logger.info(f"Command transcribed: '{command}'")
                    self._process_command(command)
                
                time.sleep(0.1)  # Small delay to prevent busy loop
                
            except Exception as e:
                logger.error(f"Voice worker error: {e}")
                self._update_status("Error")
                time.sleep(1)
    
    def _process_command(self, command: str):
        """Process a transcribed command through AI and simulator."""
        self._update_status("Thinking...")
        logger.info(f"AI processing started for: '{command}'")
        
        try:
            # Parse intent with AI
            if self.ai_engine:
                response = self.ai_engine.parse_intent(command)
            else:
                response = None
                logger.warning("AI engine unavailable.")
            
            if response and response.actions:
                # ── Filter and validate actions ────────────────────
                valid_actions = []
                clarification_needed = False
                
                for action in response.actions:
                    # Check if target exists
                    if hasattr(self.simulator, 'devices') and action.target in self.simulator.devices:
                        # Check if action is valid for target
                        valid_actions_for_target = self.simulator._get_valid_actions(action.target)
                        if action.action in valid_actions_for_target:
                            valid_actions.append(action)
                        else:
                            logger.warning(f"Removed invalid action: {action.action} for {action.target}")
                            clarification_needed = True
                    else:
                        logger.warning(f"Removed action for unknown device: {action.target}")
                        clarification_needed = True
                
                # ── Check for ambiguous commands ────────────────────
                if not clarification_needed and len(valid_actions) > 1:
                    # Check if user specified "one" or "a" but AI returned multiple devices
                    command_lower = command.lower()
                    if re.search(r'\b(one|a|an)\s+(light|door)\b', command_lower):
                        # Check if all actions target the same device type
                        device_types = set()
                        for action in valid_actions:
                            if action.target in self.simulator.LIGHT_DEVICES:
                                device_types.add("light")
                            elif action.target in self.simulator.DOOR_DEVICES:
                                device_types.add("door")
                        
                        if len(device_types) == 1 and len(valid_actions) > 1:
                            # User said "one light" but AI returned multiple lights
                            clarification_needed = True
                            valid_actions = []
                
                # ── Handle clarification or execute ────────────────
                if clarification_needed:
                    # Ambiguous or invalid command
                    feedback = "I'm sorry, I couldn't perform that action. Please try to be more specific."
                    logger.info(f"Clarification needed for: '{command}'")
                    self._update_activity(command, "Clarification needed", feedback)
                    if self.voice_pipeline:
                        self.voice_pipeline.speak(feedback, blocking=False)
                    return
                
                elif valid_actions:
                    # Execute valid commands ONCE
                    self._update_status("Executing...")
                    logger.info(f"Executing {len(valid_actions)} action(s).")
                    
                    understood_parts = []
                    failed_parts = []
                    execution_results = []
                    any_adjusted = False
                    
                    # Execute each action exactly ONCE
                    for action in valid_actions:
                        result = self.simulator.apply_command(
                            action=action.action,
                            target=action.target,
                            value=action.value
                        )
                        execution_results.append(result)
                        
                        if result["success"]:
                            understood_parts.append(result["message"])
                            if result.get("was_adjusted", False):
                                any_adjusted = True
                        else:
                            failed_parts.append(result["message"])
                    
                    # Generate spoken feedback based on execution results
                    if any_adjusted:
                        # At least one action was clamped - use simulator's corrected message
                        adjusted_messages = [r["message"] for r in execution_results if r.get("was_adjusted", False)]
                        non_adjusted_messages = [r["message"] for r in execution_results if r["success"] and not r.get("was_adjusted", False)]
                        
                        # Combine adjusted and non-adjusted messages naturally
                        if adjusted_messages and non_adjusted_messages:
                            response.spoken_feedback = " ".join(non_adjusted_messages) + " Also, " + " ".join(adjusted_messages)
                        elif adjusted_messages:
                            response.spoken_feedback = " ".join(adjusted_messages)
                    elif failed_parts and not understood_parts:
                        # All actions failed
                        response.spoken_feedback = " ".join(failed_parts)
                    # If no adjustments and no total failure, keep AI's original spoken_feedback
                    
                    # Build understood text for GUI
                    if understood_parts and failed_parts:
                        understood = "; ".join(understood_parts) + " | Failed: " + "; ".join(failed_parts)
                    elif understood_parts:
                        understood = "; ".join(understood_parts)
                    else:
                        understood = "Failed: " + "; ".join(failed_parts)
                    
                    logger.info(f"Actions completed: {understood}")
                    
                    # Update GUI
                    self.root.after(0, self._update_all_devices)
                    self.root.after(0, self._update_activity, 
                                   command, understood, response.spoken_feedback)
                    
                    # Speak response
                    if self.voice_pipeline:
                        self.voice_pipeline.speak(response.spoken_feedback, blocking=False)
                    
                    logger.info("AI processing completed.")
                else:
                    # No valid actions after filtering
                    feedback = "I couldn't perform that action. Please try a different command."
                    logger.info(f"No valid actions for: '{command}'")
                    self._update_activity(command, "No valid actions", feedback)
                    if self.voice_pipeline:
                        self.voice_pipeline.speak(feedback, blocking=False)
                
            elif response and response.spoken_feedback:
                # No actions, just feedback (clarification or unsupported command)
                logger.info(f"AI returned no actions: '{response.spoken_feedback}'")
                self._update_activity(command, "No actions", response.spoken_feedback)
                if self.voice_pipeline:
                    self.voice_pipeline.speak(response.spoken_feedback, blocking=False)
                
            else:
                # No response at all
                feedback = "I'm sorry, I didn't understand that command."
                logger.info(f"No response for: '{command}'")
                self._update_activity(command, "Not understood", feedback)
                if self.voice_pipeline:
                    self.voice_pipeline.speak(feedback, blocking=False)
                
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            self._update_status("Error")
            feedback = "Sorry, I encountered an error processing your request."
            self._update_activity(command, "Error", feedback)
            if self.voice_pipeline:
                self.voice_pipeline.speak(feedback, blocking=False)
        
        finally:
            # Return to ready state
            self._update_status("Ready")
    
    # ══════════════════════════════════════════════════════════════════
    #  CLEANUP
    # ══════════════════════════════════════════════════════════════════
    
    def _on_closing(self):
        """Handle application closure."""
        logger.info("Application closing...")
        
        if self.voice_pipeline:
            self.voice_pipeline.stop()
        
        self.root.destroy()
        logger.info("Application closed.")


# ══════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app = ApexAssistantApp(root)
    root.mainloop()