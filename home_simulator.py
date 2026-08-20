"""
home_simulator.py
=================
Module: Simulated Smart-Home State Machine
Author: [Your Name] - Partner 2 (Interface & Integration Engineer)
Date: August 2026

Description:
    Maintains the simulated state of all smart-home devices.
    Receives structured commands from ai_engine.py and applies
    state changes. Returns updated states for GUI rendering.
    
    This module does NOT interact with Tkinter directly.
    It is a pure state management layer.

Device Types:
    - Lights: living_room, kitchen, bedroom
    - Thermostat: temperature control
    - Security: front_door, back_door (lock/unlock)
    - Entertainment: tv, speaker (on/off, play/pause/stop, volume)

Supported Actions:
    turn_on, turn_off, set_temp, increase_temp, decrease_temp,
    lock, unlock, set_volume, play, pause, stop
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger("ApexLogger")


@dataclass
class DeviceState:
    """Represents the current state of a single device."""
    name: str
    state: str  # "on", "off", "locked", "unlocked", "playing", "paused", "stopped"
    value: Optional[int] = None  # temperature, volume, etc.


class HomeSimulator:
    """
    Simulated smart-home environment.
    
    Maintains the state of all virtual devices and applies
    structured commands from the AI engine.
    """
    
    # ── Device Registry ────────────────────────────────────────────
    LIGHT_DEVICES = ["living_room_light", "kitchen_light", "bedroom_light"]
    DOOR_DEVICES = ["front_door", "back_door"]
    ENTERTAINMENT_DEVICES = ["tv", "speaker", "entertainment_unit"]
    
    # ── Valid Actions per Device Category ──────────────────────────
    LIGHT_ACTIONS = ["turn_on", "turn_off"]
    DOOR_ACTIONS = ["lock", "unlock"]
    THERMOSTAT_ACTIONS = ["set_temp", "increase_temp", "decrease_temp"]
    ENTERTAINMENT_ACTIONS = ["turn_on", "turn_off", "play", "pause", "stop", "set_volume", "increase_volume", "decrease_volume"]
    
    def __init__(self):
        """Initialize all devices to their default states."""
        self.devices: Dict[str, DeviceState] = {}
        self.last_actions: List[Dict[str, Any]] = []
        
        # ── Initialize Lights ──────────────────────────────────────
        for light in self.LIGHT_DEVICES:
            self.devices[light] = DeviceState(
                name=light.replace("_", " ").title(),
                state="off"
            )
        
        # ── Initialize Thermostat ──────────────────────────────────
        self.devices["thermostat"] = DeviceState(
            name="Thermostat",
            state="on",
            value=22  # Default temperature in Celsius
        )
        
        # ── Initialize Doors ───────────────────────────────────────
        for door in self.DOOR_DEVICES:
            self.devices[door] = DeviceState(
                name=door.replace("_", " ").title(),
                state="locked"
            )
        
        # ── Initialize Entertainment ───────────────────────────────
        self.devices["tv"] = DeviceState(
            name="TV",
            state="off",
            value=None
        )
        self.devices["speaker"] = DeviceState(
            name="Speaker",
            state="off",
            value=50  # Default volume
        )
        
        logger.info("HomeSimulator initialized with default device states.")
    
    def get_state(self) -> Dict[str, Any]:
        """Return the complete state of all devices."""
        return {
            "devices": {name: asdict(state) for name, state in self.devices.items()},
            "last_actions": self.last_actions
        }
    
    def get_device_state(self, device_id: str) -> Optional[DeviceState]:
        """Get the state of a specific device."""
        return self.devices.get(device_id)
    
    def apply_command(self, action: str, target: str, value: Optional[int] = None) -> Dict[str, Any]:
        """
        Apply a single structured command to the simulator.
        
        Args:
            action: The action to perform (turn_on, set_temp, etc.)
            target: The target device identifier
            value: Optional parameter (temperature, volume, etc.)
            
        Returns:
            Dictionary containing:
                - success: Whether the command was applied
                - message: Human-readable description
                - device: The updated device state
        """
        # ── Validate Target ────────────────────────────────────────
        if target not in self.devices:
            logger.warning(f"Unknown target device: {target}")
            return {
                "success": False,
                "message": f"Unknown device: {target}",
                "device": None
            }
        
        device = self.devices[target]
        
        # ── Validate Action for Target Type ────────────────────────
        valid_actions = self._get_valid_actions(target)
        if action not in valid_actions:
            logger.warning(f"Invalid action '{action}' for device '{target}'")
            return {
                "success": False,
                "message": f"Cannot perform '{action}' on {device.name}",
                "device": None
            }
        
        # ── Apply Action ───────────────────────────────────────────
        old_state = device.state
        old_value = device.value
        
        try:
            if action == "turn_on":
                device.state = "on"
            elif action == "turn_off":
                device.state = "off"
            elif action == "lock":
                device.state = "locked"
            elif action == "unlock":
                device.state = "unlocked"
            elif action == "set_temp":
                if value is None or not (10 <= value <= 35):
                    return {
                        "success": False,
                        "message": f"Temperature must be between 10-35°C",
                        "device": None
                    }
                device.value = value
                device.state = "on"
            elif action == "increase_temp":
                increment = value if value else 2
                new_temp = (device.value or 22) + increment
                if new_temp > 35:
                    new_temp = 35
                device.value = new_temp
                device.state = "on"
            elif action == "decrease_temp":
                decrement = value if value else 2
                new_temp = (device.value or 22) - decrement
                if new_temp < 10:
                    new_temp = 10
                device.value = new_temp
                device.state = "on"
            elif action == "increase_volume":
                # Increase by 10% or specified amount
                increment = value if value else 10
                new_volume = min(100, (device.value or 50) + increment)
                device.value = new_volume
            elif action == "decrease_volume":
                # Decrease by 10% or specified amount
                decrement = value if value else 10
                new_volume = max(0, (device.value or 50) - decrement)
                device.value = new_volume
            elif action == "play":
                device.state = "playing"
                if device.name == "TV":
                    device.state = "playing"
            elif action == "pause":
                device.state = "paused"
            elif action == "stop":
                device.state = "stopped"
            
            # ── Log State Change ───────────────────────────────────
            change_description = self._describe_change(action, target, old_state, device.state, old_value, device.value)
            self.last_actions.append({
                "action": action,
                "target": target,
                "old_state": old_state,
                "new_state": device.state,
                "old_value": old_value,
                "new_value": device.value,
                "description": change_description,
                "timestamp": self._get_timestamp()
            })
            
            logger.info(f"State change: {change_description}")
            
            return {
                "success": True,
                "message": change_description,
                "device": asdict(device)
            }
            
        except Exception as e:
            logger.error(f"Error applying command: {e}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "device": None
            }
    
    def apply_commands(self, commands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply multiple commands in sequence.
        
        Args:
            commands: List of command dictionaries with 'action', 'target', 'value' keys
            
        Returns:
            List of result dictionaries from apply_command
        """
        results = []
        for cmd in commands:
            result = self.apply_command(
                action=cmd.get("action"),
                target=cmd.get("target"),
                value=cmd.get("value")
            )
            results.append(result)
        return results
    
    def _get_valid_actions(self, target: str) -> List[str]:
        """Determine which actions are valid for a given target."""
        if target in self.LIGHT_DEVICES:
            return self.LIGHT_ACTIONS
        elif target in self.DOOR_DEVICES:
            return self.DOOR_ACTIONS
        elif target == "thermostat":
            return self.THERMOSTAT_ACTIONS
        elif target in self.ENTERTAINMENT_DEVICES:
            return self.ENTERTAINMENT_ACTIONS
        return []
    
    def _describe_change(self, action: str, target: str, old_state: str, 
                        new_state: str, old_value: Optional[int], new_value: Optional[int]) -> str:
        """Generate a human-readable description of a state change."""
        device = self.devices[target]
        
        if action in ["turn_on", "turn_off"]:
            return f"{device.name} turned {'on' if new_state == 'on' else 'off'}"
        elif action in ["lock", "unlock"]:
            return f"{device.name} {'locked' if new_state == 'locked' else 'unlocked'}"
        elif action == "set_temp":
            return f"{device.name} set to {new_value}°C"
        elif action == "increase_temp":
            return f"{device.name} increased to {new_value}°C"
        elif action == "decrease_temp":
            return f"{device.name} decreased to {new_value}°C"
        elif action == "increase_volume":
            return f"{device.name} volume increased to {new_value}%"
        elif action == "decrease_volume":
            return f"{device.name} volume decreased to {new_value}%"
        elif action == "play":
            return f"{device.name} is now playing"
        elif action == "pause":
            return f"{device.name} paused"
        elif action == "stop":
            return f"{device.name} stopped"
        return f"{device.name} state changed"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for logging."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def reset_all(self):
        """Reset all devices to default states."""
        self.__init__()
        logger.info("All devices reset to default states.")


# ── Quick Test Function ──────────────────────────────────────────
if __name__ == "__main__":
    # Test the simulator independently
    sim = HomeSimulator()
    
    # Test light control
    result = sim.apply_command("turn_on", "living_room_light")
    print(f"Light test: {result['message']}")
    
    # Test thermostat
    result = sim.apply_command("set_temp", "thermostat", 25)
    print(f"Thermostat test: {result['message']}")
    
    # Test door
    result = sim.apply_command("unlock", "front_door")
    print(f"Door test: {result['message']}")
    
    # Test volume
    result = sim.apply_command("set_volume", "speaker", 70)
    print(f"Volume test: {result['message']}")
    
    # Print all states
    print("\nFinal Device States:")
    for name, state in sim.devices.items():
        print(f"  {state.name}: {state.state} (value: {state.value})")