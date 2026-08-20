"""Integration test for AI → Simulator pipeline."""
from ai_engine import AIEngine
from home_simulator import HomeSimulator

# Initialize components
sim = HomeSimulator()
ai = AIEngine()

# Test commands
test_commands = [
    "turn on the living room lights",
    "can you please set the temperature to 25 degrees",
    "lock all doors",
    "turn off all lights except the kitchen",
    "it's getting dark in here",
    "I'm going to sleep now",
    "set the volume to 75 percent",
    "pause the TV"
]

for command in test_commands:
    print(f"\n{'='*50}")
    print(f"Command: {command}")
    
    response = ai.parse_intent(command)
    if response and response.actions:
        print(f"Actions: {len(response.actions)}")
        for action in response.actions:
            print(f"  - {action.action} {action.target} {action.value}")
            result = sim.apply_command(action.action, action.target, action.value)
            print(f"    Result: {result['message']}")
        print(f"Feedback: {response.spoken_feedback}")
    elif response:
        print(f"No actions: {response.spoken_feedback}")
    else:
        print("Failed to parse command")

# Show final state
print(f"\n{'='*50}")
print("Final State:")
for name, state in sim.devices.items():
    print(f"  {state.name}: {state.state} (value: {state.value})")