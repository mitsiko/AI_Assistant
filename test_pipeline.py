# test_pipeline.py (temporary, not part of deliverables)
from ai_engine import AIEngine

engine = AIEngine()

# Test 1: Direct command
r1 = engine.parse_intent("Turn on the kitchen lights and lock the front door")
print(r1.model_dump_json(indent=2) if r1 else "FAILED")

# Test 2: Ambiguous command (rubric edge case)
r2 = engine.parse_intent("It's getting dark in here and I'm freezing")
print(r2.model_dump_json(indent=2) if r2 else "FAILED")

# Test 3: Multi-device complex
r3 = engine.parse_intent("Set the thermostat to 22 and turn off all lights")
print(r3.model_dump_json(indent=2) if r3 else "FAILED")
