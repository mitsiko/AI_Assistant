import os
import json
import queue
import pyaudio
from vosk import Model, KaldiRecognizer, SetLogLevel

# 1. Hide Vosk's messy C++ backend logs from your console
SetLogLevel(-1)

# 2. THE BULLETPROOF PATH FINDER
# Automatically finds the folder where THIS python script lives
current_script_folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_script_folder, "models", "vosk-model-small-en-us-0.15")

# 3. Verify it exists before trying to load it
if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    print("Please ensure the 'am', 'conf', etc. folders are inside the model folder.")
    exit(1)

print("Loading Vosk Model... (This takes ~1 second)")
model = Model(model_path)
recognizer = KaldiRecognizer(model, 16000)

print("✅ Model loaded successfully! Speak into your microphone... (Press Ctrl+C to stop)")

# 4. Open the microphone stream
mic = pyaudio.PyAudio()
stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

try:
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        
        # 5. Process the audio
        if recognizer.AcceptWaveform(data):
            # Get the final spoken sentence
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            
            if text:
                print(f"YOU SAID: {text}")
                
except KeyboardInterrupt:
    print("\nStopping test...")
finally:
    stream.stop_stream()
    stream.close()
    mic.terminate()
    print("Microphone closed. Goodbye!")