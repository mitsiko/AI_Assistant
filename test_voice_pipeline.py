from voice_pipeline import VoicePipeline

print("Initializing Voice Pipeline...")

voice = VoicePipeline()

print("\nVoice pipeline ready!")
print('Say "Hey Sophia" followed by a command.')
print('Example: "Hey Sophia, turn on the living room lights."')
print("Press Ctrl+C to stop.\n")

try:
    command = voice.listen_for_command()

    if command:
        print(f"\nYOU SAID: {command}")

        voice.speak(
            f"I heard your command: {command}"
        )
    else:
        print("\nNo command detected.")

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    voice.stop()
    print("Voice pipeline shut down.")