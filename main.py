import pyaudio
import wave
import numpy as np
import time
import os
from pydub import AudioSegment
from pydub.playback import play
from src.laughter_detection import detect_laughter

# Constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "temp_audio.wav"

# Initialize PyAudio
p = pyaudio.PyAudio()

# Load laughter sound
laughter_sound = AudioSegment.from_wav("laugh.wav")  # Replace with your laughter sound file

def record_audio():
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Done recording.")

    stream.stop_stream()
    stream.close()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def main_loop():
    try:
        while True:
            # Record audio
            record_audio()

            # Detect laughter
            laughter_data = detect_laughter(WAVE_OUTPUT_FILENAME)

            if laughter_data:
                print("Laughter detected!")
                play(laughter_sound)
            else:
                print("No laughter detected.")

            # Optional: Add a small delay before the next recording
            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping the program...")
    finally:
        p.terminate()
        if os.path.exists(WAVE_OUTPUT_FILENAME):
            os.remove(WAVE_OUTPUT_FILENAME)

if __name__ == '__main__':
    main_loop()
