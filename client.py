import sounddevice as sd
import numpy as np
import socket
import argparse
import threading

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Real-time audio streaming client')
parser.add_argument('--model', type=str, default='large-v3', help='Whisper model to use')
parser.add_argument('--language', type=str, default='en', help='Language code for transcription')
args = parser.parse_args()

# Server configuration
SERVER_HOST = 'localhost'
SERVER_PORT = 43007

# Audio configuration
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 1024

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    sock.sendall(indata.tobytes())

def receive_transcriptions():
    while True:
        try:
            data = sock.recv(1024).decode().strip()
            if not data:
                break
            if not len(data) == 1024:
                # Parse the timestamps and text
                parts = data.split(" ", 2)
                if len(parts) == 3:
                    time_start = int(parts[0])
                    time_end = int(parts[1])
                    text = parts[2].strip()
                    print(f"Start time: {time_start} ms")
                    print(f"End time: {time_end} ms")
                    print(f"Text: {text}")
                    print("---")
                else:
                    print(f"Received invalid data: {data}")
        except socket.error as e:
            print(f"Socket error: {e}")
            break

# Create a socket connection to the server
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_HOST, SERVER_PORT))

# Send the model and language to the server
sock.sendall(f"{args.model}|{args.language}\n".encode())

# Start the receiving thread
receiving_thread = threading.Thread(target=receive_transcriptions)
receiving_thread.start()

# Start the audio stream from the microphone
stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16',
                        blocksize=BLOCKSIZE, callback=audio_callback)

print("Recording and sending audio to the server. Press Ctrl+C to stop.")

try:
    with stream:
        sd.sleep(30 * 1000)  # Sleep for 5 seconds (adjust as needed)
except KeyboardInterrupt:
    print("Recording stopped by the user.")
finally:
    # Close the socket connection gracefully
    sock.shutdown(socket.SHUT_RDWR)
    sock.close()
    print("Socket connection closed.")

# Wait for the receiving thread to finish
receiving_thread.join()