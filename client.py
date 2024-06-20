import sounddevice as sd
import numpy as np
import socket
import argparse
import threading
import time
import asyncio
from llm import llm_request

# Silence configuration
SILENCE_THRESHOLD = 0.01  # Adjust this value based on your microphone sensitivity
SILENCE_DURATION = 4  # Silence duration in seconds to trigger the alert
last_received_time = time.time()
speech = {
    "start_time": 0,
    "end_time": 0,
    "text": ""
}

# Create a socket connection to the server
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def audio_callback(indata, frames, time_info, status):        
        if status:
            print(status)
        
        sock.sendall(indata.tobytes())

async def receive_transcriptions():
    global last_received_time
    global speech
    last_received_time = time.time()
    pause_threshold = 4  # Pause threshold in seconds

    while True:
        try:
            data = sock.recv(1024).decode().strip()
            if not data:
                break
            if not len(data) == 1024:
                current_time = time.time()
                pause_duration = current_time - last_received_time

                if pause_duration > pause_threshold:
                    print(f"Alert: Pause longer than {pause_threshold} seconds detected!")

                last_received_time = current_time

                # Parse the timestamps and text
                parts = data.split(" ", 2)
                if len(parts) == 3:
                    time_start = int(parts[0])
                    time_end = int(parts[1])
                    text = parts[2].strip()
                    print(f"Start time: {time_start} ms")
                    print(f"End time: {time_end} ms")
                    print(f"Text: {text}")
                    speech["text"] += text.replace("\n", " ")
                    print(f'Full text: {speech["text"]}')
                    print("---")

                    # Make an asynchronous call to the LLM here
                    await call_llm(speech["text"])
                else:
                    print(f"Received invalid data: {data}")
        except socket.error as e:
            print(f"Socket error: {e}")
            break

async def call_llm(text):
    # Implement your asynchronous LLM call here
    # This is just a placeholder example
    # await asyncio.sleep(1)
    system_content = "Вы оператор сервисного центра по ремонту бытовой технике. Старайтесь не отвечать длинным текстом без крайней необходимости, ведь это телефонный разговор. Клиент не сможет прочитать ваш ответ, он будет его слушать."
    text = llm_request(system_content, text)
    print(f"[>> LLM >>] {text}")

async def main():
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

    sock.connect((SERVER_HOST, SERVER_PORT))

    # Send the model and language to the server
    sock.sendall(f"{args.model}|{args.language}\n".encode())

    # Start the receiving coroutine
    receiving_task = asyncio.create_task(receive_transcriptions())

    # Start the audio stream from the microphone
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16',
                            blocksize=BLOCKSIZE, callback=audio_callback)

    print("Recording and sending audio to the server. Press Ctrl+C to stop.")

    try:
        with stream:
            await asyncio.sleep(60)  # Sleep for N seconds (adjust as needed)
    except KeyboardInterrupt:
        print("Recording stopped by the user.")
    finally:
        # Close the socket connection gracefully
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()
        print("Socket connection closed.")

    # Wait for the receiving coroutine to finish
    await receiving_task

if __name__ == "__main__":
    asyncio.run(main())
