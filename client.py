import sounddevice as sd
import numpy as np
import socket
import argparse
import threading
import time
import asyncio
from llm import llm_request
from tts import speech_synthesis

# Add this global variable
is_speech_active = True

# system_content = "Вы оператор колл центра. Ваша задача выяснить доволен ли клиент ремонтом и выслушать замечания клиента. Если понадобится завершить разговор, используйте слово hangup."
system_content = "Вы - Татьяна, оператор сервисного центра Айсберг. Ваша задача выяснить доволен ли клиент недавним ремонтом и записать отзыв клиента. Если понадобится завершить разговор, используйте слово hangup. Не забывайте что ваш пол - женский, имейте это ввиду когда говорите от вашего имени."
llm_messages=[
            {"role": "system", "content": system_content}
        ]

# Add this constant at the top of your file, with other constants
TRANSCRIPTION_DELAY = 1  # Delay in seconds to wait for new transcriptions

# Silence configuration
# SILENCE_THRESHOLD = 0.01  # Adjust this value based on your microphone sensitivity
# SILENCE_DURATION = 4  # Silence duration in seconds to trigger the alert
last_received_time = time.time()
speech = {
    "start_time": 0,
    "end_time": 0,
    "speech_end_time": 0,
    "text": ""
}

# Create a socket connection to the server
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def audio_callback(indata, frames, time_info, status):        
    global is_speech_active
    
    if status:
        print(status)
    
    if is_speech_active:
        sock.sendall(indata.tobytes())
    # else:
    #     # Optionally, send silent data instead of completely muting
    #     silent_data = np.zeros_like(indata)
    #     sock.sendall(silent_data.tobytes())

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
                    print(f"Speech end time: {speech['speech_end_time']} ms")
                    print(f"Text: {text}")
                    speech["text"] += text.replace("\n", " ")
                    print(f'Full text: {speech["text"]}')
                    print("---")

                    # Make an asynchronous call to the LLM here
                    await call_llm(time_end, speech["text"])
                else:
                    print(f"Received invalid data: {data}")
        except socket.error as e:
            print(f"Socket error: {e}")
            break

async def call_llm(time_end, user_text):
    global is_speech_active
    global llm_messages
    global speech
    global last_received_time
    
    # Wait for the delay period or until new transcription arrives
    # delay_start = time.time()
    # while time.time() - delay_start < TRANSCRIPTION_DELAY:
    #     if time.time() - last_received_time < TRANSCRIPTION_DELAY:
    #         # New transcription received, reset the delay
    #         delay_start = time.time()
    #     await asyncio.sleep(0.1)  # Short sleep to prevent busy waiting
    
    # speech_start_time_int = time_end
    speech_start_time = time.time()
    
    # Rest of your existing code...
    model = "gpt-4o"
    # model = "gpt-3.5-turbo-1106"
    
    llm_messages.append({"role": "user", "content": user_text})
    text = await llm_request(llm_messages, model)
    if "hangup" in text:
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()
        print("Socket connection closed.")
        exit()

    llm_messages.append({"role": "assistant", "content": text})
    
    if speech["start_time"] < time_end:        
        print(f"[>> LLM >>] {text}")
        # Speech synthesis
        model = 'tts-1'
        voice_id = "shimmer"
        speed = 1.1
        # Mute microphone
        is_speech_active = False
        await speech_synthesis(text, model, voice_id, speed)
        print('# Speech synthesis done!')
        speech["speech_end_time"] = time.time() - speech_start_time
        # Unmute microphone
        is_speech_active = True
        # Clean text
        speech["text"] = ""
    else:
        print(f'# llm interrupted. start_time: {speech["start_time"]}, time_end: {time_end}')
        await asyncio.sleep(1)
    
    # Unmute microphone
    is_speech_active = True
    

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
            await asyncio.sleep(600)  # Sleep for N seconds (adjust as needed)
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
