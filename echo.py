#!/usr/bin/env python3
import sys
import argparse
import logging
import numpy as np
import sounddevice as sd
import asyncio
import websockets
import json

logger = logging.getLogger(__name__)

class WhisperClient:
    def __init__(self, host, port, lang="auto", translate=False, use_vad=False):
        self.uri = f"ws://{host}:{port}/ws"
        self.lang = lang
        self.translate = translate
        self.use_vad = use_vad

    async def connect(self):
        self.websocket = await websockets.connect(self.uri)
        await self.send_config()

    async def send_config(self):
        config = {
            "language": self.lang,
            "task": "translate" if self.translate else "transcribe",
            "use_vad": self.use_vad
        }
        await self.websocket.send(json.dumps(config))

    async def transcribe(self, audio_chunk):
        await self.websocket.send(audio_chunk.tobytes())
        response = await self.websocket.recv()
        return json.loads(response)

    async def close(self):
        await self.websocket.close()

class AudioProcessor:
    def __init__(self, client, min_chunk):
        self.client = client
        self.min_chunk = min_chunk
        self.audio_buffer = []
        self.last_end = None

    def process_audio(self, indata, frames, time_info, status):
        if status:
            print(status)
        
        self.audio_buffer.append(indata.copy())
        if len(self.audio_buffer) * BLOCKSIZE >= self.min_chunk * SAMPLING_RATE:
            audio_chunk = np.concatenate(self.audio_buffer)
            self.audio_buffer = []
            asyncio.create_task(self.process_chunk(audio_chunk))

    async def process_chunk(self, audio_chunk):
        result = await self.client.transcribe(audio_chunk)
        self.format_and_process_output(result)

    def format_and_process_output(self, result):
        if 'text' in result and result['text'].strip():
            beg, end = result.get('start', 0) * 1000, result.get('end', 0) * 1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)
            self.last_end = end
            
            text = result['text']
            print(f"{beg:.0f} {end:.0f} {text}", flush=True, file=sys.stderr)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost", help="Whisper server host")
    parser.add_argument("--port", type=int, default=8000, help="Whisper server port")
    parser.add_argument("--lang", type=str, default="auto", help="Language code (e.g., 'en', 'fr', 'auto')")
    parser.add_argument("--translate", action="store_true", help="Translate to English")
    parser.add_argument("--use-vad", action="store_true", help="Use Voice Activity Detection")
    parser.add_argument("--min-chunk", type=float, default=1.0, help="Minimum audio chunk size in seconds")
    args = parser.parse_args()

    client = WhisperClient(args.host, args.port, args.lang, args.translate, args.use_vad)
    await client.connect()

    audio_processor = AudioProcessor(client, args.min_chunk)
    
    stream = sd.InputStream(
        samplerate=SAMPLING_RATE,
        channels=CHANNELS,
        dtype='float32',
        blocksize=BLOCKSIZE,
        callback=audio_processor.process_audio
    )

    print("Recording audio. Press Ctrl+C to stop.")

    try:
        with stream:
            while True:
                await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Recording stopped by the user.")
    finally:
        stream.close()
        await client.close()

if __name__ == "__main__":
    SAMPLING_RATE = 16000
    CHANNELS = 1
    BLOCKSIZE = 1024

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program terminated by user.")