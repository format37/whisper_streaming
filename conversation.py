#!/usr/bin/env python3
from whisper_online import *
import sys
import argparse
import os
import logging
import numpy as np
import sounddevice as sd
import asyncio
from asyncio import Queue
from llm import llm_request
from tts import speech_synthesis
import time


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

# Keep only the warmup-file argument
parser.add_argument("--warmup-file", type=str, dest="warmup_file", 
        help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")

# options from whisper_online
add_shared_args(parser)
args = parser.parse_args()
# Print args
print(args)

set_logging(args, logger, other="")

# setting whisper object by args 
SAMPLING_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 1024

size = args.model
language = args.lan  # Use the language argument from add_shared_args
asr, online = asr_factory(args)
min_chunk = args.min_chunk_size

# Warm up Whisper
if args.warmup_file:
    if os.path.isfile(args.warmup_file):
        a = load_audio_chunk(args.warmup_file, 0, 1)
        asr.transcribe(a)
        logger.info("Whisper is warmed up.")
    else:
        logger.critical("The warm up file is not available.")
        sys.exit(1)
else:
    logger.warning("Whisper is not warmed up. The first chunk processing may take longer.")

# LLM configuration
# system_content = "Вы - Татьяна, оператор сервисного центра Айсберг. Ваша задача выяснить доволен ли клиент недавним ремонтом и записать отзыв клиента. Если понадобится завершить разговор, используйте слово hangup. Не забывайте что ваш пол - женский, имейте это ввиду когда говорите от вашего имени."
system_content = "Вы - Татьяна, оператор колл центра. Вы звоните клиенту, что бы выяснить оставлял ли он заказ на ремонт. Если оставлял, напишите ЗАЯВКА, если нет, ОШИБКА. Для завершения разговора напишите hangup. Если это заявка, после завершения разговора клиент будет переведен на оператора для приема заявки."
llm_messages = [
    {"role": "system", "content": system_content}
]

is_speech_active = True
start_time = time.time()  # Store the start time of the program
speech = {
    "start_time": 0,
    "end_time": 0,
    "speech_end_time": 0,
    "text": ""
}

class AudioProcessor:
    def __init__(self, online_asr_proc, min_chunk):
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk
        self.last_end = None
        self.audio_buffer = []
        self.loop = asyncio.get_event_loop()
        self.task_queue = Queue()
        self.current_task = None

    def process_audio(self, indata, frames, time_info, status):
        if status:
            print(status)
        
        if is_speech_active:
            self.audio_buffer.append(indata.copy())
            if len(self.audio_buffer) * BLOCKSIZE >= self.min_chunk * SAMPLING_RATE:
                audio_chunk = np.concatenate(self.audio_buffer)
                self.audio_buffer = []
                self.process_chunk(audio_chunk)

    def process_chunk(self, audio_chunk):
        self.online_asr_proc.insert_audio_chunk(audio_chunk)
        o = online.process_iter()
        if o[0] is not None:
            self.format_and_process_output(o)

    def format_and_process_output(self, o):
        global start_time, speech
        
        beg, end = o[0]*1000, o[1]*1000
        if self.last_end is not None:
            beg = max(beg, self.last_end)
        self.last_end = end
        
        text = o[2]
        print(f"{beg:.0f} {end:.0f} {text}", flush=True, file=sys.stderr)
        
        current_time = (time.time() - start_time) * 1000  # Current time in ms relative to start
        speech_end_relative = (speech["speech_end_time"] - start_time) * 1000  # Convert speech_end_time to relative time in ms
        
        logger.debug(f"Current time: {current_time:.2f}ms, Speech end: {speech_end_relative:.2f}ms, Segment end: {end:.2f}ms")
        
        # Add a small buffer (100ms) and use current_time for comparison
        if current_time > speech_end_relative + 100:
            speech["text"] += text.replace("\n", " ")
            speech["text"] = speech["text"].replace("продолжение","").strip()
            speech["text"] = speech["text"].replace("следует","").strip()
            if speech["text"]:
                print(f'Full text: {speech["text"]}')
                logger.debug(f'Queueing LLM task with text: {speech["text"]}')
                self.loop.create_task(self.process_llm_task(end, speech["text"]))
            else:
                logger.debug("process_llm_task canceled: No text in this segment")
        else:
            logger.debug(f"Skipping text before speech synthesis end: {text}")

    async def process_llm_task(self, time_end, user_text):
        task = self.loop.create_task(call_llm(time_end, user_text))
        await self.task_queue.put(task)
        
        if self.current_task is None:
            self.current_task = self.loop.create_task(self.process_queue())

    async def process_queue(self):
        while True:
            task = await self.task_queue.get()
            try:
                await task
            except asyncio.CancelledError:
                print("Task was cancelled")
            finally:
                self.task_queue.task_done()
            
            if self.task_queue.empty():
                self.current_task = None
                break

async def call_llm(time_end, user_text):
    global is_speech_active
    global llm_messages
    global speech
    
    model = "gpt-4o"
    # model = "gpt-3.5-turbo-instruct"
    
    llm_messages.append({"role": "user", "content": user_text})
    text = await llm_request(llm_messages, model)
    llm_messages.append({"role": "assistant", "content": text})
    
    if speech["start_time"] < time_end:
        print(f"[>> LLM >>] {text}")
        # Speech synthesis
        model = 'tts-1'
        voice_id = "shimmer"
        speed = 1.4
        # Mute microphone
        logger.debug("Muting microphone for speech synthesis")
        is_speech_active = False
        await speech_synthesis(text.replace("hangup",""), model, voice_id, speed)
        print('# Speech synthesis done!')
        speech["speech_end_time"] = time.time()  # Update speech end time
        logger.debug(f"Speech synthesis ended at {speech['speech_end_time']} (global time)")
        logger.debug("Unmuting microphone after speech synthesis")
        # Unmute microphone
        is_speech_active = True
        # Clean text
        speech["text"] = ""
    else:
        print(f'# llm interrupted. start_time: {speech["start_time"]}, time_end: {time_end}')
        await asyncio.sleep(1)
    
    # Unmute microphone
    is_speech_active = True
    if "hangup" in text:
        print(f"Conversation ended. LLM answer: {text}")
        # Close tasks
        for task in asyncio.all_tasks():
            task.cancel()
        sys.exit(0)

async def main():
    audio_processor = AudioProcessor(online, min_chunk)
    
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
        # Cancel any ongoing tasks
        if audio_processor.current_task:
            audio_processor.current_task.cancel()
        # Cancel all queued tasks
        while not audio_processor.task_queue.empty():
            task = await audio_processor.task_queue.get()
            task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program terminated by user.")
