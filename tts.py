from pathlib import Path
from openai import OpenAI
import os
import time
import random
from playsound import playsound
import asyncio

async def speech_synthesis(text, model, voice, speed=1.0):
    # Speed from 0.25 to 4.0
    # Set your API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key is not set. You need to set the OPENAI_API_KEY environment variable this way:\nexport OPENAI_API_KEY=yourkey")

    client = OpenAI()
    # https://platform.openai.com/docs/guides/text-to-speech
    speech_file_path = Path(__file__).parent / "speech.mp3"
    
    time_start = time.time()
    response = client.audio.speech.create(
    model=model,
    voice=voice,
    speed=speed,
    input=text,
    )
    response.stream_to_file(speech_file_path)
    time_end = time.time()
    print(f"Time elapsed: {time_end - time_start} seconds")

    # Play audio
    playsound(str(speech_file_path))
    print('Speech synthesis done!')

async def main():
    voices = [
    {
        "id": "alloy",
        "male": False,
        "name": "Елена"
    },
    {
        "id": "echo",
        "male": True,
        "name": "Иван"
    },
    {
        "id": "fable",
        "male": False,
        "name": "Татьяна"
    },
    {
        "id": "onyx",
        "male": True,
        "name": "Павел"
    },
    {
        "id": "nova",
        "male": False,
        "name": "Ольга"
    },
    {
        "id": "shimmer",
        "male": False,
        "name": "Ирина"
    }
    ]

    # Choose random voice
    voice = random.choice(voices)

    text = f"Вас приветствует сервисная служба Айсберг Оператор Татьяна Как я могу вам помочь?"
    # Read text from text.txt
    # with open('text.txt', 'r') as file:
    #     text = file.read().replace('\n', '')
    # text = 'Привет, это тест.'

    model = 'tts-1' # models (tts-1, tts-1-hd)
    # voice_id = voice["id"]
    voice_id = "shimmer"
    print(f"model: {model}, voice: {voice_id}, name: {voice['name']}")
    speed = 1.1
    
    await speech_synthesis(text, model, voice_id, speed)

if __name__ == "__main__":
    asyncio.run(main())
