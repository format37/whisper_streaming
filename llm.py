import os
import asyncio
from openai import OpenAI

async def llm_request(messages, model):
    # Set your API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key is not set. You need to set the OPENAI_API_KEY environment variable this way:\nexport OPENAI_API_KEY=yourkey")
    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

async def main():
    system_content = "You are a helpful assistant."
    user_content = "Who won the world series in 2020?"
    model = "gpt-4o"
    messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    message = await llm_request(messages, model)
    print(message)
    print('Done')

if __name__ == "__main__":
    asyncio.run(main())
