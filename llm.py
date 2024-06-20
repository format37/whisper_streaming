import os
from openai import OpenAI

def llm_request(system_content, user_content):

    # Set your API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key is not set. You need to set the OPENAI_API_KEY environment variable this way:\nexport OPENAI_API_KEY=yourkey")
    client = OpenAI()

    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    )
    return response.choices[0].message.content

def main():
    system_content = "You are a helpful assistant."
    user_content = "Who won the world series in 2020?"
    message = llm_request(system_content, user_content)
    print(message)
    print('Done')

if __name__ == "__main__":
    main()
