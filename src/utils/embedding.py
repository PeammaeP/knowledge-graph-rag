import os
from openai import OpenAI

open_ai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def get_embedding(texts):
    response = open_ai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    return list(map(lambda n: n.embedding, response.data))

def get_model_stream(system_message, user_message):
    stream = open_ai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        stream=True,
    )
    return stream
