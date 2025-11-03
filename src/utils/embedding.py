import os
from openai import OpenAI

open_ai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def embedding(texts):
    response = open_ai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    return list(map(lambda n: n.embedding, response.data))
