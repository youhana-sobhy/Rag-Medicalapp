"""Minimal example: turn a piece of text into an embedding vector with Voyage AI."""

import voyageai
from dotenv import load_dotenv

load_dotenv()
client = voyageai.Client()


def generate_embedding(text, model="voyage-3-large", input_type="query"):
    result = client.embed([text], model=model, input_type=input_type)
    return result.embeddings[0]


if __name__ == "__main__":
    vector = generate_embedding("What is retrieval-augmented generation?")
    print(f"Dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
