import os 
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    response = client.embeddings.create(input = text , model ="text-embedding-3-small")
    return response.data[0].embedding

def get_embeddings_batch(text):
    response = client.embeddings.create(input = text , model ="text-embedding-3-small")
    return [item.embedding for item in response.data]