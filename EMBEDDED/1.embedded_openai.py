from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the embeddings model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small")

# Convert text into vector
vector = embeddings.embed_query("Hello, world!")

print(vector)
