from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the embeddings model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small")
document = ["delhi is the capital of india",
            "paris is the capital of france"
            "kolkata is the capital of west bengal"]

# Convert text into vector
vector = embeddings.embed_documents(document)

print(str(vector))
