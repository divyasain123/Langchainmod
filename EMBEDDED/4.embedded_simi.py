from langchain_openai import OpenAIEmbeddings 
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()

embedding=OpenAIEmbeddings(model="text-embedding-3-small",dimensions=300)
documents = [
   " Meryl Streep: A chameleon of the screen, she is widely regarded as one of the greatest actors of all time due to her remarkable versatility and unparalleled command of accents.",
"Marlon Brando: Considered the definitive method actor, he revolutionized American cinema with his raw, intense performances in films like The Godfather and A Streetcar Named Desire.",
"Denzel Washington: Known for his powerful screen presence and moral complexity, he consistently delivers commanding and dignified performances, earning multiple Academy Awards along the way.",
"Tom Hanks: Affectionately known as America's Dad, he is cherished for his everyman appeal and ability to bring warmth and depth to a wide array of iconic characters.",
"Daniel Day-Lewis: A master of immersion, he is famous for his extreme dedication to method acting, often staying in character for the entire duration of a film's production to deliver transformative performances."
]

query = "tell me about Tom Hanks"
doc_embeddings=embedding.embed_documents(documents)
query_embedding=embedding.embed_query(query)

score=cosine_similarity([query_embedding],doc_embeddings)[0] # always pass query in as a list and documnet is alredy a list of lists-- [0] to get first row

index,score=sorted(list(enumerate(score)),key=lambda x:x[1])[-1] #get the last item with highest score

print(query)
print(documents[index])

