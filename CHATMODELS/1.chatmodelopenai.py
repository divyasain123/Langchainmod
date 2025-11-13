from langchain_openai import ChatOpenAI  # open ai is inherting from base chat openai-which is inherting from base chat model(kind of mother class-for every chat model like google,anthropic etc) 
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)

result = model.invoke("Write a 5 line poem on cricket")

print(result.content)