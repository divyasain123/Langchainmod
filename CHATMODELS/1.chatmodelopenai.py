from langchain import chatopenai # open ai is inherting from base chat openai-which is inherting from base chat model(kind of mother class-for every chat model like google,anthropic etc) 
from dotenv import load_dotenv 

load_dotenv()
chatopenai(model='gpt-3.5-turbo-instruct')
