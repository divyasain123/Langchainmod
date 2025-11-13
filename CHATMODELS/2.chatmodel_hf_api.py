from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # ✅ publicly available Inference API model
    task="text-generation",
    temperature=0.5,
    max_new_tokens=10)

model = ChatHuggingFace(llm=llm)
result = model.invoke("Who won the Women's cricket World Cup 2025?")
print(result.content)
