from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm_chatgpt = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = hub.pull("rlm/rag-prompt")
generation_chain = prompt | llm_chatgpt | StrOutputParser()