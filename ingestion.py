from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
load_dotenv()

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# 1. Define the source URLs
url_list = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# 2. INGESTION: Scrape the web pages
# WebBaseLoader goes to each URL and scrapes the text. 
# .load() executes the fetch and returns a list of "Document" objects for each URL.
docs = [WebBaseLoader(url).load() for url in url_list]
# 3. FLATTENING: Combine into one single list
# Because .load() returns a list for *each* URL, 'docs' becomes a list of lists.
# This list comprehension flattens it down into one single, continuous list of Documents.
doc_list = [item for sublist in docs for item in sublist]
# 4. CHUNKING: Set up the text splitter
# Large texts are too big for AI models to read all at once. 
# RecursiveCharacterTextSplitter breaks the text into smaller chunks (250 tokens each).
# 'from_tiktoken_encoder' ensures the chunk size is measured in AI tokens, not just letters.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
# Apply the text splitter to our flattened list of documents
doc_splitter =text_splitter.split_documents(doc_list)

# 5. VECTOR STORE (Now using the chunks, not the full docs)
# vector_store = Chroma.from_documents(
#     documents=doc_splitter,
#     collection_name="rag-chroma",
#     embedding=OpenAIEmbeddings(),
#     persist_directory="./.chroma"
# )

# 6. STORAGE / RETRIEVAL: Set up the Vector Database
# Chroma is a vector database. This initializes the database, telling it where to save 
# the data locally (./.chroma) and to use OpenAI's model to convert the text into numbers (embeddings).
retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings()
).as_retriever(search_kwargs={"k": 3})