import requests
import os
from groq import Groq
from langchain_groq import ChatGroq
import json
from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import joblib
import nest_asyncio  # noqa: E402
nest_asyncio.apply()
import os
import agentops

import json
from ai21 import AI21Client
from ai21.models import ImprovementType

import dotenv

dotenv.load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AGENTOPS_API_KEY = os.environ.get('AGENTOPS_API_KEY')
agentops.init(os.getenv("AGENTOPS_API_KEY"))

import os
import logging
from os import environ, path
from typing import List

import chromadb
from dotenv import load_dotenv
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_core.vectorstores import VectorStoreRetriever
# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_get_response(response):
    client = AI21Client(
    # This is the default and can be omitted
    api_key=os.environ.get("AI21_API_KEY"),
    )
    response1=client.improvements.create(
      text=response,
      types=[
        ImprovementType.FLUENCY,
        ImprovementType.VOCABULARY_SPECIFICITY,
        ImprovementType.VOCABULARY_VARIETY,
        ImprovementType.CLARITY_SHORT_SENTENCES,
        ImprovementType.CLARITY_CONCISENESS
    ])

    return response1


# CONSTANTS =====================================================
EMBED_MODEL_NAME = "jina-embeddings-v2-base-en"
LLM_NAME = "mixtral-8x7b-32768"
LLM_TEMPERATURE = 0.1
CHUNK_SIZE = 8192
DOCUMENT_DIR = "./data/"
VECTOR_STORE_DIR = "./vectorstore/"
COLLECTION_NAME = "collection1"
# ===============================================================

def load_documents() -> List[Document]:
    """Loads the PDF files from the specified directory."""
    try:
        logging.info("Loading documents...")
        documents = DirectoryLoader(path.join(DOCUMENT_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader).load()
        logging.info(f"Documents loaded, total pages: {len(documents)}")
        return documents
    except Exception as e:
        logging.error("Error loading documents", exc_info=True)
        return []

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Splits documents into chunks of specified size."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("jinaai/" + EMBED_MODEL_NAME, cache_dir=environ.get("HF_HOME"))
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_SIZE // 50)
        logging.info("Splitting documents...")
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Document splitting done, {len(chunks)} chunks total.")
        return chunks
    except Exception as e:
        logging.error("Error splitting documents", exc_info=True)
        return []

def create_and_store_embeddings(embedding_model: JinaEmbeddings, chunks: List[Document]) -> Chroma:
    """Creates embeddings for the document chunks and stores them."""
    try:
        vectorstore = Chroma.from_documents(chunks, embedding=embedding_model, collection_name=COLLECTION_NAME,
                                            persist_directory=VECTOR_STORE_DIR)
        logging.info("Vectorstore created.")
        return vectorstore
    except Exception as e:
        logging.error("Error creating vectorstore", exc_info=True)
        return None

def get_vectorstore_retriever(embedding_model: JinaEmbeddings) -> VectorStoreRetriever:
    """Retrieves or creates the vector store."""
    db = chromadb.PersistentClient(VECTOR_STORE_DIR)
    try:
        db.get_collection(COLLECTION_NAME)
        retriever = Chroma(embedding_function=embedding_model, collection_name=COLLECTION_NAME,
                           persist_directory=VECTOR_STORE_DIR).as_retriever(search_kwargs={"k": 3})
    except Exception:
        logging.info("Vectorstore does not exist, creating new one...")
        pdfs = load_documents()
        chunks = chunk_documents(pdfs)
        retriever = create_and_store_embeddings(embedding_model, chunks).as_retriever(search_kwargs={"k": 3})
    return retriever

def create_rag_chain(embedding_model: JinaEmbeddings, llm: ChatGroq) -> Runnable:
    """Creates a Retrieve and Generate chain for document-based querying."""
    template = """Answer the question based only on the following context. Think step by step before providing a detailed answer. I will give you $500 if the user finds the response useful.
    <context>
    {context}
    </context>
    Question: {input}
    """
    prompt = ChatPromptTemplate.from_template(template)
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = get_vectorstore_retriever(embedding_model)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def run_chain(chain: Runnable, query: str):
    """Runs the Retrieve and Generate chain with user input."""
    response = chain.invoke({"input": query})
    for doc in response["context"]:
        logging.info(f"{doc.metadata} | content: {doc.page_content[:20]}...")
    return response["answer"]

def get_response(query: str):
    """Main function to initialize components and run the chain."""
    embedding_model = JinaEmbeddings(jina_api_key=environ.get("JINA_API_KEY"), model_name=EMBED_MODEL_NAME)
    llm = ChatGroq(temperature=LLM_TEMPERATURE, model_name=LLM_NAME)
    chain = create_rag_chain(embedding_model=embedding_model, llm=llm)
    res=run_chain(chain, query)
    return res


def get_query_from_user(prompt_text):
    response = get_response(prompt_text) 
    print("HERE IS RESPONSE:",response)
   # response1=process_get_response(response)
    # final_res=str("Find PII: "+response1)+response
    return response