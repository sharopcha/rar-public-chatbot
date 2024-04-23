import asyncio  # This is a sample Python script.

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel
from openai import OpenAI
from langchain.chains import LLMChain
from langchain.memory import VectorStoreRetrieverMemory

import os
# os.environ['OPENAI_API_KEY']=""
# os.environ["LANGCHAIN_API_KEY"] = ""
# os.environ["LANGCHAIN_TRACING_V2"] = ""
# os.environ["LANGCHAIN_PROJECT"] = ""

# OpenAI API
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"), #you can put the key here directy
)

def load_and_split_data():
    loader = TextLoader('rar-information.txt', encoding='utf-8')
    data = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    return all_splits


def get_retriver():
    collection = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
    splits = load_and_split_data()
    if(collection._collection.count() == 0):
        # Add to vectorDB
        vectorstore = Chroma.from_documents(documents=splits,
                                            persist_directory="./chroma_db",
                                            embedding=OpenAIEmbeddings(),
                                            )
        return vectorstore.as_retriever()
    return collection.as_retriever()

template = """You are helpful chat assistant at Romanian Auto Register. Answer user question based on the given context.
User may ask question in romanian or english language answer based on the question language.
If you dont know the answer, kindly say that you dont know answer and let them to contact RAR.
{context}

Question: {question}
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI(verbose=True, model="gpt-4")
retriever = get_retriver()
# RAG chain
chain = (
    {'context': retriever, "question": RunnablePassthrough()}
    | prompt
    | model.bind(stop=["\nAnswer:"])
    | StrOutputParser()
)


# async def run():
#     chunks = []
#     async for chunk in chain.astream("Ce tip de documente am nevoie pentru eliberare civ?"):
#         chunks.append(chunk)
#         print(chunk, end="", flush=True)
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     # print(chain.invove())
#     asyncio.run(run())
