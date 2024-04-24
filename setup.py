#pip install streamlit langchain langchain_community langchain_openai faiss-cpu transformers wikipedia

import streamlit as st
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
import os
import langchain
import streamlit as st
os.environ["OPENAI_API_KEY"]='sk-proj-3elbhRpY0vRpr8e7LEKDT3BlbkFJEotEcjYkidAm0K8uxHp8'

def setup_wikipedia():
    wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki_query_runner = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)
    return wiki_query_runner


def setup_arxiv():
    arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    arxiv_query_runner = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)
    return arxiv_query_runner

def setup_vector_store():
    loader = WebBaseLoader("https://docs.smith.langchain.com/")
    documents = loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
    vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
    retriever=vectordb.as_retriever()
    retriever_tool=create_retriever_tool(retriever,"langsmith_search",
                      "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!")
    return retriever_tool


def setup_openai_model():
    llm = ChatOpenAI(model='gpt-4', temperature=0)
    return llm
def setup_openai_tools():
    tools=[setup_wikipedia(),setup_arxiv(),setup_vector_store()] 
    return tools
def agent_setup():
    tools=setup_openai_tools()
    prompt = hub.pull("hwchase17/openai-functions-agent")
    prompt.messages
    agent=create_openai_tools_agent(setup_openai_model(),tools,prompt)
    return agent

def invoke_query(prompt):
    tools=setup_openai_tools()
    #print(prompt)
    agent_executor=AgentExecutor(agent=agent_setup(),tools=tools,verbose=False)
    result=agent_executor.invoke({"input":prompt})
    
    return result['output']
def invoke_query_noRAG(prompt):
    llm = ChatOpenAI(model='gpt-3.4', temperature=0)
    result=llm.invoke(prompt)
    return result.content
