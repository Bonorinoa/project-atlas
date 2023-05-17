import logging
import json
import pandas as pd
import os
import nltk
import spacy

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.agents import load_tools, initialize_agent, AgentType, Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank, LLMChainExtractor
from langchain.utilities import GoogleSerperAPIWrapper

os.environ["OPENAI_API_KEY"] = ""
os.environ["SERPAPI_API_KEY"] = ""
os.environ["WOLFRAM_ALPHA_APPID"] = ""
os.environ["SERPER_API_KEY"] = ""
# ----------

# TODO: Fix build_chain function
# TODO: Write function to build agent from memory given agent's name
# TODO: Write function to process perma4 responses
# TODO: Write generic function to build custom langchain tools (i.e., summarise, suggest, search-chat)
# TODO: Write functions to build, save and load information from memory
# TODO: Implement asynchronous versions of llm/chain builders

# interestingly the es_core_news_sm dictionary in spanish is better at identifying entities than the english one
# python -m spacy download en_core_web_sm <- run in terminal to download the english dictionary (es_core_news_sm for spanish)
nlp = spacy.load("en_core_web_sm")

# entities and keywords from query  
def extract_entities_keywords(text):
    '''
    Function to extract entities and keywords from a text using spacy library.
    params:
        text: str
    return:
        entities: list
        keywords: list    
    '''
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    keywords = [token.lemma_ for token in doc if token.is_stop == False and token.is_punct == False]
    return entities, keywords

def build_llm(max_tokens=260, 
              temperature=0.6, 
              provider="openai"):
    '''
    Function to build a LLM model using lanchain library. 
    Default model is text-davinci-003 for OpenAI provider, but you can change it to any other model depending on the provider's models.
    note that for chat models you would set provider = "ChatOpenAI" for example.
    params:
        max_tokens: int, default 260
        temperature: float, default 0.6
        provider: str, default 'openai'
    return:
        llm: Langchain llm object
    '''
    llm = None
    
    if provider == "openai":
        llm = OpenAI(model_name='text-davinci-003', temperature=temperature, max_tokens=max_tokens)
    elif provider == "ChatOpenAI":
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature, max_tokens=max_tokens)
    
    return llm

def build_llm_tools(tools: list,
                    max_tokens=260, 
                    temperature=0.6, 
                    provider="openai"):
    '''
    Function to build agent (llm + tools) using lanchain library.
    params:
        tools: list of tools
        model_name: str, default 'text-davinci-003'
        max_tokens: int, default 260
        temperature: float, default 0.6
        provider: str, default 'openai'
    return:
        agent: Langchain agent object
    '''
    agent = None
    if provider == "openai":
        llm = build_llm(temperature=temperature, max_tokens=max_tokens, provider=provider)

        tools = load_tools(tools, llm=llm)

        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        
    return agent

def build_chain(prompt: PromptTemplate,
                llm):
    '''
    Function to build LLMChain using lanchain library.
    params:
        prompt: str
        llm: Langchain llm object
    return:
        chain: Langchain LLMChain object
    '''
    chain = LLMChain(llm=llm, 
                     prompt=prompt)
    
    return chain

def build_agent_from_memory(ai_profile):
    '''
    Function to build agent from memory using lanchain library.
    params:
        ai_profile: dict
    return:
        agent: Langchain agent object
    '''
    agent = None
    
    
    
    return agent
