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

# REMEMBER TO ADD YOUR API KEYS HERE

# ----------

# TODO: Fix build_chain function
# TODO: Write generic function to build custom langchain tools (i.e., summarise, suggest, search-chat)
# TODO: Write functions to save and load information from memory
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

def compute_cost(tokens, engine):
    
    model_prices = {"text-davinci-003": 0.02, 
                    "gpt-3.5-turbo": 0.002, 
                    "gpt-4": 0.03}
    model_price = model_prices[engine]
    
    cost = (tokens / 1000) * model_price

    return cost

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

def read_perma4(path: str, return_test_answers=False):
    '''
    Function to read perma4 dataset.
    params:
        path: str to json file
        return_test_answers: bool, default False
    return:
        data: pandas dataframe
    '''
    data = pd.read_json(path)
    
    questions = data['questions']
    
    if return_test_answers:
        return data
    else:
        return questions
    
def memory_to_pandas(memory_path: str):
    '''
    Function to convert memory to pandas dataframe.
    params:
        memory_path: path to memory json file
    return:
        df: pandas dataframe
    '''
    with open(memory_path) as f:
        data = json.load(f)    
    
    return data
    
def run_agent_from_profile(agent_profile: dict, 
                           query: str):
    '''
    Function to build agent from memory using lanchain library.
    params:
        agent_profile: dict
        memory: pandas dataframe
    return:
        agent: Langchain agent object
    '''
    agent = None 
    
    name = agent_profile['name']
    agent_type = agent_profile['agent_type']
    personality = agent_profile['personality']
    knowledge = agent_profile['knowledge']
    tools = agent_profile['tools']
    description = agent_profile['description']
    max_tokens = agent_profile['max_tokens']
    temperature = agent_profile['temperature']
    
    engine = build_llm(model_name='text-davinci-003', 
                       max_tokens=max_tokens, temperature=temperature)
    llm_tools = load_tools(tools, llm=engine)
    
    prompt_template = '''You are {name}. {description}. You have a {personality} personality and {knowledge} knowledge.'''
    prompt = PromptTemplate(input_variables=[name, description, query, personality, knowledge],
                            template=prompt_template)
    
    if agent_type == "zeroShot":
        print(f"Building (zeroShot) {name} agent...")
        zeroShot_agent = initialize_agent(tools=llm_tools, llm=engine, 
                                 agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        
        # Build prompt before running to specify custom agent's prompt using its description, personality, and knowledge.
        ## 
        
        zeroShot_chain = LLMChain(llm=engine,
                                  prompt=prompt)
                                  
        agent_response = zeroShot_chain.run(query)
        
        
        #agent = zeroShot_agent
    
    elif agent_type == "selfAskSearch":
        print(f"Building (selfAskSearch) {name} agent...")
        search = GoogleSerperAPIWrapper()
        # intermediate answer tool
        self_tools = [Tool(name="Intermediate Answer",
                        func=search.run,
                        description="useful for when you need to ask with search")]
        
        sealfAsk_agent = initialize_agent(tools=self_tools, llm=engine, 
                                 agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
        agent = sealfAsk_agent
    
    return agent_response


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
