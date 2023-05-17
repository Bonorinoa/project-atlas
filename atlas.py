from utils import (build_agent_from_memory, 
                   build_chain, extract_entities_keywords,
                   build_llm, build_llm_tools)
import streamlit as st
import json
from langchain.prompts import PromptTemplate

# TODO: Fix build_chain function
# TODO: Test building agents from memory
# TODO: Test processing sample user profiles in memory with agents
# TODO: Test digital nudging for simulated scenario
# TODO: Write better prompts/descriptions (template for PromptTemplate, input_variables come from user profile or agent input or user input) for AI profiles in memory

def main():
    st.title("Atlas Intelligence")

    st.sidebar.header("User Profile")
    name = st.sidebar.text_input("Name", "Type Here")
    age = st.sidebar.text_input("Age", "Type Here")
    tastes = st.sidebar.text_input("Tastes", "Type A List Here (separated by comma)")
    occupation = st.sidebar.text_input("Occupation", "Type Here")
    location = st.sidebar.text_input("Location", "Type Here")
    
    with open("test_long_term_memory.json", "r") as f:
        memory = json.load(f)
    
    llm_test = build_llm()
    llm_tools_test = build_llm_tools(tools=["google-serper"])
    
    template_test = '''{prompt}'''
    prompt_test = PromptTemplate(template=template_test, 
                                input_variables=["prompt"])
    
    #chain_test = build_chain(prompt=prompt_test, 
    #                         llm=llm_test)
    
    #chain_tools_test = build_chain(prompt=prompt_test,
    #                               llm=llm_tools_test)
    
    entities, keywords = extract_entities_keywords(text=template_test)
    
    st.sidebar.header("User Input")
    user_query = st.sidebar.text_input("Query", "Type Here")
    
    if st.button("Submit"):
        
        llm_response = llm_test(user_query)
        llm_tools_response = llm_tools_test(user_query)['output']
    
        st.write(f"Entities: {entities}; \nKeywords: {keywords}")
        st.write("----------------------------------------------------")
        st.write(f"LLM no tools:\n {llm_response}")
        st.write("----------------------------------------------------")
        st.write(f"LLM with tools:\n {llm_tools_response}")
    
if __name__ == "__main__":
    main()