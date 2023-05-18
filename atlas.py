from utils import (run_agent_from_profile, 
                   build_chain, extract_entities_keywords,
                   build_llm, build_llm_tools, memory_to_pandas)
import streamlit as st
import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# TODO: Fix build_chain function
# TODO: Test processing sample user profiles in memory with agents
# TODO: Test digital nudging for simulated scenario
# TODO: Write better prompts/descriptions (template for PromptTemplate, input_variables come from user profile or agent input or user input) for AI profiles in memory

# load memory globally
memory_path = "test_long_term_memory.json"
memory_df = memory_to_pandas(memory_path)

# Agents
# 1. Well-being coach
coach = memory_df['AI_profiles'][0]

# 2. Journalist
journalist = memory_df['AI_profiles'][1]

# 3. Recommendationg Engine
recommendator = memory_df["AI_profiles"][2]

# 4. Digital Nudger
nudger = memory_df["AI_profiles"][3]

# 5. Report Generator
report_gen = memory_df["AI_profiles"][4]

def main():
    st.title("Atlas Intelligence Demo")

    # user information we want to demo
    st.sidebar.header("User Profile")
    name = st.sidebar.text_input("Name", "Type Here")
    age = st.sidebar.text_input("Age", "Type Here")
    tastes = st.sidebar.text_input("Tastes", "Type A List Here (separated by comma)")
    occupation = st.sidebar.text_input("Occupation", "Type Here")
    location = st.sidebar.text_input("Location", "Type Here")
    
    # build llm no tools
    llm_demo = build_llm()
    
    ## build prompt for main LLM
    template_demo = '''{prompt}'''
    prompt_demo = PromptTemplate(template=template_demo, 
                                input_variables=["prompt"])
    
    ## build chain with llm and prompt
    chain_demo = LLMChain(prompt=prompt_demo, 
                          llm=llm_demo)
    
    #--- ignore for now
    #llm_tools_test = build_llm_tools(tools=["google-serper"])
    
    #chain_test = build_chain(prompt=prompt_test, 
    #                         llm=llm_test)
    
    #chain_tools_test = build_chain(prompt=prompt_test,
    #                               llm=llm_tools_test)
    #---
    
    st.sidebar.header("User Input")
    user_query = st.sidebar.text_input("Query", "What is your purpose?")
    
    entities, keywords = extract_entities_keywords(text=user_query)
    
    if st.button("Submit"):
        
        llm_response = chain_demo.run(user_query)
        #llm_tools_response = llm_tools_test(user_query)['output']
    
        st.write(f"Entities: {entities}; \nKeywords: {keywords}")
        st.write("----------------------------------------------------")
        st.write(f"LLM no tools:\n {llm_response}")
        st.write("----------------------------------------------------")
        #st.write(f"LLM with tools:\n {llm_tools_response}")
    
if __name__ == "__main__":
    main()
    