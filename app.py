# Import modules from llama_index and langchain
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import os
import streamlit as st

# Adapted based on LlamaIndex documentation https://gpt-index.readthedocs.io/en/latest/index.html
# and Dan Shipper's work https://www.lennysnewsletter.com/p/i-built-a-lenny-chatbot-using-gpt

def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define LLM (ChatGPT gpt-3.5-turbo)
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.save_to_disk('index.json')

    return index

def ask_me_anything(question):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(question, response_mode="compact")

    st.markdown(f"You asked: **{question}**")
    st.markdown(f"Bot says: **{response.response}**")

def main():
    st.title('Ask Me Anything')
    question = st.text_input('What would you like to ask?')
    if st.button('Ask'):
        ask_me_anything(question)

if __name__ == "__main__":
    openai_key = st.secrets.openai_key
    os.environ["OPENAI_API_KEY"] = openai_key
    construct_index('textdata')
    main()