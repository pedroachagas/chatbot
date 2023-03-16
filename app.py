# Import modules from llama_index and langchain
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import os
import streamlit as st

def construct_index(directory_path, api_key):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define LLM (ChatGPT gpt-3.5-turbo)
    llm_predictor = LLMPredictor(llm=OpenAI(api_key=api_key, temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.save_to_disk('index.json')

    return index

def ask_me_anything(question, api_key):
    if not os.path.isfile('index.json'):
        st.markdown('Por favor, construa um índice primeiro.')
        return

    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    llm = OpenAI(api_key=api_key, temperature=0, model_name="gpt-3.5-turbo", max_tokens=256)
    response = index.query(question, response_mode="compact", llm=llm)

    st.markdown(f"**ChatGPCela**: {response.response}")

def main():
    st.set_page_config(page_title="ChatGPCela", page_icon=":smiley:")
    st.title('ChatGPCela')
    
    # Sidebar for index building button and API key toggle
    with st.sidebar:
        st.subheader("Configurações")
        use_user_api_key = st.checkbox('Usar sua chave de API do OpenAI')
        if use_user_api_key:
            api_key = st.text_input('Digite sua chave de API do OpenAI')
        else:
            api_key = st.secrets.openai_key

        if st.button('Construir Índice'):
            directory_path = 'textdata'
            construct_index(directory_path, api_key)
            st.markdown('Índice construído com sucesso!')

    question = st.text_input('O que você gostaria de saber?')

    if st.button('Perguntar'):
        ask_me_anything(question, api_key)

if __name__ == "__main__":
    main()