import ollama
import streamlit as st
from icon import page_icon
from openai import OpenAI

from utilities.utils import process_pdf, chat_stream,chat_pdf

st.set_page_config(
    page_title="Chat playground",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

def extract_model_names(models_info: list) -> tuple:
    """
    Extracts the model names from the models information.

    :param models_info: A dictionary containing the models' information.

    Return:
        A tuple containing the model names.
    """

    return tuple(model["name"] for model in models_info["models"])

def pdf_toggle_fun():
   
    if st.session_state.pdf_toggle:
        st.session_state.pdf_processed = False
        st.session_state.vector_db = None
        
     
def main():
    
    """
    The main function that runs the application.
    """
    
    st.markdown("""
        <style>
            .header {
                font-size: 36px;
                font-weight: bold;
                color: #1e90ff;
                text-align: center;
                margin-top: 20px;
            }
            .subheader {
                font-size: 18px;
                color: #4b4b4b;
                text-align: center;
                margin-top: 10px;
                font-style: italic;
            }
        </style>
        <div class="header">💬 Chat and Question by Uploading PDF 📄</div>
        <div class="subheader">Ask questions based on the content of your PDF document</div>
    """, unsafe_allow_html=True)

    page_icon("💬")
    st.subheader("Ollama Playground", divider="red", anchor=False)

    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    )
    
    models_info = ollama.list()
    
    available_models = extract_model_names(models_info)
    
    if available_models:
        selected_model = st.selectbox(
            "Pick a model available locally on your system ↓", available_models
        )
    else:
        st.warning("You have not pulled any model from Ollama yet!", icon="⚠️")
        if st.button("Go to settings to download a model"):
            st.page_switch("pages/03_⚙️_Settings.py")
            
            
    # Streaming mode toggle
    st.toggle("Enable PDF Question-Answer Mode",key='pdf_toggle' , on_change=pdf_toggle_fun)
     
    message_container = st.container(height=500, border=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "vector_db" not in st.session_state:
         st.session_state.vector_db = None
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
        
   
    
    #Displaying Previous Messages:
    for message in st.session_state.messages: 
        avatar = "🤖" if message["role"] == "assistant" else "😎"
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
         
    
        
    # Input area for prompt or file upload
    col1, col2 = st.columns([3, 1])
   
    
    if st.session_state.pdf_toggle:
        with col1:
            prompt = st.chat_input("Enter a prompt here...")
        with col2:
            uploaded_file = st.file_uploader("Upload a PDF", type="pdf") 
            
         # Process PDF file if uploaded and not yet processed
        if uploaded_file and not st.session_state.pdf_processed:
            st.session_state.vector_db = process_pdf(uploaded_file,message_container)
            st.session_state.pdf_processed = True
            message_container.success("PDF processed and vector database created successfully.")

            with message_container.chat_message("assistant", avatar="🤖"):
                 st.markdown("The PDF has been processed. You can now ask questions based on its content.")
        
    
        # Handle prompt input with retrieval and RAG processing
        if prompt and st.session_state.vector_db is not None and uploaded_file:
                chat_pdf(prompt,message_container,selected_model)
   
   
    else:
        prompt = st.chat_input("Enter a prompt here...")
        if prompt:
            chat_stream(prompt,client,selected_model,message_container)
        
            
   
                
   
            
    # st.write(st.session_state)
    
    
        
            
   
         
    
            
if __name__ == "__main__":
    main()
    
    