
from langchain_community.document_loaders import UnstructuredPDFLoader
from IPython.display import display as Markdown
from tqdm.autonotebook import tqdm as notebook_tqdm
from tempfile import NamedTemporaryFile
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import streamlit as st 


from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
            
import os




def chat_stream(prompt,client,selected_model,message_container):
            try:
                st.session_state.messages.append(
                    {"role": "user", "content": prompt})

                message_container.chat_message("user", avatar="😎").markdown(prompt)

                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner("model working..."):
                        stream = client.chat.completions.create(
                            model=selected_model,
                            messages=[
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ],
                            stream=True,
                        )
                    # stream response
                    response = st.write_stream(stream)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response})
            except Exception as e:
                st.error(e, icon="⛔️")
                

def chat_pdf(prompt,message_container,selected_model):
    try:
        llm = ChatOllama(model=selected_model)
        
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}""",
            )   
        
    
        
        # RAG prompt
        template = """Answer the question based ONLY on the following context:
            {context}
            Question: {question}
            """

        prompt_template = ChatPromptTemplate.from_template(template)
                # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Retrieve and answer based on vector DB
        retriever = MultiQueryRetriever.from_llm(
                st.session_state.vector_db.as_retriever(),
                llm,
                prompt=QUERY_PROMPT
            )

        # Process question with RAG chain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        message_container.chat_message("user", avatar="😎").markdown(prompt)
        # Run the chain and get response
        with message_container.chat_message("assistant", avatar="🤖"):
            with st.spinner("Model working..."):
                response = chain.invoke( prompt)
                st.markdown(response)
        
        # Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
                st.error(f"Error: {e}", icon="⛔️")
    
                
                
    

# Function to process the PDF and initialize vector DB
def process_pdf(uploaded_file,message_container):
    # # Save uploaded file temporarily
    # with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
    #     temp_file.write(uploaded_file.getbuffer())
    #     temp_file_path = temp_file.name'
    # Save PDF to local path for processing
    local_path = "uploaded_pdf.pdf"
    with open(local_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and split PDF content
    loader = UnstructuredPDFLoader(file_path=local_path)
    with message_container.chat_message("assistant", avatar="🤖"):
        with st.spinner("Extracting text from PDF..."):
            data = loader.load()
            # st.markdown(data[0].page_content)


            # Split content into chunks for embedding
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)

            # Embed chunks into a vector database
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=OllamaEmbeddings(model="nomic-embed-text"),
                collection_name="local-rag"
            )
    return vector_db