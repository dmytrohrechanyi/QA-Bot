import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
import os
import tempfile
import faiss
import pickle
from datetime import timedelta

def initialize_session_state():

    # Check if the key 'history' is present in the session_state, if not, it will be added
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Analogous for 'generated' and 'past'
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! I know a lot about your study program. Ask me! 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

# Funktion für den Chat-Dialog
def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

# Function to display chat history
def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()    

    previous_chat_history = len(st.session_state['generated'])
    # Creation of the website/user interface
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Your question:", placeholder="Ask your question here", key='input')
            submit_button = st.form_submit_button(label='Send') #, disabled=st.session_state['submitted'])

        # Process after submitting the form
        if submit_button and len(user_input) > 0:
            st.session_state['submitted'] = True
            with st.spinner('Generating response...'):
                # Generate answer
                output = conversation_chat(user_input, chain, st.session_state['history'])

           # Updating session state data
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            
            # st.session_state['submitted'] = False
    now_chat_history = len(st.session_state['generated'])
    if (st.session_state['generated'] and now_chat_history > previous_chat_history) or previous_chat_history == 1:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                # Quelle: https://www.dicebear.com/styles/adventurer/
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="open-peeps")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

def create_conversational_chain(vector_store):
    # Initialization llm
    llm = LlamaCpp(
    streaming = True,
    model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.5,
    top_p=1,
    verbose=True,
    n_ctx=4096
)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

@st.cache_resource
def load_vector_store(save_path="vector_store"):
    index = faiss.read_index(os.path.join(save_path, "faiss_index"))

    with open(os.path.join(save_path, "text_chunks.pkl"), "rb") as f:
        text_chunks = pickle.load(f)
    
    with open(os.path.join(save_path, "embeddings_model.pkl"), "rb") as f:
        embeddings_model = pickle.load(f)
    
    return index, text_chunks, embeddings_model

@st.cache_resource
def load_conversational_chain(_index, _text_chunks, _embeddings_model):
    vector_store = FAISS.from_documents(_text_chunks, embedding=_embeddings_model)

    llm = LlamaCpp(
        streaming=True,
        model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        temperature=0.5,
        top_p=1,
        verbose=True,
        n_ctx=4096
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )
    return chain

def main():
    # Initialization session state
    initialize_session_state()

    # Streamlit Application Title
    st.title("Chat with your study regulations :books:")
    # Initialisierung Streamlit
    st.sidebar.title("Document Processing")
    # select options - trained model or a new model
    options = ["Trained Model", "New Model"]
    uploaded_files = []
    selected_option = st.sidebar.radio("Your choice: ", options)
    if selected_option == "Trained Model":
        st.sidebar.empty()
    else:
        uploaded_files = st.sidebar.file_uploader("Upload your study regulations", accept_multiple_files=True)

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)
        
        with st.spinner('Generating a new model...'):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
            text_chunks = text_splitter.split_documents(text)

            # embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                            model_kwargs={'device': 'cpu'})

            # vector store
            vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

            # the chain object
            chain = create_conversational_chain(vector_store)
    else:
        with st.spinner('Wait for loading the saved model...'):
            save_path = "vector_store"
            index, text_chunks, embeddings_model = load_vector_store(save_path)
            chain = load_conversational_chain(index, text_chunks, embeddings_model)
    
    display_chat_history(chain)

if __name__ == "__main__":
    main()