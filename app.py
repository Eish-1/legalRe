import os
import streamlit as st
import random
import time
import base64
# from legalre_main import LegalRe # Old import
from src.legalre_main import LegalRe # Corrected import path
from langchain_groq import ChatGroq
# from langchain_community.embeddings import HuggingFaceEmbeddings # Deprecated
from langchain_huggingface import HuggingFaceEmbeddings # New import
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain.schema import HumanMessage
import uuid
import logging

#This page implements the streamlit UI
# Set page configuration
st.set_page_config(page_title="LegalRe", page_icon="logo/logo.png", layout="wide")

# Custom CSS for better UI
def add_custom_css():
    """Function for a beautiful streamlit UI"""
    custom_css = """
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        .st-chat-input {
            border-radius: 15px;
            padding: 10px;
            border: 1px solid #ddd;
            margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        .stButton > button {
            background-color: #0066cc;
            color: white;
            font-size: 16px;
            border-radius: 20px;
            padding: 10px 20px;
            margin-top: 5px;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #0052a3;
        }
        .st-chat-message-assistant {
            background-color: #f7f7f7;
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .st-chat-message-user {
            background-color: #d9f0ff;
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .chat-input-container {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #f0f0f0;
            padding: 20px;
            box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
            display: flex;
            gap: 10px;
        }
        .chat-input {
            flex-grow: 1;
        }
        .st-title {
            font-family: 'Arial', sans-serif;
            font-weight: bold;
            color: #333;
            display: flex;
            align-items: center;
            gap: 15px;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .logo {
            width: 40px;
            height: 30px;
        }
        .st-sidebar {
            background-color: #f9f9f9;
            padding: 20px;
        }
        .st-sidebar header {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .st-sidebar p {
            font-size: 14px;
            color: #666;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

add_custom_css()
##Below Code implementation is tha main functioanlity in building the streamlit application
# Title with Logo
logo_path = "logo/logo.png"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    st.markdown(f"""
    <div class="st-title">
        <img src="data:image/png;base64,{encoded_image}" alt="LegalRe Logo" class="logo">
        <span>LegalRe - An AI Legal Assistant </span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="st-title">
        <span>LegalRe - Your Legal Assistant 📖</span>
    </div>
    """, unsafe_allow_html=True)

# Sidebar improvements
st.sidebar.header("About LegalRe")
st.sidebar.markdown("""
**LegalRe** is a free, open-source AI legal assistant that helps answer legal questions based on provided documents.

_Disclaimer_: This tool is in its pilot phase, and responses may not be 100% accurate.
""")

# Function to configure logging verbosity
def configure_logging(verbose=False):
    # Set the base level
    base_level = logging.INFO if verbose else logging.WARNING
    
    # Configure root logger
    logging.basicConfig(
        level=base_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Set specific modules to warning or higher to reduce noise
    logging.getLogger('langchain').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Keep LegalRe's own logs at the selected verbosity
    logging.getLogger('legalre').setLevel(base_level)

# Add to the sidebar (after the about section):
with st.sidebar:
    st.markdown("---")
    verbose_logging = st.sidebar.checkbox("Verbose logging", value=False, help="Show detailed processing logs in the terminal")
    # Configure logging based on user preference
    configure_logging(verbose_logging)

load_dotenv()

#random thread id for session
# Ensure session state for thread_id if needed across reruns, 
# but for now, generating per run might be fine for basic history.
if 'thread_id' not in st.session_state:
     st.session_state.thread_id = str(uuid.uuid4())
thread_id = st.session_state.thread_id

# Load API key
groq_api_key = os.getenv('GROQ_API_KEY')

# Check if the Groq API key is available
if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop() # Stop execution if key is missing

# Caching functions
@st.cache_resource(show_spinner=False)
def get_llm(api_key):
    with st.spinner("Initializing language model..."):
        logging.info("Initializing Groq LLM...")
        return ChatGroq(
            model="deepseek-r1-distill-llama-70b",
            temperature=0.7,
            groq_api_key=api_key
        )

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    with st.spinner("Loading embedding model..."):
        EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
        logging.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

@st.cache_resource(show_spinner=False)
def get_vector_store(_embeddings):
    with st.spinner("Connecting to document database..."):
        CHROMA_DB_DIR = "chroma_db_legal_bot_part1"
        logging.info(f"Loading vector store from: {CHROMA_DB_DIR}")
        if not os.path.isdir(CHROMA_DB_DIR):
            st.error(f"ChromaDB directory not found at '{CHROMA_DB_DIR}'. Please run the embedding script first.")
            st.stop()
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=_embeddings
        )
        logging.info(f"Loaded vector store with {vector_store._collection.count()} documents.")
        return vector_store

@st.cache_resource # Cache the main RAG class instance
def get_legalre_instance(_llm, _embeddings, _vector_store):
    print("Initializing LegalRe instance...")
    return LegalRe(_llm, _embeddings, _vector_store)

# Use cached functions for initialization
llm = get_llm(groq_api_key)
embeddings = get_embedding_model()
vector_store = get_vector_store(embeddings)
law = get_legalre_instance(llm, embeddings, vector_store)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    role = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(message["content"])

# Chat input prompt fixed at the bottom
st.markdown("<div class='chat-input-container'>", unsafe_allow_html=True)
# User Input
prompt = st.chat_input("Have a legal question? Let's work through it.")

st.markdown("</div>", unsafe_allow_html=True)

if prompt:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate answer from LLM
    query = prompt
    try:
        # Invoke the conversational chain
        result = law.conversational(query, thread_id)
        final_response = result # Assuming result is the direct answer string now

    except Exception as e:
        st.error(f"An error occurred while processing your request: {e}")
        final_response = "Sorry, I encountered an error. Please try again."
        # Log the error for debugging
        print(f"Error during conversational invocation: {e}")

    # Assistant's response
    def response_generator(response_text):
        # Simplified response generator - directly yield the final answer
        for word in response_text.split():
            yield word + " "
            time.sleep(0.02) # Slightly faster typing effect

    # Display assistant response in chat message container
    # Get the container object first
    chat_container_obj = st.chat_message("assistant") 
    # Use the obtained object as the context manager
    with chat_container_obj: 
        # Use write_stream for smoother output
        # Check if final_response is not None and is a string
        if isinstance(final_response, str):
                # Use the container object directly
                response = chat_container_obj.write_stream(response_generator(final_response)) 
                # Add assistant response to chat history ONLY IF IT'S A STRING
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
                # Handle cases where the response might not be a string (e.g., error object, None)
                error_message = f"Received unexpected response format: {type(final_response)}"
                print(error_message)
                # Use the container object directly
                chat_container_obj.error(error_message) 
                st.session_state.messages.append({"role": "assistant", "content": "Error: Received unexpected response format."})
