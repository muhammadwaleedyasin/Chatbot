import streamlit as st
import openai
import pandas as pd
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor
import time

load_dotenv()

# Get API key from env variable
openai.api_key = os.getenv('OPENAI_API_KEY')

@st.cache_resource
def load_and_process_files(file_paths, urls):
    documents = []
    
    for file_path in file_paths:
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8')
                text_data = df.to_string(index=False)
                text_file_path = file_path.replace('.csv', '.txt')
                with open(text_file_path, 'w', encoding='utf-8') as f:
                    f.write(text_data)
                loader = TextLoader(text_file_path, encoding='utf-8')
            else:
                continue
            
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Error processing file {file_path}: {str(e)}")
            continue
    
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Error processing URL {url}: {str(e)}")
            continue
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore

@st.cache_resource
def create_conversational_chain(_vectorstore):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer or the answer isn't explicitly stated in the context, say "I don't have information about that in my current knowledge base." Don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        verbose=True,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        output_key="answer"
    )

def get_gpt_response(prompt, context):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            *context,
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

def display_response_animation(response, role):
    st.chat_message(role).markdown("")
    chat_placeholder = st.empty()
    display_text = ""
    for char in response:
        display_text += char
        chat_placeholder.markdown(display_text)
        time.sleep(0.01)  # Adjust for faster/slower typing effect

# You can define file paths and URLs
file_paths = ['','','']
urls = [
    # You can add  URLs here
]

vectorstore = load_and_process_files(file_paths, urls)
conversational_chain = create_conversational_chain(vectorstore)

st.title('Security Help Desk!')

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'untrained_response' not in st.session_state:
    st.session_state.untrained_response = None

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('How can I assist you today?')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    response = conversational_chain({"question": prompt})
    
    assistant_response = response['answer']
    
    if "I don't have information about that in my current knowledge base" not in assistant_response:
        sources = set()
        if 'source_documents' in response:
            for doc in response['source_documents']:
                if 'source' in doc.metadata:
                    sources.add(doc.metadata['source'])
        
        if sources:
            st.markdown(f"**Sources:** {', '.join(sources)}")
    else:
        with ThreadPoolExecutor() as executor:
            gpt_future = executor.submit(get_gpt_response, prompt, st.session_state.messages)
            assistant_response = gpt_future.result()
    
    display_response_animation(assistant_response, 'assistant')
    st.session_state.messages.append({'role': 'assistant', 'content': assistant_response})

    untrained_response = get_gpt_response(prompt, st.session_state.messages)
    st.session_state.untrained_response = untrained_response



with st.sidebar:
    st.header("Untrained Chatbot Response")
    if st.session_state.untrained_response:
        st.markdown(st.session_state.untrained_response)