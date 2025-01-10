import os
import pymongo
import logging
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import time
import traceback

# Load environment variables and LLM setup
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")

# MongoDB setup
client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['pdf_pipeline']
collection = db['documents']

# Logger setup
logging.basicConfig(filename='pipeline_errors.log', level=logging.ERROR)

# Token count and rate limiting
token_limit_per_minute = 6000
max_tokens_per_request = 4096
tokens_used_this_minute = 0
start_time = time.time()

def estimate_token_count(text):
    return len(text) // 4

def reset_token_count():
    global tokens_used_this_minute, start_time
    elapsed_time = time.time() - start_time
    if elapsed_time >= 60:
        tokens_used_this_minute = 0
        start_time = time.time()

# Function to extract PDF metadata
def extract_pdf_metadata(file_name, file_size, page_count, length_classification):
    metadata = {
        'file_name': file_name,
        'size': file_size,
        'page_count': page_count,
        'length_classification': length_classification
    }
    collection.insert_one(metadata)
    return metadata

# Classify PDF length based on the number of pages
def classify_pdf_length(page_count):
    if page_count <= 10:
        return 'short'
    elif page_count <= 30:
        return 'medium'
    else:
        return 'long'

# Summarization logic for short, medium, and long documents
def summarize_pdf(docs, length_classification):
    global tokens_used_this_minute
    reset_token_count()

    full_text = ' '.join(doc.page_content for doc in docs)
    total_tokens = estimate_token_count(full_text)
    summaries = []

    try:
        if length_classification in ['short', 'medium']:
            if total_tokens > max_tokens_per_request:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_tokens_per_request // 2,
                    chunk_overlap=50
                )
                chunks = text_splitter.split_text(full_text)
                for chunk in chunks:
                    doc_chunk = Document(page_content=chunk)
                    template = """Write a concise and short summary of the following speech in 300 words.\nSpeech={text}"""
                    prompt = PromptTemplate(input_variables=["text"], template=template)
                    chain = load_summarize_chain(llm, chain_type='stuff', prompt=prompt, verbose=True)
                    summaries.append(chain.run([doc_chunk]))
            else:
                doc_full = Document(page_content=full_text)
                template = """Write a concise and short summary of the following speech in 200 words.\nSpeech={text}"""
                prompt = PromptTemplate(input_variables=["text"], template=template)
                chain = load_summarize_chain(llm, chain_type='stuff', prompt=prompt, verbose=True)
                summaries.append(chain.run([doc_full]))
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
            final_documents = text_splitter.split_documents(docs)
            chunk_template = """Write a concise and short summary of the following speech.\nSpeech:{text}\nSummary:"""
            map_reduce_prompt = PromptTemplate(input_variables=["text"], template=chunk_template)
            final_prompt = """Provide a final summary with these important points. Add a motivational title, start the precise summary with an introduction, and provide the summary in bullet points for the speech.\nSpeech:{text}"""
            final_prompt_template = PromptTemplate(input_variables=["text"], template=final_prompt)
            summary_chain = load_summarize_chain(
                llm=llm, 
                chain_type="map_reduce",
                map_prompt=map_reduce_prompt,
                combine_prompt=final_prompt_template,
                verbose=False
            )
            summaries.append(summary_chain.run(final_documents))
    except Exception as e:
        logging.error(f"Error summarizing document: {traceback.format_exc()}")
        summaries.append("Error summarizing this document.")

    return " ".join(summaries)

# Custom keyword extraction using LLM
def custom_keyword_extraction(docs):
    try:
        all_text = ' '.join(doc.page_content for doc in docs)
        prompt = f"Extract the top 20 important keywords from the following text:\n\n{all_text}"
        response = llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else "Error extracting keywords."
    except Exception as e:
        logging.error(f"Error extracting keywords: {traceback.format_exc()}")
        return "Error extracting keywords."

# Process a single PDF
def process_pdf(file_name, documents, pdf_bytes):
    try:
        page_count = len(documents)
        file_size = len(pdf_bytes)
        length_classification = classify_pdf_length(page_count)
        metadata = extract_pdf_metadata(file_name, file_size, page_count, length_classification)
        summary = summarize_pdf(documents, length_classification)
        keywords = custom_keyword_extraction(documents)
        collection.update_one(
            {'file_name': metadata['file_name']},
            {'$set': {'summary': summary, 'keywords': keywords}}
        )
        return summary, keywords
    except Exception as e:
        logging.error(f"Error processing {file_name}: {traceback.format_exc()}")
        return None, None

# Streamlit UI for uploading and processing PDFs
st.title("PDF Summarization & Keyword Extraction")
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        temppdf = f"./temp_{uploaded_file.name}"
        with open(temppdf, "wb") as file:
            file.write(uploaded_file.getvalue())
        try:
            loader = PyPDFLoader(temppdf)
            documents = loader.load()
            file_bytes = uploaded_file.read()
            summary, keywords = process_pdf(uploaded_file.name, documents, file_bytes)
            if summary and keywords:
                st.subheader(f"Summary for {uploaded_file.name}")
                st.write(summary)
                st.subheader(f"Keywords for {uploaded_file.name}")
                st.write(keywords)
            else:
                st.error(f"Error processing {uploaded_file.name}. Check logs for details.")
        finally:
            os.remove(temppdf)

# Display MongoDB entries
st.subheader("Stored Document Data in MongoDB")
if st.button('Load Stored Data'):
    documents = list(collection.find({}))
    if documents:
        for doc in documents:
            st.write(f"**File Name:** {doc['file_name']}")
            st.write(f"**Size (bytes):** {doc['size']}")
            st.write(f"**Page Count:** {doc['page_count']}")
            st.write(f"**Length Classification:** {doc['length_classification']}")
            st.write(f"**Summary:** {doc.get('summary', 'Not available')}")
            keywords = doc.get('keywords', [])
            if isinstance(keywords, list):
                st.write(f"**Keywords:** {', '.join(keywords)}")
            else:
                st.write(f"**Keywords:** {keywords}")
            st.write("---")
    else:
        st.write("No documents found in MongoDB.")
