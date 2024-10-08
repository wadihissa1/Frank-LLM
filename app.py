from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from flask import g
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import traceback
import logging
import requests
from bs4 import BeautifulSoup
import pdfkit
from urllib.parse import urljoin
import logging
import asyncio
import aiohttp
import tempfile
import whisper
from pytube import YouTube
import re
from pytube.innertube import _default_clients
from pytube import cipher
import importlib
def lazy_import(module_name):
    return importlib.import_module(module_name)

def lazy_import_pytube_youtube():
    pytube = lazy_import("pytube")
    return pytube.YouTube


# Fix for Pytube throttling issue
_default_clients["ANDROID"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["ANDROID_EMBED"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS_EMBED"]["context"]["clientVersion"] = "19.08.35"
_default_clients["IOS_MUSIC"]["context"]["client"]["clientVersion"] = "6.41"
_default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID_CREATOR"]

def get_throttling_function_name(js: str) -> str:
    function_patterns = [
        r'a\.[a-zA-Z]\s*&&\s*\([a-z]\s*=\s*a\.get\("n"\)\)\s*&&\s*'
        r'\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])?\([a-z]\)',
        r'\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])\([a-z]\)',
    ]
    for pattern in function_patterns:
        regex = re.compile(pattern)
        function_match = regex.search(js)
        if function_match:
            if len(function_match.groups()) == 1:
                return function_match.group(1)
            idx = function_match.group(2)
            if idx:
                idx = idx.strip("[]")
                array = re.search(
                    r'var {nfunc}\s*=\s*(\[.+?\]);'.format(
                        nfunc=re.escape(function_match.group(1))),
                    js
                )
                if array:
                    array = array.group(1).strip("[]").split(",")
                    array = [x.strip() for x in array]
                    return array[int(idx)]
    raise Exception("Throttling function name not found")

cipher.get_throttling_function_name = get_throttling_function_name

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
MODEL = "gpt-3.5-turbo"

# Initialize model
if "gpt" in MODEL:
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=MODEL)
    embeddings = OpenAIEmbeddings()
else:
    model = Ollama()
    embeddings = LlamaEmbeddings()

template = """
You are Frank, the assistant at Antonine University. Answer the question based on the context below.
If you can't answer the question, reply "I'm sorry, I don't have the information you're asking for right now. Could you please provide more details or try asking in a different way?".

Context: {context}

Question: {question}

Assistant Name: Frank
"""
prompt = PromptTemplate.from_template(template)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check the dimension of your embeddings
def get_embedding_sample():
    for _ in range(3):  # Retry mechanism
        try:
            return embeddings.embed_documents(["test"])[0]
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Error getting embedding sample: {str(e)}. Retrying...")
    raise ConnectionError("Failed to get embedding sample after multiple retries.")

embedding_sample = get_embedding_sample()
embedding_dimension = len(embedding_sample)
logging.debug(f"Embedding dimension: {embedding_dimension}")

# Create or connect to an existing index
index_name = "antonine-university"
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=PINECONE_ENV
        )
    )

# Initialize the index
index = pc.Index(index_name)

def sanitize_embedding(embedding):
    embedding = np.array(embedding, dtype=np.float32)
    embedding = np.nan_to_num(embedding, nan=0.0, posinf=1.0, neginf=-1.0)
    embedding = np.clip(embedding, -1.0, 1.0)
    return embedding.tolist()

def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return (embedding / norm).tolist()

def is_valid_embedding(embedding):
    return all(np.isfinite(embedding)) and all(-1.0 <= x <= 1.0 for x in embedding)

def fetch_main_content(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch content from {url}. Status code: {response.status_code}")
    
    soup = BeautifulSoup(response.content, "html.parser")
    content = soup.find('div', {'id': 'mw-content-text'})  # Wikipedia-specific
    if not content:
        raise Exception("Could not find the main content on the page.")
    
    return str(content)

# Specify the path to wkhtmltopdf
path_to_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'

config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)

def save_content_to_pdf(html_content, pdf_path):
    pdfkit = lazy_import("pdfkit")  # Lazy import when the function is called
    html_file_path = "content.html"
    with open(html_file_path, "w", encoding="utf-8") as file:
        file.write(html_content)
    
    options = {
        'no-images': '',
        'disable-external-links': '',
        'disable-javascript': '',
        'enable-local-file-access': '',
        'quiet': ''
    }
    
    pdfkit.from_file(html_file_path, pdf_path, configuration=config, options=options)
    
def save_contents_to_pdf(contents, pdf_path):
    html_content = "<br>".join(contents)
    
    # Save HTML to file first
    html_file_path = "content.html"
    with open(html_file_path, "w", encoding="utf-8") as file:
        file.write(html_content)
    
    # Convert HTML file to PDF with wkhtmltopdf
    options = {
        'no-images': '',  # Do not load images
        'disable-external-links': '',  # Disable external links
        'disable-javascript': '',  # Disable JavaScript
        'enable-local-file-access': '',  # Enable local file access
        'no-stop-slow-scripts': '',  # Do not stop slow scripts (if JavaScript is enabled)
        'load-error-handling': 'ignore',  # Ignore load errors for missing content
        'quiet': ''  # Suppress wkhtmltopdf warnings
    }
    
    try:
        pdfkit.from_file(html_file_path, pdf_path, configuration=config, options=options)
    except IOError as e:
        logging.error(f"IOError during PDF conversion: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"General error during PDF conversion: {str(e)}")
        raise

def get_conversation_context():
    if 'conversation_context' not in g:
        g.conversation_context = []
    return g.conversation_context

async def fetch_content(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def fetch_all_urls(base_url):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Handling both absolute and relative URLs
    urls = [
        (a['href'] if a['href'].startswith('http') else urljoin(base_url, a['href']))
        for a in soup.find_all('a', href=True)
        if a['href'].startswith(base_url) or a['href'].startswith('/')
    ]
    return urls

async def fetch_content_from_urls(urls):
    tasks = [fetch_content(url) for url in urls]
    return await asyncio.gather(*tasks)

@app.route('/fetch_and_process', methods=['POST'])
def fetch_and_process():
    data = request.json
    url = data.get('url', '')
    if not url:
        return jsonify({'error': 'URL is required'}), 400

    try:
        # Fetch and process content from the specific URL
        html_content = fetch_main_content(url)
        
        # Save the content to a PDF
        pdf_path = "website_content.pdf"
        save_content_to_pdf(html_content, pdf_path)
        logging.debug(f"Saved content to PDF: {pdf_path}")

        # Load and split the PDF document
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()

        # Insert documents into Pinecone with unique keys
        for i, page in enumerate(pages):
            embedding = embeddings.embed_documents([page.page_content])[0]
            embedding = normalize_embedding(embedding)
            embedding = sanitize_embedding(embedding)
            if is_valid_embedding(embedding):
                # Use a unique key combining URL and page number
                unique_key = f"{hash(url)}_page_{i}"
                logging.debug(f"Inserting embedding with key {unique_key}: {embedding}")
                index.upsert([(unique_key, embedding, {'page_content': page.page_content})])

        return jsonify({'message': 'Web Page content has been processed and indexed successfully'}), 200

    except Exception as e:
        logging.error(f"Error processing website content: {str(e)}")
        return jsonify({'error': f"Error processing website content: {str(e)}"}), 500

@app.route('/fetch_urls', methods=['POST'])
def fetch_urls():
    data = request.json
    base_url = data.get('url', '')
    if not base_url:
        return jsonify({'error': 'URL is required'}), 400

    try:
        # Fetch all URLs from the base URL
        urls = asyncio.run(fetch_all_urls(base_url))
        logging.debug(f"Fetched URLs: {urls[:5]}")  # Print only the first 5 URLs for brevity

        # Limit the number of URLs to process
        urls = urls[:5]  # Adjust this number as needed

        # Fetch content from each URL
        contents = asyncio.run(fetch_content_from_urls(urls))
        logging.debug(f"Fetched contents from URLs")

        # Save the content to a PDF
        pdf_path = "website_contents.pdf"
        save_contents_to_pdf(contents, pdf_path)
        logging.debug(f"Saved contents to PDF: {pdf_path}")

        # Load and split the PDF document
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()

        # Insert documents into Pinecone
        for i, page in enumerate(pages):
            embedding = embeddings.embed_documents([page.page_content])[0]
            embedding = normalize_embedding(embedding)
            embedding = sanitize_embedding(embedding)
            logging.debug(f"Inserting embedding for page_{i}: {embedding}")
            if is_valid_embedding(embedding):
                # Create a unique key using a hash of the base URL and the index (to ensure uniqueness across multiple URLs)
                unique_key = f"{hash(urls[i // len(pages)])}_{i}"  # Using i // len(pages) to get the corresponding URL for the page
                index.upsert([(unique_key, embedding, {'page_content': page.page_content})])

        return jsonify({'message': 'URLs have been processed, content saved to PDF, and embeddings stored in Pinecone successfully'}), 200

    except Exception as e:
        logging.error(f"Error processing URLs: {str(e)}")
        return jsonify({'error': f"Error processing URLs: {str(e)}"}), 500

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    try:
        file = request.files['file']
        if file and file.filename.endswith('.pdf'):
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                file.save(temp_file.name)
                temp_file_path = temp_file.name

            # Load the PDF document using PyPDFLoader
            loader = PyPDFLoader(temp_file_path)
            pages = loader.load_and_split()

            # Insert documents into Pinecone with unique keys
            for i, page in enumerate(pages):
                embedding = embeddings.embed_documents([page.page_content])[0]
                embedding = normalize_embedding(embedding)
                embedding = sanitize_embedding(embedding)
                if is_valid_embedding(embedding):
                    # Use a unique key combining file name (hashed) and page number
                    unique_key = f"{hash(file.filename)}_page_{i}"
                    logging.debug(f"Inserting embedding with key {unique_key}: {embedding}")
                    index.upsert([(unique_key, embedding, {'page_content': page.page_content})])

            return jsonify({'message': 'PDF content has been processed and indexed successfully'}), 200

        return jsonify({'error': 'Invalid file format'}), 400

    except Exception as e:
        logging.error(f"Error processing uploaded PDF: {str(e)}")
        return jsonify({'error': f"Error processing uploaded PDF: {str(e)}"}), 500

last_context = None
last_topic = None

@app.route('/ask', methods=['POST'])
def ask_question():
    conversation_context = get_conversation_context()
    data = request.json
    question = data.get('question', '')

    if not question:
        return jsonify({'error': 'Question is required'}), 400

    # Predefined responses for common questions
    predefined_responses = {
        "what is your name": "My name is Frank.",
        "who are you": "I am Frank, the assistant at Antonine University.",
        "what do you know": "I am Frank, the assistant at Antonine University. I can help you with information regarding tuition fees and other related details."
    }

    # Check for predefined responses
    if question.lower() in predefined_responses:
        return jsonify({'response': predefined_responses[question.lower()]}), 200

    try:
        query_embedding = embeddings.embed_documents([question])[0]
        query_embedding = normalize_embedding(query_embedding)
        query_embedding = sanitize_embedding(query_embedding)
        logging.debug(f"Query embedding: {query_embedding}")

        if not is_valid_embedding(query_embedding) or len(query_embedding) != embedding_dimension:
            raise ValueError("Invalid embedding or dimension mismatch")

    except Exception as e:
        logging.error(f"Error creating or validating embedding: {str(e)}")
        return jsonify({'response': "There was an error processing your question. Please try again."}), 500

    try:
        result = index.query(vector=[query_embedding], top_k=5, include_metadata=True)
        logging.debug(f"Pinecone query result: {result}")

        context_list = []
        for match in result.get('matches', []):
            metadata = match.get('metadata', {})
            if 'page_content' in metadata:
                context_list.append(metadata['page_content'])
            else:
                logging.warning(f"Missing 'page_content' in match: {match}")

        context = "\n".join(context_list) if context_list else ""

    except Exception as e:
        logging.error(f"Error querying Pinecone: {traceback.format_exc()}")
        return jsonify({'response': "There was an error fetching information. Please try again later."}), 500

    if not context:
        # If no direct context is found, suggest similar options or give a more specific fallback response
        try:
            similar_results = index.query(vector=[query_embedding], top_k=3, include_metadata=True)
            similar_suggestions = []
            for match in similar_results.get('matches', []):
                if match['score'] > 0.65:  # Further lowered threshold
                    metadata = match.get('metadata', {})
                    if 'page_content' in metadata:
                        similar_suggestions.append(metadata['page_content'])

            if similar_suggestions:
                suggestion_text = "Based on the information provided, the fees vary depending on the program and number of credits. Here are some specific programs and their fees that might help you:\n"
                suggestion_text += "\n".join([f"- {suggestion}" for suggestion in similar_suggestions])
                suggestion_text += "\n\nFor more specific details, you can refer to the tuition fees provided for each faculty or department. If you have a particular program in mind, please let me know."
                return jsonify({'response': suggestion_text}), 200

            # Provide a more specific fallback response with query refinement
            return jsonify({
                'response': "Based on the information provided, the fees vary depending on the program and number of credits. For specific details, you can refer to the list of tuition fees provided for each faculty or department. If you have a specific program in mind, please let me know."
            }), 200

        except Exception as e:
            logging.error(f"Error suggesting similar questions: {traceback.format_exc()}")
            return jsonify({'response': "There was an error processing your request. Please try again later."}), 500

    # Add the current question and context to the conversation history
    conversation_context.append(f"Question: {question}\nContext: {context}")

    # Keep only the last 5 interactions to keep context manageable
    conversation_history = "\n\n".join(conversation_context[-5:])

    # Refine the prompt to include the conversation history
    formatted_prompt = prompt.format(context=conversation_history, question=question)

    try:
        # Get the response from the model
        response = model.invoke(formatted_prompt)
    except Exception as e:
        logging.error(f"Error invoking model: {traceback.format_exc()}")
        return jsonify({'response': "There was an error generating a response. Please try again later."}), 500

    # Check if response is redundant
    if response.content.strip() in [res.split("\n")[-1].strip() for res in conversation_context[-5:]]:
        response.content = "I don't know"

    # Update conversation context with the current question and the model's response
    conversation_context.append(f"Answer: {response.content}")

    return jsonify({'response': response.content})



@app.route('/test', methods=['GET'])
def test_pinecone():
    query = "What is Hollywood going to start doing?"
    try:
        query_embedding = embeddings.embed_documents([query])[0]
        query_embedding = normalize_embedding(query_embedding)
        query_embedding = sanitize_embedding(query_embedding)
        logging.debug(f"Test query embedding: {query_embedding}")

        # Check for embedding validity and dimension consistency
        if not is_valid_embedding(query_embedding) or len(query_embedding) != embedding_dimension:
            raise ValueError("Invalid embedding or dimension mismatch")

    except Exception as e:
        logging.error(f"Error creating or validating embedding: {str(e)}")
        return jsonify({'response': f"Error creating or validating embedding: {str(e)}"}), 500

    try:
        result = index.query(queries=[query_embedding], top_k=3, include_metadata=True)
        matches = [{"score": match['score'], "content": match['metadata']['page_content']} for match in result['matches']]
        return jsonify({'matches': matches})
    except Exception as e:
        # Log the exception traceback
        error_trace = traceback.format_exc()
        logging.error(f"Error querying Pinecone: {error_trace}")
        return jsonify({'response': f"Error querying Pinecone: {str(e)}"}), 500

@app.route('/reset', methods=['POST'])
def reset_context():
    global conversation_context
    conversation_context = []
    return jsonify({'message': 'Conversation context has been reset'})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.json
    youtube_url = data.get("url")
    if not youtube_url:
        return jsonify({"error": "URL is required"}), 400
    
    try:
        # Download and transcribe the audio
        YouTube = lazy_import_pytube_youtube()
        whisper = lazy_import("whisper")

        youtube = YouTube(youtube_url)
        audio = youtube.streams.filter(only_audio=True).first()
        whisper_model = whisper.load_model("base")
        with tempfile.TemporaryDirectory() as tmpdir:
            file = audio.download(output_path=tmpdir)
            transcription = whisper_model.transcribe(file, fp16=False)["text"].strip()
            with open("transcription.txt", "w") as file:
                file.write(transcription)
        
        # Load and split the transcription
        loader = TextLoader("transcription.txt")
        text_documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        documents = text_splitter.split_documents(text_documents)

        # Embed and send to Pinecone using unique keys
        for i, document in enumerate(documents):
            embedding = embeddings.embed_documents([document.page_content])[0]
            embedding = normalize_embedding(embedding)
            embedding = sanitize_embedding(embedding)
            if is_valid_embedding(embedding):
                # Use a unique key combining YouTube URL (hashed) and chunk number
                unique_key = f"{hash(youtube_url)}_chunk_{i}"
                logging.debug(f"Inserting embedding with key {unique_key}: {embedding}")
                index.upsert([(unique_key, embedding, {'page_content': document.page_content})])
        
        return jsonify({"message": "Transcription and embedding completed"}), 200

    except Exception as e:
        logging.error(f"Error during transcription: {str(e)}")
        return jsonify({"error": f"Error during transcription: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='172.20.10.3', port=5000, debug=False)
