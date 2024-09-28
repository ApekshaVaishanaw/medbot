from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
import asyncio
import logging
import requests
from asyncio import new_event_loop, set_event_loop
from concurrent.futures import ThreadPoolExecutor
from twilio.rest import Client
import base64

# Load environment variables
load_dotenv()

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")
YOUR_PHONE_NUMBER = os.environ.get("YOUR_PHONE_NUMBER", "")

# Initialize Twilio Client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Flask app initialization
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pinecone configurations
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-bot"

# Loading the index
docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512,
                            'temperature': 0.8})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs)

# Helper functions for WhatsApp handling
def download_media(media_url: str, file_path: str) -> None:
    response = requests.get(
        media_url,
        auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    )
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
    else:
        logger.error(f"Failed to download media from {media_url}. Status code: {response.status_code}")

async def process_whatsapp_message(incoming_msg: str, media_urls: list) -> None:
    loop = asyncio.get_event_loop()

    if media_urls:
        media_files = []
        for media_url in media_urls:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(media_url))
            download_media(media_url, file_path)
            media_files.append(file_path)

        for file_path in media_files:
            # Process media files if needed
            pass

    with ThreadPoolExecutor() as pool:
        # Use the `qa` model to generate a response
        result = await loop.run_in_executor(
            pool, qa.invoke, {"query": incoming_msg}
        )

    response_message = result["result"]

    try:
        # Send the model's response to the user's WhatsApp number
        twilio_client.messages.create(
            body=response_message,
            from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
            to=f"whatsapp:{YOUR_PHONE_NUMBER}",
        )
        logger.info("Message sent to WhatsApp successfully.")
    except Exception as e:
        logger.error(f"Failed to send message via Twilio: {e}")

# Flask routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa.invoke({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

@app.route("/whatsapp", methods=["POST"])
def whatsapp_message():
    incoming_msg = request.values.get("Body", "").strip()
    media_urls = [request.values.get(f'MediaUrl{i}') for i in range(0, len(request.values)) if request.values.get(f'MediaUrl{i}')]

    logger.info(f"Received message: {incoming_msg}")
    logger.info(f"Media URLs: {media_urls}")

    loop = new_event_loop()
    set_event_loop(loop)
    loop.run_until_complete(process_whatsapp_message(incoming_msg, media_urls))

    return "Message processing initiated", 202

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
