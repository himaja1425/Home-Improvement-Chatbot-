from flask import Flask, render_template, request, jsonify
import openai
import sqlite3
import PyPDF2
import chromadb
from chromadb.config import Settings
import time
import uuid  # Import UUID for generating unique IDs
import os
from dotenv import load_dotenv
app = Flask(__name__)

load_dotenv()


# Initialize OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')
 #'your_openai_api_key_here'  # Replace with your actual API key

# Initialize ChromaDB with a Persistent Client
storage_path = "data/chromadb_storage"
client = chromadb.PersistentClient(path=storage_path)

# Create or get the Chroma collection
def get_faqs_collection():
    try:
        collection = client.get_or_create_collection("faqs")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
    return collection

# Create a lightweight SQLite database
conn = sqlite3.connect('data/leads.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS leads
                  (name TEXT, email TEXT, phone TEXT, service TEXT, action TEXT)''')
conn.commit()

def extract_faqs_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to handle API errors and retries
def create_embedding_with_retry(text, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response['data'][0]['embedding']
        except openai.error.APIError as e:
            print(f"APIError on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e

# Extract FAQs and store them in ChromaDB
def add_faqs_to_chromadb():
    faqs_text = extract_faqs_from_pdf("data/faqs.pdf")
    collection = get_faqs_collection()
    faqs = faqs_text.split("\n\n")
    ids = [str(uuid.uuid4()) for _ in faqs]  # Generate unique IDs for each FAQ
    embeddings = [create_embedding_with_retry(faq) for faq in faqs]
    collection.add(ids=ids, documents=faqs, embeddings=embeddings)
    print("FAQs added to ChromaDB successfully.")

# Call the function to add FAQs to ChromaDB
add_faqs_to_chromadb()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')

    # Handle basic interactions and questions
    if "hello" in user_input.lower() or "hi" in user_input.lower():
        response = "Hi there! I'm here to help you connect with top-rated contractors. How can I assist you today?"
    elif "service" in user_input.lower():
        response = "What type of service do you need? (e.g., landscaping, plumbing) Please provide your name and contact details so I can assist you better."
    elif "cost" in user_input.lower() or "price" in user_input.lower():
        response = fetch_faq_answer(user_input)
    else:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ]
        ).choices[0].message['content']

    return jsonify({"response": response})

def fetch_faq_answer(question):
    collection = get_faqs_collection()
    query_embedding = create_embedding_with_retry(question)
    result = collection.query(query_embeddings=[query_embedding], n_results=1)
    return result['documents'][0] if result['documents'] else "I couldn't find the information you're looking for."

@app.route('/lead', methods=['POST'])
def save_lead():
    data = request.json
    cursor.execute("INSERT INTO leads (name, email, phone, service, action) VALUES (?, ?, ?, ?, ?)",
                   (data['name'], data['email'], data['phone'], data['service'], data['action']))
    conn.commit()
    return jsonify({"message": "Lead saved successfully!"})

@app.route('/generate_completion', methods=['POST'])
def generate_completion():
    user_input = request.json.get('prompt')
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]
    )
    return jsonify({"completion": response.choices[0].message['content']})

@app.route('/generate_image', methods=['POST'])
def generate_image():
    prompt = request.json.get('prompt')
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    return jsonify({"image_url": response['data'][0]['url']})

@app.route('/create_embedding', methods=['POST'])
def create_embedding():
    text = request.json.get('text')
    embedding = create_embedding_with_retry(text)
    return jsonify({"embedding": embedding})

if __name__ == '__main__':
    app.run(debug=True)
