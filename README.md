# End-to-End Medical Chatbot

A RAG-based (Retrieval-Augmented Generation) medical chatbot that answers health-related questions using medical PDF documents as its knowledge base. Built with LangChain, Pinecone, Groq, and Flask.

## Tech Stack

- **LLM**: Groq (LLaMA 3.3 70B Versatile)
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Vector Store**: Pinecone (serverless, AWS us-east-1)
- **Framework**: LangChain
- **Backend**: Flask
- **Frontend**: HTML, CSS, jQuery

## How It Works

1. Medical PDF documents are loaded from the `data/` directory
2. Documents are split into chunks (500 chars, 20 overlap) using `RecursiveCharacterTextSplitter`
3. Chunks are embedded using HuggingFace sentence-transformers and stored in Pinecone
4. When a user asks a question, the top 3 most similar chunks are retrieved
5. The retrieved context + question are sent to Groq's LLaMA model to generate a concise answer

## Project Structure

```
├── app.py                 # Flask application & RAG chain
├── store_index.py         # Script to index PDFs into Pinecone
├── src/
│   ├── helper.py          # PDF loading, text splitting, embeddings
│   └── prompt.py          # System prompt template
├── templates/
│   └── chat.html          # Chat interface
├── static/
│   └── style.css          # Styling
├── data/                  # Place medical PDF files here
├── setup.py               # Package setup
├── requirements.txt       # Dependencies
└── .env                   # API keys (not tracked)
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/rafi4482/End-to-End-Medical-Chatbot
cd End-to-End-Medical-Chatbot
```

### 2. Create and activate a virtual environment

```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

### 5. Add medical PDFs

Place your medical PDF documents in the `data/` directory.

### 6. Index the documents

```bash
python store_index.py
```

This will load the PDFs, split them into chunks, generate embeddings, and store them in Pinecone.

### 7. Run the application

```bash
python app.py
```

The app will be available at `http://localhost:8080`.
