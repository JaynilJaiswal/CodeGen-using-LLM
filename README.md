# RAG-Based Document Retrieval with Mistral Model and ChromaDB
This project involves building a Retrieval Augmented Generation (RAG) system that processes a large JSON file containing over 590,000 records. The records are embedded using the HuggingFace transformer model and stored in a ChromaDB collection for efficient retrieval.

## Features
- Persistent Storage: Embedding and storing JSON data using ChromaDB.
- Mistral Model Integration: Utilizing the Mistral model for generating embeddings.
- LangChain for Querying: Leveraging LangChain and ChatOllama to handle complex queries.
- Progress Tracking: Real-time tracking of embedding and storage processes.

## Prerequisites
- Python 3.8+
- pip for package management
- virtualenv for environment management (optional but recommended)
- CUDA-supported GPU (optional for faster processing)


## Setting Up Ollama with the Mistral Model

To integrate Ollama with the Mistral model for your RAG project, follow the steps below:

### 1. Install Ollama

First, you'll need to install Ollama. If you haven't already installed it, you can do so by running:
```bash
curl -o- https://ollama.com/install.sh | bash
```

Or, if you're using Homebrew on macOS, you can install it via:
```bash
brew install ollama
```

### 2. Install the Mistral Model
Once Ollama is installed, you can download the Mistral model. Run the following command in your terminal:
```bash
ollama pull mistral
```
This will download and set up the Mistral model, making it available for use with Ollama.

### 3. Configure Your Project
In your project, ensure that you are using the Mistral model with Ollama embeddings. You can configure this in your environment or directly in the script.
Set the `TEXT_EMBEDDING_MODEL` environment variable to `"mistral"`:
```bash
export TEXT_EMBEDDING_MODEL="mistral"
```
Alternatively, you can set this directly in your code where you're initializing the Ollama embeddings:
```bash
embedding = OllamaEmbeddings(model="mistral", show_progress=True)
```

### 4. Install Dependencies
Install the necessary Python packages:
```bash
pip install -r requirements.txt
```
Ensure your `requirements.txt` includes:
```bash
tqdm
transformers
chromadb
torch
langchain
langchain_core
langchain_community
langchain_huggingface
langchain_text_splitters
langchain_ollama
```

## Setup dataset to create persistent vector database
I have used dataset from CMU CoNaLa, the Code/Natural Language Challenge. This dataset was collected by scrawling on StackOverflow and consists of 600k mined samples.
```bash
wget http://www.phontron.com/download/conala-corpus-v1.1.zip
unzip conala-corpus-v1.1.zip
mv conala-corpus/conala-mined.jsonl conala-corpus/conala-mined.json
```

