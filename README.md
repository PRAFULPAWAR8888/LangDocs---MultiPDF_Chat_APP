# 📚 MultiPDF Chat App

## 🔍 Overview

The **MultiPDF Chat App** is a Python-based application that allows users to **interact with multiple PDF documents using natural language**. Users can upload one or more PDFs and ask questions related to their content. The application processes the documents, understands their context using embeddings, and generates accurate answers using a language model.

⚠️ The app only answers questions **based on the uploaded PDFs**.

---

## 🏗️ Project Architecture

![Project Architecture](PDF-LangChain.jpg)

**Architecture Explanation:**

1. PDFs are uploaded via the Streamlit UI
2. Text is extracted from PDFs
3. Text is split into overlapping chunks
4. Embeddings are generated using HuggingFace models
5. FAISS stores vectors for fast similarity search
6. OpenAI LLM generates answers using relevant chunks
7. Chat history is maintained for conversational context

---

## 🧠 How the Application Works

### 1️⃣ PDF Text Extraction

* PDFs are read using **PyPDF2**
* Text is extracted page by page

### 2️⃣ Text Chunking

* Extracted text is split into smaller chunks
* Chunking helps in efficient embedding and retrieval
* Overlap ensures context continuity

### 3️⃣ Embedding Generation

* Uses **HuggingFace sentence-transformers**
* Converts text chunks into numerical vectors

### 4️⃣ Vector Storage (FAISS)

* Embeddings are stored in a **FAISS vector database**
* Enables fast semantic similarity search

### 5️⃣ Conversational Retrieval

* User questions are converted into embeddings
* Most relevant chunks are retrieved
* Chat history is preserved for context-aware responses

### 6️⃣ Response Generation

* **OpenAI GPT model** generates answers
* Responses are strictly based on PDF content

---

## 🛠️ Tech Stack & Tools

| Tool / Library             | Purpose                         |
| -------------------------- | ------------------------------- |
| **Python**                 | Core programming language       |
| **Streamlit**              | Web UI                          |
| **PyPDF2**                 | PDF text extraction             |
| **LangChain**              | LLM orchestration               |
| **HuggingFace Embeddings** | Text vectorization              |
| **FAISS**                  | Vector database                 |
| **OpenAI GPT**             | Answer generation               |
| **dotenv**                 | Environment variable management |

---

## 📂 Project Structure

```
├── app.py
├── htmlTemplates.py
├── requirements.txt
├── .env
├── assets/
│   └── project_architecture.png
├── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone <repository-url>
cd multipdf-chat-app
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set OpenAI API Key

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_secret_api_key
```

---

## ▶️ Usage

Run the application using Streamlit:

```bash
streamlit run app.py
```

### Steps:

1. Upload one or more PDF files from the sidebar
2. Click **Process**
3. Ask questions related to the uploaded PDFs
4. Receive accurate, context-aware responses

---

## ✨ Key Features

* Chat with **multiple PDFs**
* Conversational memory support
* Fast semantic search with FAISS
* Clean and interactive UI
* Context-aware answers

---

## 📌 Limitations

* Only answers questions related to uploaded PDFs
* Requires an active OpenAI API key
* Performance depends on PDF quality and size

---

## 🚀 Future Improvements

* Support for local LLMs
* PDF summary generation
* Source citation for answers
* File persistence across sessions

---

## 👨‍💻 Author

**Praful Pawar**
Aspiring Data Scientist | ML & AI Enthusiast

---

⭐ If you find this project helpful, give it a star!
