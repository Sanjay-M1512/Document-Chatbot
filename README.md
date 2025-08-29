
# 📄 PDF to Pinecone with Gemini Backend (Flask)

This project provides a **Flask backend** to:
1. Upload a PDF file
2. Split it into chunks efficiently
3. Store the embeddings in **Pinecone Vector Database**
4. Ask natural language questions against the uploaded PDFs
5. Retrieve relevant chunks and generate **human-readable answers** using **Gemini LLM**.

---

## 🚀 Features
- PDF upload and chunking
- Semantic search with Pinecone
- Query answering with Gemini (Google Generative AI)
- REST APIs for integration with frontend (React, Postman, etc.)

---

## 🛠️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-repo/pdf-pinecone-gemini.git
cd pdf-pinecone-gemini
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Environment Variables

Create a `.env` file in the root folder and add:

```ini
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
PINECONE_INDEX=your_pinecone_index_name

GOOGLE_API_KEY=your_gemini_api_key
```

---

## ▶️ Run the Flask Server
```bash
python app.py
```

The server will start on:
```
http://127.0.0.1:5000
```

---

## 📡 API Endpoints

### 1. Upload PDF
**Endpoint:**
```
POST /upload_pdf
```

**Request (Form-Data in Postman):**
- Key: `file`
- Value: (attach your PDF)

**Response:**
```json
{ "message": "PDF processed and stored in Pinecone successfully" }
```

---

### 2. Ask a Question
**Endpoint:**
```
POST /ask
```

**Request (Raw JSON in Postman):**
```json
{ "query": "What is the policy about loan repayment?" }
```

**Response:**
```json
{
  "answer": "The loan repayment policy requires ...",
  "relevant_chunks": ["chunk1...", "chunk2..."]
}
```

---

## ✅ Example Flow in Postman

1. Call `/upload_pdf` → Upload your PDF → Stored in Pinecone
2. Call `/ask` → Send natural language query → Get Gemini-generated answer

---

## 📦 Dependencies
- Flask
- Pinecone
- PyPDF2
- LangChain
- Google Generative AI (Gemini)

---

## 📝 Notes
- You must have valid **Pinecone** and **Gemini API keys**.
- The backend is modular: `pdf_to_pinecone_local.py` handles PDF → Chunks → Pinecone, while `app.py` runs Flask APIs.

---

👩‍💻 Built by Sanjay with ❤️
