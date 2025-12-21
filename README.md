# 🔧 KEITH Running Floor II Installation Assistant

A RAG-powered (Retrieval-Augmented Generation) chatbot that helps technicians and installers with questions about the KEITH Running Floor II® unloading system installation.

Built for **Keith Manufacturing Company** using OpenAI, Pinecone, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green.svg)

## 🚀 Features

- **Intelligent Q&A**: Ask questions about installation procedures and get accurate, context-aware answers
- **Source Citations**: Every answer includes references to specific pages in the manual
- **Conversation Memory**: Maintains context across multiple questions
- **Professional UI**: Clean, branded interface with Keith Manufacturing styling
- **Safety Awareness**: Highlights important warnings and safety guidelines

## 📋 Prerequisites

Before you begin, you'll need:

1. **Python 3.9+** installed on your machine
2. **OpenAI API Key** - Get one at [platform.openai.com](https://platform.openai.com)
3. **Pinecone Account** - Sign up at [pinecone.io](https://www.pinecone.io) (free tier works!)
4. **The Installation Manual PDF** - `keith_running_floor_ii_installation_manual.pdf`

## 🛠️ Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/running-floor-rag.git
cd running-floor-rag
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Pinecone

1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Create a new **Starter** (free) or **Standard** project
3. Create an index with these settings:
   - **Name**: `running-floor-manual`
   - **Model**: Select `text-embedding-3-small` (OpenAI)
   - **Dimensions**: 1536
   - **Metric**: cosine
4. Note your **API Key** and **Environment** (e.g., `us-east-1-aws`)

### Step 5: Configure Environment Variables

**For local development:**

```bash
cp .env.example .env
```

Edit `.env` with your actual keys:

```
OPENAI_API_KEY=sk-your-actual-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENV=us-east-1-aws
PINECONE_INDEX=running-floor-manual
```

**For Streamlit Cloud deployment:**

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` with your keys.

### Step 6: Ingest the PDF

Place your PDF in the project directory, then run:

```bash
python ingest.py
```

This will:
- Load and chunk the PDF into ~500 token segments
- Create embeddings using OpenAI's ada-002 model
- Upload vectors to your Pinecone index

Expected output:
```
Loading PDF: keith_running_floor_ii_installation_manual.pdf
Loaded 55 pages
Created 150+ chunks
Creating embeddings...
Uploading to Pinecone...
✅ Ingestion complete!
```

### Step 7: Run the App Locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## ☁️ Deploy to Streamlit Cloud

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

**Important:** Don't commit your `.env` or `secrets.toml` files!

Add to `.gitignore`:
```
.env
.streamlit/secrets.toml
*.pdf
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Select your GitHub repository
4. Set **Main file path** to `app.py`
5. Click **Advanced settings** and add your secrets:

```toml
OPENAI_API_KEY = "sk-your-key"
PINECONE_API_KEY = "your-key"
PINECONE_ENV = "us-east-1-aws"
PINECONE_INDEX = "running-floor-manual"
```

6. Click **Deploy!**

## 📁 Project Structure

```
running-floor-rag/
├── app.py                  # Main Streamlit application
├── ingest.py               # PDF processing & Pinecone upload
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
├── .streamlit/
│   └── secrets.toml.example    # Streamlit secrets template
└── README.md               # This file
```

## 💡 Usage Examples

Try asking the chatbot:

- "How do I prepare a trailer for Walking Floor installation?"
- "What are the steps for installing the drive unit in a center frame trailer?"
- "How should I install the floor seals?"
- "What torque should I use for floor bolts?"
- "How do I route hydraulic tubing?"
- "What's the minimum drive gap needed?"

## 🔧 Configuration Options

### Chunk Size
In `ingest.py`, adjust chunk parameters for different results:

```python
chunks = load_and_split_pdf(pdf_path, chunk_size=500, chunk_overlap=100)
```

- **Smaller chunks** (300-400): More precise answers, may miss context
- **Larger chunks** (600-800): More context, may include irrelevant info

### Search Results
In `app.py`, adjust `top_k` to return more or fewer source documents:

```python
matches = search_knowledge_base(prompt, index, top_k=5)
```

### Temperature
Adjust `temperature` in `get_chat_response()` for response creativity:
- **0.1-0.3**: More factual, consistent (recommended for technical docs)
- **0.5-0.7**: More varied responses

## 🐛 Troubleshooting

### "Index not found" error
- Ensure you ran `ingest.py` first
- Verify index name matches in Pinecone console

### "API key invalid" error
- Double-check your API keys in `.env` or `secrets.toml`
- Ensure no extra spaces or quotes

### Slow responses
- First query may take longer due to cold start
- Pinecone free tier has some latency

### Poor answer quality
- Try rephrasing your question
- Ensure the PDF was properly ingested (check Pinecone console for vector count)

## 📊 Cost Estimation

**OpenAI Costs (approximate):**
- Embeddings (text-embedding-3-small): ~$0.00002 per 1K tokens (very cheap!)
- GPT-4 Turbo: ~$0.01-0.03 per query
- Initial ingestion: ~$0.01-0.02 (one-time)
- Monthly usage (100 queries/day): ~$30-90

**Pinecone:**
- Free tier: Sufficient for this project (check current limits at pinecone.io)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📝 License

Internal use only - Keith Manufacturing Company

## 🏭 About Keith Manufacturing

Keith Manufacturing Company, headquartered in Madras, Oregon, is the inventor and leading manufacturer of the WALKING FLOOR® conveyor system. 

- Website: [keithwalkingfloor.com](https://www.keithwalkingfloor.com)
- Phone: 800-547-6161
- Fax: 541-475-2169

---

Built with ❤️ for Keith Manufacturing Company
