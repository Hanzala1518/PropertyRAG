# Setup Instructions for Property RAG (Pinecone + Gemini) — Python 3.13 Compatible
# Setup Instructions for Property RAG (Pinecone + Gemini)

## Prerequisites
- Python 3.10+
- Git
- Google Gemini API key (Google AI Studio)
	- https://ai.google.dev/gemini-api/docs/api-key
- Pinecone account and API key (create an index via UI or let the script create one)
	- https://app.pinecone.io/

## Step 1: Clone the repository & create a virtual environment
PowerShell (Windows):
```powershell
git clone <your-repo-url>
cd project
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
git clone <your-repo-url>
cd project
python3 -m venv venv
source venv/bin/activate
```

## Step 2: Install dependencies
```powershell
pip install -r requirements.txt
```

If you see a pandas build error mentioning Visual Studio tools or Meson, you're likely on an unsupported Python version (e.g., 3.13). Use Python 3.11 as shown above, then recreate and activate the venv before installing requirements.

If script activation fails due to policy, run once:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Step 3: Get your API keys
- Gemini: https://ai.google.dev/gemini-api/docs/api-key
- Pinecone: https://app.pinecone.io/

## Step 4: Configure environment variables
Create a file named `.env` in the `project/` folder and fill in:
```
GOOGLE_API_KEY="your-gemini-key"
PINECONE_API_KEY="your-pinecone-key"
PINECONE_INDEX="properties-index"
PINECONE_CLOUD="aws"       # or "gcp"
PINECONE_REGION="us-east-1"  # e.g., "us-east-1" or "us-central1"
```

## Step 5: Ingest the data
1. Place your CSV file named `property_data_cleaned.csv` into the `data/` folder.
2. Run ingestion:
```powershell
python ingest.py
```
This will create the Pinecone index (if missing) and upsert all vectors.

## Step 6: Run the application (two terminals)
Terminal 1 (Backend):
```powershell
uvicorn main:app --reload
```
Terminal 2 (Frontend):
Open the built-in UI at: http://127.0.0.1:8000/
Or run the Streamlit UI:
```powershell
streamlit run app.py
```

## How it works
- Embeddings: Gemini text-embedding-004 via google-generativeai. The script detects your Pinecone index dimension and requests embeddings with matching output_dimensionality (e.g., 384 or 768). No LangChain required.
- Vector DB: Pinecone for fast nearest neighbor search.
- Reranking: Uses a CrossEncoder if installed; otherwise falls back to a Gemini-based scoring step (pure-Python) to maintain accuracy on Python 3.13.
- LLM: Gemini (gemini-1.5-flash) to generate answers strictly from retrieved context.

## Troubleshooting
- Module import errors: Ensure your venv is activated and `pip install -r requirements.txt` completed successfully.
- Pinecone auth/index errors: For serverless (pinecone-client v3), verify `PINECONE_API_KEY`, `PINECONE_CLOUD`, `PINECONE_REGION`, and `PINECONE_INDEX`. For legacy v2, set `PINECONE_ENVIRONMENT` instead. Ensure the index exists or let `ingest.py` create it.
- Gemini errors: Verify `GOOGLE_API_KEY` has access and that billing and quotas are configured if required.
- Performance: Increase top_k in `search_pinecone` to improve recall; adjust reranker or use the built-in Gemini-based fallback scoring.

## Notes
- Re-run `python ingest.py` whenever you change the CSV; it will upsert updated vectors.
- The project avoids native builds (no pandas/numpy/torch required) to support Python 3.13 on Windows.
- Costs: Using Gemini and Pinecone may incur costs. Monitor usage and set sensible limits.
