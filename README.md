# 🏡 PropertyRAG - AI-Powered Real Estate Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade, intelligent question-answering system for real estate powered by **Google Gemini**, **Pinecone Vector Database**, and advanced **Retrieval-Augmented Generation (RAG)** technology.

---

## 🌟 Key Features

### 🧠 **Advanced AI Intelligence**
- **Natural Language Understanding** - Ask questions in plain English about properties
- **Intent Classification** - Automatically routes queries to 9 specialized handlers
- **Entity Extraction** - Identifies locations, prices, bedrooms, amenities from queries
- **Multi-Turn Conversations** - Maintains context across follow-up questions
- **Spell Correction & Synonym Mapping** - Understands real estate abbreviations (BHK, Sqft, etc.)

### 🔍 **Sophisticated Search & Retrieval**
- **Hybrid Search** - Combines semantic search with query expansion
- **Smart Re-ranking** - CrossEncoder-based precision ranking
- **Result Diversity** - Removes near-duplicates using Jaccard similarity
- **Intelligent Caching** - LRU cache for embeddings and query results
- **Fallback Strategies** - Multiple retrieval methods for robustness

### 💎 **Professional Response Generation**
- **Confidence Scoring** - Multi-factor confidence calculation for every answer
- **Follow-Up Suggestions** - Context-aware follow-up questions
- **Enhanced Source Citations** - Excerpts, relevance scores, credibility indicators
- **Structured Formatting** - Clear sections with confidence labels
- **Chain-of-Thought Reasoning** - Transparent step-by-step analysis

### 📊 **Deterministic Analytics Engine**
- **Area Comparisons** - Compare multiple areas across metrics (crime, schools, transport)
- **Area Rankings** - Rank areas by specific criteria (top/bottom N)
- **Statistical Aggregations** - Average, min, max calculations across properties
- **Smart Intent Detection** - Routes to analytics vs. RAG based on query

### 🎨 **Modern Web Interface**
- **Dark Mode UI** - Professional glassmorphism design
- **Real-Time Chat** - Interactive chat interface with typing indicators
- **Property Cards** - Rich property displays with hover effects
- **Advanced Sidebar** - Filters, settings, bookmarks, system status
- **Search History** - Track and revisit previous queries
- **Export Functionality** - Download results as JSON

### ⚡ **Performance & Monitoring**
- **Request Caching** - Instant responses for repeated queries
- **Metrics Dashboard** - `/metrics` endpoint for system monitoring
- **Comprehensive Logging** - Structured logs with emoji indicators
- **Query Analytics** - JSONL logs of all queries with latency tracking
- **Background Tasks** - Async metrics logging

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface (Streamlit)               │
│  • Dark mode UI with glassmorphism                          │
│  • Chat interface with history                              │
│  • Property cards with filters                              │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/REST API
┌──────────────────────▼──────────────────────────────────────┐
│              FastAPI Backend (main.py)                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Stage 1: Query Understanding                       │   │
│  │  • Preprocess (abbreviations, spelling)             │   │
│  │  • Extract entities (location, price, beds)         │   │
│  │  • Classify intent (9 types)                        │   │
│  │  • Resolve coreferences (multi-turn)                │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Stage 2: Routing                                   │   │
│  │  • Deterministic Analytics (if applicable)          │   │
│  │  • RAG Pipeline (if semantic query)                 │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Stage 3: Retrieval & Generation                    │   │
│  │  • Hybrid search with caching                       │   │
│  │  • Re-ranking (CrossEncoder)                        │   │
│  │  • Sophisticated prompt engineering                 │   │
│  │  • LLM generation (Gemini)                          │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼───────┐ ┌───▼────────┐ ┌──▼─────────┐
│   Pinecone    │ │   Gemini   │ │   Local    │
│ Vector Store  │ │    LLM     │ │   Cache    │
│ (Embeddings)  │ │ (Generate) │ │ (Queries)  │
└───────────────┘ └────────────┘ └────────────┘
```

---

## 📁 Project Structure

```
PropertyRAG/
├── project/                          # Main application directory
│   ├── main.py                       # FastAPI backend (1700+ lines)
│   ├── app.py                        # Streamlit frontend (1000+ lines)
│   ├── advanced_rag.py               # Prompt engineering module (650+ lines)
│   ├── ingest.py                     # Data ingestion to Pinecone
│   ├── test_production_api.py        # API testing script
│   ├── requirements.txt              # Python dependencies
│   ├── .env.example                  # Environment variables template
│   ├── .gitignore                    # Git ignore rules
│   ├── data/
│   │   └── property_data_cleaned.csv # Property dataset
│   └── README.md                     # This file
├── EDA/
│   └── EDA.ipynb                     # Exploratory data analysis notebook
└── .venv/                            # Virtual environment (not tracked)
```

---

## 🐍 Python Files Explained

### 1. **`main.py`** (Backend API - 1700+ lines)

**Purpose**: Production-grade FastAPI backend with intelligent RAG system

**Key Components**:

#### **Query Understanding Module** (Lines 170-574)
- `ABBREVIATIONS` - 25+ real estate abbreviations (BHK, Sqft, etc.)
- `SYNONYMS` - Property-related synonym mappings
- `SPELLING_CORRECTIONS` - Common typo corrections
- `QueryIntent` - 9 intent types enum
- `preprocess_query()` - Expands abbreviations and fixes spelling
- `extract_entities()` - Regex-based extraction of locations, prices, bedrooms
- `classify_intent()` - Routes to appropriate intent handler
- `expand_query()` - Generates synonym-based query variations
- `detect_ambiguity()` - Identifies missing critical information
- `resolve_coreference()` - Handles follow-up questions with context

#### **Intelligent Retrieval** (Lines 400-550)
- `hybrid_search()` - Query expansion + diversity filtering + caching
- `cached_embed_query()` - LRU-cached embeddings for efficiency
- `compute_cache_key()` - MD5-based cache key generation

#### **Deterministic Analytics** (Lines 800-1350)
- `_comparison_answer()` - Area comparison tables
- `_area_ranking_answer()` - Area rankings by criteria
- `_aggregate_answer()` - Statistical calculations
- `_detect_topn()` - Top/bottom N detection

#### **API Endpoints**
- `POST /ask` - Main query endpoint (260 lines, 8-stage processing)
- `GET /status` - Backend health check
- `GET /metrics` - System metrics dashboard
- `POST /clear_cache` - Clear query cache
- `POST /reset_conversation` - Reset session context

#### **8-Stage Processing Pipeline**:
1. **Query Understanding** - Preprocess, extract, classify, resolve
2. **Routing** - Deterministic analytics or RAG
3. **Retrieval** - Hybrid search with caching
4. **Re-ranking** - CrossEncoder precision ranking
5. **Prompt Engineering** - Intent-specific sophisticated prompts
6. **Generation** - Gemini LLM with fallbacks
7. **Formatting** - Professional response structure
8. **Logging** - Metrics, conversation context, analytics

---

### 2. **`advanced_rag.py`** (Prompt Engineering - 650+ lines)

**Purpose**: Sophisticated prompt templates and response formatting

**Key Components**:

- **`ROLE_BASED_SYSTEM_PROMPT`** - Expert consultant persona (15+ years experience)
- **`INTENT_PROMPTS`** - 7 specialized prompt templates:
  - `property_search` - Property listings with recommendations
  - `comparison` - Side-by-side comparisons with pros/cons
  - `market_trends` - Market analysis with trends
  - `investment_advice` - ROI analysis with disclaimers
  - `legal_query` - Regulations with legal disclaimers
  - `amenities_question` - Amenities with distances
  - `area_ranking` - Rankings with methodology transparency

- **Response Formatting**:
  - `format_professional_response()` - Structures with confidence, follow-ups
  - `format_sources()` - Enhanced citations with excerpts, scores
  - `generate_follow_ups()` - Context-aware follow-up questions
  - `calculate_response_confidence()` - Multi-factor confidence scoring

- **Error Handling**:
  - `generate_clarifying_questions()` - Ambiguity resolution
  - `handle_out_of_scope()` - Non-real-estate query detection
  - `format_error_response()` - User-friendly error messages

---

### 3. **`app.py`** (Frontend UI - 1000+ lines)

**Purpose**: Modern Streamlit web interface with dark mode

**Key Features**:

- **Custom CSS Framework** (500+ lines):
  - Glassmorphism effects with backdrop-filter
  - Animated gradients and transitions
  - Professional dark theme colors
  - Responsive design with media queries
  - Hover effects and skeleton loaders

- **UI Components**:
  - **Navigation Header** - Home, Analytics, Favorites, Settings
  - **Hero Section** - Animated background with feature badges
  - **Feature Grid** - 4 interactive feature cards
  - **Search Container** - Query input with history
  - **Chat Interface** - User/assistant avatars, typing indicators
  - **Property Cards** - Rich displays with metadata and hover effects
  - **Advanced Sidebar**:
    - System Status (backend health, vector count)
    - Filters (price range, bedrooms, property type)
    - Settings (model, response length, sources)
    - Bookmarks (save favorite queries)
    - Quick Actions (export, clear)

- **Session Management**:
  - Search history tracking
  - Conversation context
  - Bookmarks and favorites
  - Filter persistence

---

### 4. **`ingest.py`** (Data Ingestion)

**Purpose**: Ingest CSV property data into Pinecone vector database

**Features**:
- Parallel embedding generation with ThreadPoolExecutor
- Pinecone v3 serverless support with auto-fallback to v2
- Dimension auto-detection (INDEX_DIM)
- Batch upsert for efficiency
- Progress tracking
- Error handling with retries

**Usage**:
```bash
python ingest.py
```

---

### 5. **`test_production_api.py`** (Testing)

**Purpose**: Comprehensive API testing suite

**Test Cases**:
- Simple property search
- Comparison queries
- Area ranking queries
- Follow-up questions (multi-turn)
- Ambiguous queries (clarification)
- Out-of-scope queries
- Metrics endpoint
- Backend status check

**Usage**:
```bash
python test_production_api.py
```

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.13+** (or 3.10+)
- **Google Gemini API Key** ([Get it here](https://makersuite.google.com/app/apikey))
- **Pinecone API Key** ([Get it here](https://www.pinecone.io/))
- **Git** (for cloning and version control)

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/PropertyRAG.git
cd PropertyRAG/project
```

---

### Step 2: Create Virtual Environment

**Windows (PowerShell)**:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**:
```bash
python -m venv venv
venv\Scripts\activate.bat
```

**macOS/Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include**:
- `fastapi` - Web framework for backend API
- `uvicorn` - ASGI server for FastAPI
- `streamlit` - Frontend web UI framework
- `google-generativeai` - Google Gemini LLM
- `pinecone-client` - Pinecone vector database
- `sentence-transformers` (optional) - CrossEncoder re-ranking
- `python-dotenv` - Environment variable management
- `pydantic` - Data validation
- `requests` - HTTP client

---

### Step 4: Configure Environment Variables

Create a `.env` file in the `project/` directory:

```bash
# Create .env file
cp .env.example .env  # or manually create
```

Add your API keys to `.env`:

```env
# Google Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=property-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

**Get API Keys**:
- **Gemini**: https://makersuite.google.com/app/apikey
- **Pinecone**: https://app.pinecone.io/ (Sign up → API Keys)

---

### Step 5: Ingest Data into Pinecone

```bash
python ingest.py
```

This will:
- Read `data/property_data_cleaned.csv`
- Generate embeddings using Gemini
- Create Pinecone index (if not exists)
- Upsert vectors to Pinecone
- Takes ~5-10 minutes for large datasets

**Expected Output**:
```
🚀 Starting property data ingestion...
📊 Loaded 1000 properties from CSV
🔄 Generating embeddings (parallel)...
✅ Generated 1000 embeddings
☁️ Creating Pinecone index...
✅ Index created successfully
📤 Upserting vectors to Pinecone...
✅ Ingestion complete! Total properties: 1000
```

---

### Step 6: Run the Backend (FastAPI)

**Option 1: Using uvicorn directly**
```bash
uvicorn main:app --reload --port 8000
```

**Option 2: Using Python module**
```bash
python -m uvicorn main:app --reload --port 8000
```

**Expected Output**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
✅ Advanced RAG modules loaded successfully
```

**Backend will be available at**: http://localhost:8000

**API Documentation**: http://localhost:8000/docs (Swagger UI)

---

### Step 7: Run the Frontend (Streamlit)

**Open a new terminal** (keep backend running), activate venv, and run:

```bash
streamlit run app.py
```

**Expected Output**:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**Frontend will open automatically at**: http://localhost:8501

---

### Step 8: Test the System

#### **Via Web UI** (Streamlit at http://localhost:8501):
1. Type a query: "Show me 2 BHK flats in Mumbai under 50 lakhs"
2. View AI-generated response with sources
3. Check confidence score and follow-up suggestions
4. Try follow-up: "What about Pune?"

#### **Via API** (curl or Postman):
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare properties in Bandra vs Andheri"}'
```

#### **Via Test Script**:
```bash
python test_production_api.py
```

#### **View Metrics**:
```bash
curl http://localhost:8000/metrics
```

---

## 🎯 Example Queries

### Property Search
```
"Show me 3 BHK apartments in Bangalore under 1 crore"
"Find properties near good schools in Pune"
"2 bedroom flats with parking in Mumbai"
```

### Area Comparison
```
"Compare Bandra vs Andheri for families"
"Which is better for investment: Gurgaon or Noida?"
"Compare SW1A vs E1 for crime rates"
```

### Market Trends
```
"What are property trends in Bangalore?"
"Is real estate market up or down in Delhi?"
"Property price predictions for 2025"
```

### Area Rankings
```
"Top 5 areas with lowest crime"
"Bottom 3 areas for school ratings"
"Best areas for public transport"
```

### Multi-Turn Conversation
```
User: "Tell me about properties in Mumbai"
AI: [Response with details]
User: "What about prices there?"
AI: [Contextual response about Mumbai prices]
```

---

## 📊 System Monitoring

### Metrics Dashboard
Access real-time metrics at: http://localhost:8000/metrics

**Tracked Metrics**:
- Total queries processed
- Cache hit rate
- Average retrieval latency
- Successful/failed generations
- Deterministic vs RAG responses
- Intent breakdown

### Logs

**Backend Logs** (`rag_system.log`):
```bash
tail -f rag_system.log
```

**Query Analytics** (`query_metrics.jsonl`):
```bash
tail -f query_metrics.jsonl
```

**Log Format**:
```json
{
  "timestamp": "2025-10-29T10:30:00",
  "session_id": "user123",
  "query": "Show me 2 BHK in Mumbai",
  "intent": "property_search",
  "num_results": 12,
  "confidence": 0.85,
  "latency_seconds": 2.3
}
```

---

## 🔧 Configuration

### Query Model Options

```python
{
  "question": "Your query here",
  "session_id": "optional-session-id",  # For multi-turn
  "use_caching": true,                  # Enable caching
  "use_expansions": true                # Use query expansion
}
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | ✅ Yes | - | Google Gemini API key |
| `PINECONE_API_KEY` | ✅ Yes | - | Pinecone API key |
| `PINECONE_INDEX` | No | `property-rag` | Pinecone index name |
| `PINECONE_CLOUD` | No | `aws` | Cloud provider |
| `PINECONE_REGION` | No | `us-east-1` | Region |

---

## 🧪 Testing

### Run All Tests
```bash
python test_production_api.py
```

### Test Individual Endpoints

**Backend Health**:
```bash
curl http://localhost:8000/status
```

**Query Endpoint**:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me properties in Delhi"}'
```

**Metrics**:
```bash
curl http://localhost:8000/metrics
```

**Clear Cache**:
```bash
curl -X POST http://localhost:8000/clear_cache
```

---

## 📈 Performance Optimization

### Caching Strategy
- **Query Result Cache**: MD5-based keys for instant repeated queries
- **Embedding Cache**: LRU cache for frequent queries
- **Session Context**: In-memory conversation history

### Monitoring
- **Real-time Metrics**: `/metrics` endpoint
- **Query Analytics**: JSONL logs with latency tracking
- **Cache Hit Rate**: Monitor efficiency

### Tuning Parameters

**In `main.py`**:
```python
# Retrieval settings
top_k = 15  # Number of documents to retrieve
context_size = 8  # Documents for LLM context

# Cache settings
cache_ttl = 3600  # Cache expiration (seconds)
```

---

## 🐛 Troubleshooting

### Issue: Backend won't start
**Solution**:
- Check `.env` file has valid API keys
- Ensure virtual environment is activated
- Install dependencies: `pip install -r requirements.txt`

### Issue: No results from queries
**Solution**:
- Verify Pinecone index exists: Check Pinecone dashboard
- Re-run ingestion: `python ingest.py`
- Check backend logs: `tail -f rag_system.log`

### Issue: Slow response times
**Solution**:
- Enable caching: `"use_caching": true`
- Reduce `top_k` in `main.py`
- Check network latency to Pinecone/Gemini

### Issue: "Module not found" errors
**Solution**:
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Encoding errors in logs (Windows)
**Solution**: Already handled - emojis are UTF-8 compatible

---

## 📚 API Documentation

### Interactive Docs
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### `POST /ask`
Query the RAG system

**Request**:
```json
{
  "question": "Show me 2 BHK in Mumbai",
  "session_id": "user123",
  "use_caching": true,
  "use_expansions": true
}
```

**Response**:
```json
{
  "answer": "Based on your search...",
  "confidence": {
    "score": 0.85,
    "label": "High"
  },
  "intent": "property_search",
  "follow_ups": [
    "Would you like to see 3 BHK options?",
    "Are you interested in specific areas?"
  ],
  "sources": [...],
  "metadata": {...}
}
```

#### `GET /status`
Backend health check

#### `GET /metrics`
System performance metrics

#### `POST /clear_cache`
Clear query result cache

#### `POST /reset_conversation`
Reset conversation context for session

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Google Gemini** - Powerful LLM for embeddings and generation
- **Pinecone** - Scalable vector database
- **FastAPI** - Modern web framework
- **Streamlit** - Beautiful UI framework
- **Sentence Transformers** - CrossEncoder re-ranking

---

## 📞 Support

For issues, questions, or suggestions:
- **GitHub Issues**: [Create an issue](https://github.com/YOUR_USERNAME/PropertyRAG/issues)
- **Discussions**: [Join discussions](https://github.com/YOUR_USERNAME/PropertyRAG/discussions)

---

## 🚀 Deploying to GitHub

### Initial Setup

```bash
# Navigate to project directory
cd PropertyRAG/project

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Production-grade PropertyRAG system"

# Create repository on GitHub (via web interface)
# Then connect your local repo:
git remote add origin https://github.com/YOUR_USERNAME/PropertyRAG.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Updating Repository

```bash
# Check status
git status

# Add changes
git add .

# Commit with message
git commit -m "Your commit message"

# Push to GitHub
git push origin main
```

---

## 🎉 Success!

Your PropertyRAG system is now:
- ✅ Optimized for production
- ✅ Properly documented
- ✅ Ready for GitHub
- ✅ Easy to setup for new users

**Start exploring intelligent real estate search!** 🏡✨
