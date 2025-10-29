"""
FastAPI backend for the Property RAG system (Pinecone + Gemini, PRODUCTION-GRADE).

🏆 ENTERPRISE-READY FEATURES:
- Advanced query understanding: preprocessing, intent classification, entity extraction
- Intelligent retrieval: hybrid search, re-ranking, multi-hop, fallback strategies
- Sophisticated prompt engineering: role-based, intent-specific, chain-of-thought
- Professional responses: structured sections, confidence scores, follow-ups, citations
- Comprehensive error handling: clarifying questions, graceful degradation
- Production optimization: caching, async, logging, metrics

Usage: uvicorn main:app --reload
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import re
import os
import json
import time
import hashlib
import logging
from datetime import datetime
from collections import defaultdict
from functools import lru_cache
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
try:
    from pinecone import Pinecone  # v3
    _PC_V3 = True
except Exception:
    Pinecone = None
    _PC_V3 = False
import google.generativeai as genai
import importlib

# Optional CrossEncoder (loaded dynamically to avoid hard dependency warnings)
try:
    _st_mod = importlib.import_module("sentence_transformers")
    CrossEncoder = getattr(_st_mod, "CrossEncoder", None)
    _HAS_CROSS_ENCODER = CrossEncoder is not None
except Exception:
    CrossEncoder = None
    _HAS_CROSS_ENCODER = False

# Load environment
ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env", override=True)

# ==============================
# 📊 LOGGING SETUP
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ROOT / "rag_system.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================
# 💾 CACHING & METRICS
# ==============================
_QUERY_CACHE: Dict[str, Dict[str, Any]] = {}
_METRICS: Dict[str, Any] = defaultdict(int)
_CONVERSATION_CONTEXT: Dict[str, List[Dict[str, str]]] = {}  # session_id -> history

# Keys and config
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "properties-index")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not set. Please set it in .env.")
if not PINECONE_API_KEY:
    raise EnvironmentError("PINECONE_API_KEY not set. Please set it in .env.")
 # Cloud/region default to aws/us-east-1; can be overridden in .env

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize embedding model
EMBED_MODEL_NAME = "models/text-embedding-004"
genai.configure(api_key=GOOGLE_API_KEY)

# Resolve a supported generative model for generate_content with fallback and env override
def _select_generative_model() -> str:
    # Allow explicit override via .env
    env_model = os.getenv("GENAI_MODEL")
    if env_model:
        return env_model
    try:
        models = list(genai.list_models())
        # Filter to models that support generateContent and exclude experimental/2.5 by default
        names = [getattr(m, "name", str(m)) for m in models]
        supported = []
        for name in names:
            try:
                # must support generateContent
                m = next(mm for mm in models if getattr(mm, "name", str(mm)) == name)
                if not (hasattr(m, "supported_generation_methods") and ("generateContent" in m.supported_generation_methods)):
                    continue
                # exclude experimental/2.5 variants which often have zero free-tier quota
                if "-exp" in name or "/2.5" in name or "-2.5-" in name or name.endswith("2.5"):
                    continue
                supported.append(name)
            except Exception:
                continue
        # Prefer these in order
        preferred = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-flash-8b",
            "models/gemini-1.5-pro",
            "models/gemini-1.0-pro",
        ]
        for p in preferred:
            if p in supported:
                return p
        if supported:
            return supported[0]
    except Exception:
        pass
    # Last resort default; SDK may still raise if unavailable in region/account
    return "models/gemini-1.5-flash"

GENERATIVE_MODEL_NAME = _select_generative_model()

if _PC_V3:
    # Initialize Pinecone v3 client
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    # Detect index dimension (fallback to EMBED_DIM or 768)
    try:
        desc = pc.describe_index(PINECONE_INDEX)
        _dim = getattr(desc, "dimension", None)
        if _dim is None and isinstance(desc, dict):
            _dim = desc.get("dimension") or (desc.get("spec", {}) or {}).get("dimension")
    except Exception:
        _dim = None
else:
    # Legacy v2 fallback
    import pinecone as pine_v2
    pine_env = os.getenv("PINECONE_ENVIRONMENT")
    if not pine_env:
        raise EnvironmentError(
            "Detected pinecone-client v2 but PINECONE_ENVIRONMENT is not set. Either set PINECONE_ENVIRONMENT "
            "(e.g., 'gcp-starter') in .env or upgrade to pinecone-client>=3 to use serverless cloud/region."
        )
    pine_v2.init(api_key=PINECONE_API_KEY, environment=pine_env)
    # Detect index dimension (fallback to EMBED_DIM or 768)
    try:
        _d = pine_v2.describe_index(PINECONE_INDEX)
        _dim = (_d.get("dimension") if isinstance(_d, dict) else getattr(_d, "dimension", None))
    except Exception:
        _dim = None
    index = pine_v2.Index(PINECONE_INDEX)

# Resolve embedding dimension (prefer index dimension, else env, else 768)
try:
    _env_dim = int(os.getenv("EMBED_DIM")) if os.getenv("EMBED_DIM") else None
except Exception:
    _env_dim = None
INDEX_DIM = _dim or _env_dim or 768

def _adjust_vec(vec: List[float], dim: int) -> List[float]:
    if not isinstance(vec, list):
        vec = []
    ln = len(vec)
    if ln == dim:
        return vec
    if ln > dim:
        return vec[:dim]
    return vec + [0.0] * (dim - ln)

# ==============================
# 🧠 ADVANCED QUERY UNDERSTANDING
# ==============================

# Abbreviation expansion dictionary
ABBREVIATIONS = {
    "bhk": "bedroom hall kitchen",
    "1bhk": "1 bedroom hall kitchen",
    "2bhk": "2 bedroom hall kitchen",
    "3bhk": "3 bedroom hall kitchen",
    "4bhk": "4 bedroom hall kitchen",
    "sq": "square",
    "sqft": "square feet",
    "sq.ft": "square feet",
    "apt": "apartment",
    "apmt": "apartment",
    "bldg": "building",
    "nr": "near",
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "pk": "park",
    "hwy": "highway",
    "mins": "minutes",
    "k": "thousand",
    "m": "million",
    "lac": "hundred thousand",
    "cr": "crore",
    "lakh": "hundred thousand",
}

# Synonym mapping for better semantic understanding
SYNONYMS = {
    "flat": ["apartment", "unit", "condo"],
    "house": ["home", "residence", "villa", "bungalow"],
    "price": ["cost", "rate", "amount", "value"],
    "cheap": ["affordable", "budget", "economical", "inexpensive"],
    "expensive": ["costly", "premium", "luxury", "high-end"],
    "new": ["brand new", "newly built", "fresh", "recent"],
    "old": ["aged", "vintage", "existing", "resale"],
    "near": ["close to", "proximity to", "around", "nearby"],
    "safe": ["secure", "low crime", "protected"],
    "amenities": ["facilities", "features", "services"],
}

# Common spelling corrections
SPELLING_CORRECTIONS = {
    "appartment": "apartment",
    "appartments": "apartments",
    "appartmnt": "apartment",
    "bedrrom": "bedroom",
    "bedrom": "bedroom",
    "bathrrom": "bathroom",
    "bathrom": "bathroom",
    "kichen": "kitchen",
    "kitchn": "kitchen",
    "buildig": "building",
    "buidling": "building",
    "locaton": "location",
    "loaction": "location",
    "pric": "price",
    "priice": "price",
}

class QueryIntent:
    """Query intent classifications"""
    PROPERTY_SEARCH = "property_search"
    COMPARISON = "comparison"
    MARKET_TRENDS = "market_trends"
    INVESTMENT_ADVICE = "investment_advice"
    LEGAL_QUERY = "legal_query"
    AMENITIES_QUESTION = "amenities_question"
    AREA_RANKING = "area_ranking"
    SPECIFIC_PROPERTY = "specific_property"
    GENERAL_INFO = "general_info"

def preprocess_query(query: str) -> str:
    """
    Advanced query preprocessing: abbreviation expansion, spell correction, normalization
    """
    query = query.strip().lower()
    
    # Expand abbreviations
    for abbr, full in ABBREVIATIONS.items():
        # Word boundary matching to avoid partial matches
        query = re.sub(rf"\b{re.escape(abbr)}\b", full, query, flags=re.IGNORECASE)
    
    # Correct common misspellings
    words = query.split()
    corrected_words = [SPELLING_CORRECTIONS.get(w, w) for w in words]
    query = " ".join(corrected_words)
    
    # Normalize whitespace
    query = re.sub(r"\s+", " ", query).strip()
    
    logger.info(f"Preprocessed query: {query}")
    return query

def extract_entities(query: str) -> Dict[str, Any]:
    """
    Named entity recognition: extract locations, prices, property types, amenities
    """
    entities = {
        "locations": [],
        "price_range": {},
        "property_types": [],
        "bedrooms": None,
        "bathrooms": None,
        "amenities": [],
        "dates": []
    }
    
    # Extract price range
    price_patterns = [
        r"(?:under|below|less than|<)\s*[£$]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*([km]|lakh|lac|crore|cr)?",
        r"(?:above|over|more than|>)\s*[£$]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*([km]|lakh|lac|crore|cr)?",
        r"[£$]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:to|-)\s*[£$]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
        r"around\s*[£$]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*([km]|lakh|lac|crore|cr)?"
    ]
    
    for pattern in price_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            # Parse first match for simplicity
            if isinstance(matches[0], tuple) and len(matches[0]) >= 1:
                entities["price_range"]["extracted"] = matches[0]
            break
    
    # Extract bedrooms
    bed_match = re.search(r"(\d+)\s*(?:bed|bedroom|bhk)", query, re.IGNORECASE)
    if bed_match:
        entities["bedrooms"] = int(bed_match.group(1))
    
    # Extract bathrooms
    bath_match = re.search(r"(\d+)\s*(?:bath|bathroom)", query, re.IGNORECASE)
    if bath_match:
        entities["bathrooms"] = int(bath_match.group(1))
    
    # Extract property types
    prop_types = ["flat", "apartment", "house", "villa", "studio", "condo", "bungalow", "duplex"]
    for ptype in prop_types:
        if re.search(rf"\b{ptype}s?\b", query, re.IGNORECASE):
            entities["property_types"].append(ptype)
    
    # Extract amenities
    amenity_keywords = ["gym", "pool", "parking", "garden", "balcony", "security", "lift", "elevator", "playground"]
    for amenity in amenity_keywords:
        if re.search(rf"\b{amenity}\b", query, re.IGNORECASE):
            entities["amenities"].append(amenity)
    
    # Extract common UK locations (simplified; extend with full gazetteer)
    location_patterns = [
        r"\b(london|manchester|birmingham|liverpool|leeds|glasgow|edinburgh)\b",
        r"\b([A-Z]{1,2}\d{1,2}[A-Z]?)\b",  # Postcode outcodes
        r"in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",  # "in Chelsea", "in West End"
    ]
    for pattern in location_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        entities["locations"].extend([m if isinstance(m, str) else m[0] for m in matches])
    
    entities["locations"] = list(set([loc.strip() for loc in entities["locations"] if loc.strip()]))
    
    logger.info(f"Extracted entities: {entities}")
    return entities

def classify_intent(query: str, entities: Dict[str, Any]) -> str:
    """
    Classify query intent based on keywords and extracted entities
    """
    query_lower = query.lower()
    
    # Comparison intent
    if any(kw in query_lower for kw in ["compare", "vs", "versus", "difference between", "better", "or"]):
        return QueryIntent.COMPARISON
    
    # Area ranking
    if any(kw in query_lower for kw in ["top", "best", "worst", "safest", "most crime", "areas", "rank"]):
        return QueryIntent.AREA_RANKING
    
    # Market trends
    if any(kw in query_lower for kw in ["trend", "market", "average", "median", "statistics", "data"]):
        return QueryIntent.MARKET_TRENDS
    
    # Investment advice
    if any(kw in query_lower for kw in ["invest", "roi", "return", "profit", "appreciation", "growth"]):
        return QueryIntent.INVESTMENT_ADVICE
    
    # Legal/regulatory
    if any(kw in query_lower for kw in ["legal", "law", "regulation", "tax", "stamp duty", "documentation"]):
        return QueryIntent.LEGAL_QUERY
    
    # Amenities focus
    if len(entities.get("amenities", [])) > 0 or any(kw in query_lower for kw in ["facilities", "features", "amenities"]):
        return QueryIntent.AMENITIES_QUESTION
    
    # Specific property (likely has ID or very specific address)
    if re.search(r"\b(property|listing|id|ref)\s*#?\s*\d+", query_lower):
        return QueryIntent.SPECIFIC_PROPERTY
    
    # Property search (has location, type, or price criteria)
    if entities.get("locations") or entities.get("property_types") or entities.get("price_range"):
        return QueryIntent.PROPERTY_SEARCH
    
    # Default: general info
    return QueryIntent.GENERAL_INFO

def expand_query(query: str, entities: Dict[str, Any]) -> List[str]:
    """
    Generate multiple query variations for better retrieval
    """
    expansions = [query]
    
    # Add synonym variations
    for word, syns in SYNONYMS.items():
        if re.search(rf"\b{word}\b", query, re.IGNORECASE):
            for syn in syns[:2]:  # Limit to 2 synonyms per word
                expanded = re.sub(rf"\b{word}\b", syn, query, flags=re.IGNORECASE, count=1)
                if expanded != query:
                    expansions.append(expanded)
    
    # Add entity-focused variations
    if entities.get("bedrooms"):
        expansions.append(f"{entities['bedrooms']} bedroom property")
    
    if entities.get("locations"):
        for loc in entities["locations"][:2]:
            expansions.append(f"property in {loc}")
    
    # Limit total expansions
    return list(set(expansions))[:5]

def detect_ambiguity(query: str, entities: Dict[str, Any]) -> Optional[str]:
    """
    Detect ambiguous queries and suggest clarifications
    """
    # Missing location
    if not entities.get("locations") and any(kw in query.lower() for kw in ["find", "search", "looking for", "show me"]):
        return "Could you specify which area or location you're interested in?"
    
    # Vague property type
    if not entities.get("property_types") and any(kw in query.lower() for kw in ["property", "home", "place"]):
        return "Are you looking for a flat, house, or another type of property?"
    
    # Price range too broad or missing
    if "affordable" in query.lower() and not entities.get("price_range"):
        return "What's your budget range? (e.g., £200k-£300k)"
    
    return None

def get_conversation_context(session_id: str) -> List[Dict[str, str]]:
    """Retrieve conversation history for multi-turn context"""
    return _CONVERSATION_CONTEXT.get(session_id, [])

def update_conversation_context(session_id: str, role: str, content: str):
    """Update conversation history"""
    if session_id not in _CONVERSATION_CONTEXT:
        _CONVERSATION_CONTEXT[session_id] = []
    _CONVERSATION_CONTEXT[session_id].append({"role": role, "content": content})
    # Keep last 10 turns
    _CONVERSATION_CONTEXT[session_id] = _CONVERSATION_CONTEXT[session_id][-10:]

def resolve_coreference(query: str, context: List[Dict[str, str]]) -> str:
    """
    Resolve pronouns and references using conversation context
    Examples: "what about 3BHK?" -> "what about 3 bedroom properties in [previous location]"
    """
    if not context:
        return query
    
    query_lower = query.lower()
    
    # Detect follow-up patterns
    followup_patterns = [
        r"^(what about|how about|show me)\s+(.+)",
        r"^(and|also)\s+(.+)",
        r"^(cheaper|expensive|bigger|smaller)\s+",
    ]
    
    is_followup = any(re.match(p, query_lower) for p in followup_patterns)
    
    if is_followup and len(context) >= 2:
        # Get previous user query
        prev_query = next((c["content"] for c in reversed(context) if c["role"] == "user"), None)
        if prev_query:
            # Extract location from previous query
            prev_entities = extract_entities(prev_query)
            if prev_entities.get("locations"):
                if "in " not in query_lower:
                    query += f" in {prev_entities['locations'][0]}"
    
    return query

# ==============================
# 🔍 INTELLIGENT RETRIEVAL SYSTEM
# ==============================

def compute_cache_key(query: str, top_k: int, filters: Dict = None) -> str:
    """Generate cache key for query results"""
    key_data = f"{query}_{top_k}_{json.dumps(filters or {}, sort_keys=True)}"
    return hashlib.md5(key_data.encode()).hexdigest()

@lru_cache(maxsize=100)
def cached_embed_query(text: str) -> Tuple[float, ...]:
    """Cached embedding generation"""
    vec = _embed_query(text)
    return tuple(vec)  # Convert to tuple for hashability

def hybrid_search(
    query: str,
    top_k: int = 12,
    use_expansions: bool = True,
    diversity_threshold: float = 0.85
) -> List[Dict[str, Any]]:
    """
    Hybrid search combining semantic search with query expansion and diversity
    """
    start_time = time.time()
    
    # Check cache
    cache_key = compute_cache_key(query, top_k)
    if cache_key in _QUERY_CACHE:
        logger.info(f"Cache hit for query: {query[:50]}")
        _METRICS["cache_hits"] += 1
        return _QUERY_CACHE[cache_key]["results"]
    
    _METRICS["cache_misses"] += 1
    
    # Generate query variations
    entities = extract_entities(query)
    expansions = expand_query(query, entities) if use_expansions else [query]
    
    # Retrieve for each expansion
    all_results = []
    seen_ids = set()
    
    for exp_query in expansions:
        try:
            q_vec = _embed_query(exp_query)
            results = index.query(vector=q_vec, top_k=top_k, include_metadata=True)
            
            for match in results.get("matches", []):
                match_id = match.get("id")
                if match_id not in seen_ids:
                    meta = match.get("metadata", {})
                    doc = meta.get("document", "")
                    score = match.get("score", 0.0)
                    
                    all_results.append({
                        "id": match_id,
                        "document": doc,
                        "metadata": meta,
                        "score": score,
                        "query_variant": exp_query
                    })
                    seen_ids.add(match_id)
        except Exception as e:
            logger.error(f"Error in hybrid search for '{exp_query}': {e}")
    
    # Sort by score
    all_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Apply diversity filter to avoid redundant results
    diverse_results = []
    for item in all_results:
        is_diverse = True
        doc_text = item["document"].lower()
        for existing in diverse_results:
            # Simple Jaccard similarity check
            set1 = set(doc_text.split())
            set2 = set(existing["document"].lower().split())
            if set1 and set2:
                similarity = len(set1 & set2) / len(set1 | set2)
                if similarity > diversity_threshold:
                    is_diverse = False
                    break
        if is_diverse:
            diverse_results.append(item)
        if len(diverse_results) >= top_k:
            break
    
    # Cache results
    _QUERY_CACHE[cache_key] = {
        "results": diverse_results,
        "timestamp": datetime.now().isoformat()
    }
    
    latency = time.time() - start_time
    _METRICS["retrieval_latency_sum"] += latency
    _METRICS["retrieval_count"] += 1
    logger.info(f"Hybrid search completed in {latency:.2f}s, returned {len(diverse_results)} results")
    
    return diverse_results

# Pydantic models
class Query(BaseModel):
    question: str
    session_id: Optional[str] = Field(default="default", description="Session ID for conversation tracking")
    use_caching: Optional[bool] = Field(default=True, description="Enable query result caching")
    use_expansions: Optional[bool] = Field(default=True, description="Enable query expansion for better retrieval")
    
class EnhancedResponse(BaseModel):
    answer: str
    confidence: Dict[str, Any]
    intent: str
    follow_up_suggestions: List[str]
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    clarifying_questions: Optional[List[str]] = None
    error: Optional[bool] = False

# Import advanced modules
try:
    from advanced_rag import (
        format_professional_response,
        build_sophisticated_prompt,
        generate_clarifying_questions,
        handle_out_of_scope,
        calculate_response_confidence,
        format_error_response
    )
    _HAS_ADVANCED_RAG = True
    logger.info("✅ Advanced RAG modules loaded successfully")
except ImportError as e:
    _HAS_ADVANCED_RAG = False
    logger.warning(f"⚠️ Advanced RAG modules not available: {e}")

app = FastAPI(
    title="PropertyRAG API - Production Grade",
    description="Enterprise-ready property intelligence system with advanced NLP and RAG capabilities",
    version="2.0.0"
)


@app.get("/")
def index_page():
        # Simple HTML UI to ask questions without Streamlit
        return (
                """
                <!doctype html>
                <html>
                <head>
                    <meta charset='utf-8'>
                    <title>Property AI Assistant</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 2rem; }
                        textarea { width: 100%; height: 120px; }
                        .answer { white-space: pre-wrap; background: #f7f7f7; padding: 1rem; border-radius: 6px; }
                        .sources { font-size: 0.9em; color: #444; }
                    </style>
                </head>
                <body>
                    <h1>Property AI Assistant</h1>
                    <p>Ask a question about the properties. The assistant will only use the indexed data.</p>
                    <textarea id="q" placeholder="e.g., What are the cheapest 2-bed flats in London?"></textarea>
                    <br/><br/>
                    <button onclick="ask()">Ask</button>
                    <p id="status"></p>
                    <h2>Answer</h2>
                    <div id="ans" class="answer"></div>
                    <h3>Sources</h3>
                    <div id="src" class="sources"></div>
                    <script>
                        async function ask(){
                            const status = document.getElementById('status');
                            const ans = document.getElementById('ans');
                            const src = document.getElementById('src');
                            status.textContent = 'Loading...';
                            ans.textContent = '';
                            src.textContent = '';
                            try{
                                const q = document.getElementById('q').value;
                                const resp = await fetch('/ask', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({question: q}) });
                                if(!resp.ok){
                                    const txt = await resp.text();
                                    status.textContent = 'Error: ' + txt;
                                    return;
                                }
                                const data = await resp.json();
                                status.textContent = '';
                                ans.textContent = data.answer || '';
                                const sources = data.sources || [];
                                src.innerHTML = sources.map((s, i) => {
                                    const m = s.metadata || {};
                                    const title = `${i+1}. ${m.address || 'Unknown address'} | ${m.type_standardized || ''} | £${m.price || ''}`;
                                    const doc = s.document || '';
                                    return `<div><strong>${title}</strong><br><em>${doc}</em></div>`;
                                }).join('<hr>');
                            }catch(e){
                                status.textContent = 'Request failed: ' + e;
                            }
                        }
                    </script>
                </body>
                </html>
                """
        )


def build_prompt(context_items: List[Dict[str, Any]], user_question: str) -> str:
    # Create a structured, instruction-tuned prompt to minimize hallucinations.
    context_sections = []
    max_chars =  int(os.getenv("PROMPT_DOC_CHARS", "1200"))
    for i, item in enumerate(context_items, start=1):
        metadata = item.get("metadata", {})
        document = item.get("document", "")
        if isinstance(document, str) and len(document) > max_chars:
            document = document[:max_chars] + " …"
        # Select common property fields for clarity
        fields = [
            ("type_standardized", metadata.get("type_standardized")),
            ("address", metadata.get("address")),
            ("bedrooms", metadata.get("bedrooms")),
            ("bathrooms", metadata.get("bathrooms")),
            ("price", metadata.get("price")),
            ("flood_risk", metadata.get("flood_risk")),
            ("crime_score_weight", metadata.get("crime_score_weight")),
            ("is_new_home", metadata.get("is_new_home")),
            ("price_category", metadata.get("price_category")),
        ]
        meta_str = "; ".join([f"{k}: {v}" for k, v in fields if v is not None])
        context_sections.append(
            f"[Item {i}]\nSummary: {document}\nDetails: {meta_str}"
        )

    context_block = "\n\n".join(context_sections) if context_sections else "(no results)"
    prompt = (
        "You are an expert real estate assistant. Based ONLY on the following property information, "
        "answer the user's question. Do not use any external knowledge. If the context does not contain "
        "the answer, say that you cannot find the information.\n\n"
        f"--- CONTEXT ---\n{context_block}\n\n"
        f"--- USER QUESTION ---\n{user_question}\n\n"
        "--- ANSWER ---\n"
    )
    return prompt


@app.get("/status")
def status():
    """Return basic runtime/config status for the UI."""
    # Load data rows count (if available)
    try:
        _load_data_if_needed()
        data_rows = len(_DATA_ROWS)
    except Exception:
        data_rows = None

    # Vector count from Pinecone index, if supported
    vector_count: Optional[int] = None
    try:
        stats = index.describe_index_stats()
        if isinstance(stats, dict):
            # v3 typically has {'namespaces': {...}, 'dimension': X, 'indexFullness': Y}
            ns = stats.get("namespaces", {}) or {}
            total = 0
            for ns_name, ns_stats in ns.items():
                if isinstance(ns_stats, dict) and "vectorCount" in ns_stats:
                    total += int(ns_stats.get("vectorCount", 0))
            # Some deployments may expose 'totalVectorCount'
            if total == 0 and "totalVectorCount" in stats:
                total = int(stats.get("totalVectorCount") or 0)
            vector_count = total if total > 0 else None
    except Exception:
        pass

    return {
        "pinecone_index": PINECONE_INDEX,
        "pinecone_cloud": PINECONE_CLOUD,
        "pinecone_region": PINECONE_REGION,
        "index_dimension": INDEX_DIM,
        "vector_count": vector_count,
        "embedding_model": EMBED_MODEL_NAME,
        "generative_model": GENERATIVE_MODEL_NAME,
        "prompt_doc_chars": int(os.getenv("PROMPT_DOC_CHARS", "1200")),
        "top_k": 12,
        "data_rows": data_rows,
    }


# ----------------------
# Deterministic analytics
# ----------------------
DATA_PATH = ROOT / "data" / "property_data_cleaned.csv"
_DATA_ROWS: List[Dict[str, Any]] = []

def _to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip().replace("£", "").replace(",", "")
        return float(s) if s else None
    except Exception:
        return None

def _to_int(val: Any) -> Optional[int]:
    try:
        f = _to_float(val)
        return int(f) if f is not None else None
    except Exception:
        return None

def _to_bool(val: Any) -> Optional[bool]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("true", "yes", "y", "1"): return True
    if s in ("false", "no", "n", "0"): return False
    return None

def _load_data_if_needed() -> None:
    global _DATA_ROWS
    if _DATA_ROWS or not DATA_PATH.exists():
        return
    import csv
    with open(DATA_PATH, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        rows = []
        for r in rdr:
            # Normalize fields we care about
            r = dict(r)
            r["price_num"] = _to_float(r.get("price"))
            r["bedrooms_num"] = _to_int(r.get("bedrooms"))
            r["bathrooms_num"] = _to_int(r.get("bathrooms"))
            r["is_new_home_bool"] = _to_bool(r.get("is_new_home"))
            rows.append(r)
        _DATA_ROWS = rows

def _make_doc_from_row(row: Dict[str, Any]) -> str:
    bt = row.get("type_standardized", "")
    addr = row.get("address", "")
    bed = row.get("bedrooms_num") or row.get("bedrooms") or "?"
    bath = row.get("bathrooms_num") or row.get("bathrooms") or "?"
    price = row.get("price_num") or row.get("price") or "?"
    try:
        if isinstance(price, (int, float)):
            price = f"£{price:,.0f}"
    except Exception:
        pass
    return f"A {bed} bedroom, {bath} bathroom {bt} at {addr} priced at {price}."

def _format_money(val: Optional[float]) -> str:
    if val is None: return "£?"
    try:
        return f"£{val:,.0f}"
    except Exception:
        return str(val)

def _detect_aggregate_intent(q: str) -> Optional[str]:
    s = q.lower()
    if any(k in s for k in ["average", "avg", "mean"]):
        return "average"
    if any(k in s for k in ["median"]):
        return "median"
    if any(k in s for k in ["minimum", "min", "cheapest", "lowest"]):
        return "min"
    if any(k in s for k in ["maximum", "max", "most expensive", "highest"]):
        return "max"
    if any(k in s for k in ["count", "how many", "number of"]):
        return "count"
    if any(k in s for k in ["sum", "total cost", "total price"]):
        return "sum"
    return None

def _extract_filters(q: str) -> Dict[str, Any]:
    import re
    s = q.lower()
    filters: Dict[str, Any] = {}
    # bedrooms: "2 bed", "3 bedroom"
    m = re.search(r"(\d+)\s*(bed|bedroom)s?", s)
    if m:
        filters["bedrooms_num"] = int(m.group(1))
    # bathrooms
    m = re.search(r"(\d+)\s*(bath|bathroom)s?", s)
    if m:
        filters["bathrooms_num"] = int(m.group(1))
    # type keywords
    for t in ["flat", "apartment", "house", "studio", "bungalow", "duplex"]:
        if t in s:
            filters["type_standardized"] = t
            break
    # new home
    if "new home" in s or "new build" in s:
        filters["is_new_home_bool"] = True
    # address substring heuristic after "in" or "at"
    m = re.search(r"\b(?:in|at)\s+([a-z0-9\-\s,]+)$", s)
    if m:
        filters["address_contains"] = m.group(1).strip()
    return filters

def _apply_filters(rows: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    def ok(r: Dict[str, Any]) -> bool:
        # bedrooms exact if specified
        if "bedrooms_num" in filters:
            if (r.get("bedrooms_num") or 0) != filters["bedrooms_num"]:
                return False
        if "bathrooms_num" in filters:
            if (r.get("bathrooms_num") or 0) != filters["bathrooms_num"]:
                return False
        if "type_standardized" in filters:
            tv = str(r.get("type_standardized", "")).lower()
            if filters["type_standardized"] not in tv:
                return False
        if "is_new_home_bool" in filters:
            if (r.get("is_new_home_bool") is not None) and (r.get("is_new_home_bool") != filters["is_new_home_bool"]):
                return False
        if "address_contains" in filters:
            av = str(r.get("address", "")).lower()
            if filters["address_contains"] not in av:
                return False
        return True
    return [r for r in rows if ok(r)]

def _aggregate_answer(q: str) -> Optional[Dict[str, Any]]:
    """Return structured answer if the question is numeric/aggregate. Else None."""
    intent = _detect_aggregate_intent(q)
    if not intent:
        return None
    _load_data_if_needed()
    if not _DATA_ROWS:
        return {
            "answer": "I couldn't load the dataset to compute this.",
            "sources": [],
        }
    filters = _extract_filters(q)
    rows = _apply_filters(_DATA_ROWS, filters)
    priced = [r for r in rows if _to_float(r.get("price_num")) is not None]
    n = len(priced)
    if n == 0:
        return {"answer": "No matching properties found for your criteria.", "sources": []}

    prices = [_to_float(r.get("price_num")) for r in priced]
    prices = [p for p in prices if p is not None]

    ans_text = ""
    src_rows: List[Dict[str, Any]] = []
    if intent == "count":
        ans_text = f"Found {n} matching properties."
        src_rows = priced[:3]
    elif intent == "sum":
        total = sum(prices)
        ans_text = f"Total price for {n} matching properties is {_format_money(total)}."
        src_rows = priced[:3]
    elif intent == "average":
        avg = sum(prices) / len(prices)
        ans_text = f"Average price over {n} matching properties is {_format_money(avg)}."
        src_rows = priced[:3]
    elif intent == "median":
        srt = sorted(prices)
        mid = len(srt) // 2
        median = (srt[mid] if len(srt) % 2 == 1 else (srt[mid - 1] + srt[mid]) / 2)
        ans_text = f"Median price over {n} matching properties is {_format_money(median)}."
        src_rows = priced[:3]
    elif intent == "min":
        best = min(priced, key=lambda r: r.get("price_num") or 0)
        ans_text = f"Cheapest among {n} matching properties is {_format_money(best.get('price_num'))} at {best.get('address','unknown')}."
        src_rows = [best]
    elif intent == "max":
        best = max(priced, key=lambda r: r.get("price_num") or 0)
        ans_text = f"Most expensive among {n} matching properties is {_format_money(best.get('price_num'))} at {best.get('address','unknown')}."
        src_rows = [best]
    else:
        return None

    sources = []
    for r in src_rows:
        sources.append({
            "document": _make_doc_from_row(r),
            "metadata": r,
            "score": 1.0,
        })
    return {"answer": ans_text, "sources": sources}


# ---------------------------------------------
# Advanced analytics: rank areas by given metric
# ---------------------------------------------

_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

def _extract_postcode_outcode(address: Optional[str]) -> Optional[str]:
    if not address:
        return None
    s = str(address).upper()
    # Full UK postcode: OUTCODE INCODE e.g., SW1A 1AA -> capture OUTCODE
    m = re.search(r"\b([A-Z]{1,2}\d{1,2}[A-Z]?)\s*\d[A-Z]{2}\b", s)
    if m:
        return m.group(1)
    # Outcode only as fallback (avoid single letters or pure numbers)
    m = re.search(r"\b([A-Z]{1,2}\d{1,2}[A-Z]?)\b", s)
    if m:
        return m.group(1)
    return None

def _infer_area_from_address(address: Optional[str]) -> Optional[str]:
    if not address:
        return None
    # Prefer postcode outcode; else use last comma-separated token as district
    out = _extract_postcode_outcode(address)
    if out:
        return out
    parts = [p.strip() for p in str(address).split(',') if p.strip()]
    cand = parts[-1] if parts else None
    if not cand:
        return None
    # Normalize candidate: remove trailing numbers and extra words like "United Kingdom"
    cand = re.sub(r"\b(united\s+kingdom|uk|england)\b", "", cand, flags=re.I).strip()
    # Keep up to 3 words; drop if too short or mostly numeric
    words = [w for w in re.split(r"\s+", cand) if w]
    cand = " ".join(words[:3])
    if not cand or len(re.sub(r"[^A-Za-z]", "", cand)) < 3:
        return None
    return cand

def _detect_topn(s: str, default_n: int = 5) -> Tuple[int, bool, Optional[str]]:
    """Return (N, explicit, mode) where explicit indicates if N was explicitly requested,
    and mode is one of {"top", "bottom", None} for wording.
    """
    # bottom/least/lowest N
    m = re.search(r"(?:bottom|least|lowest)\s+(\d{1,2})", s)
    if m:
        try:
            n = int(m.group(1))
            return (max(1, min(50, n)), True, "bottom")
        except Exception:
            pass
    m = re.search(r"(?:bottom|least|lowest)\s+([a-z]+)", s)
    if m and m.group(1).lower() in _NUM_WORDS:
        return (max(1, min(50, _NUM_WORDS[m.group(1).lower()])), True, "bottom")

    # top/first/best N
    m = re.search(r"(?:top|first|best)\s+(\d{1,2})", s)
    if m:
        try:
            n = int(m.group(1))
            return (max(1, min(50, n)), True, "top")
        except Exception:
            pass
    m = re.search(r"(?:top|first|best)\s+([a-z]+)", s)
    if m and m.group(1).lower() in _NUM_WORDS:
        return (max(1, min(50, _NUM_WORDS[m.group(1).lower()])), True, "top")

    # pattern: "3 areas" or "10 postcodes"
    m = re.search(r"(\d{1,2})\s+(?:areas?|postcodes?|boroughs?|districts?)", s)
    if m:
        try:
            n = int(m.group(1))
            return (max(1, min(50, n)), True, None)
        except Exception:
            pass
    return (default_n, False, None)

def _detect_area_ranking_intent(q: str) -> Optional[Dict[str, Any]]:
    s = q.lower()
    # Must be asking about areas/postcodes/boroughs/districts OR explicitly requesting a top/bottom N
    area_hint = any(k in s for k in [
        " area", " areas", "postcode", "postcodes", "borough", "boroughs", "district", "districts",
        "neighbourhood", "neighborhood"
    ])
    metric = None
    order = None  # "asc" or "desc"

    if any(k in s for k in ["crime", "safety", "safest", "unsafe", "dangerous"]):
        metric = "crime_score_weight"
        # safest/lowest => asc; most/highest/worst => desc
        if any(k in s for k in ["safest", "least", "lowest", "low"]):
            order = "asc"
        elif any(k in s for k in ["most", "highest", "worst", "dangerous"]):
            order = "desc"
        else:
            order = "desc"
    if metric is None and any(k in s for k in ["flood", "risk", "flooding"]):
        metric = "flood_risk"
        if any(k in s for k in ["least", "lowest", "safest", "low"]):
            order = "asc"
        elif any(k in s for k in ["most", "highest", "worst", "high"]):
            order = "desc"
        else:
            order = "desc"
    if metric is None and any(k in s for k in ["price", "expensive", "cheapest", "affordable", "cost"]):
        metric = "price"
        if any(k in s for k in ["cheapest", "affordable", "least", "lowest"]):
            order = "asc"
        elif any(k in s for k in ["most", "expensive", "highest"]):
            order = "desc"
        else:
            order = "desc"
    n, n_explicit, mode = _detect_topn(s, default_n=5)
    # Only trigger area ranking when explicitly about areas OR an explicit N (top/bottom) is requested
    if not area_hint and not n_explicit:
        return None
    return {"metric": metric or "price", "order": order or "desc", "top_n": n, "n_explicit": n_explicit, "mode": mode}

def _rank_areas(rows: List[Dict[str, Any]], metric: str, order: str, top_n: int) -> List[Dict[str, Any]]:
    # Group rows by inferred area
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        area = _infer_area_from_address(r.get("address"))
        if not area:
            continue
        groups.setdefault(area, []).append(r)
    scored: List[Dict[str, Any]] = []
    for area, items in groups.items():
        vals: List[float] = []
        if metric == "price":
            for it in items:
                v = _to_float(it.get("price_num"))
                if v is not None:
                    vals.append(v)
        else:
            for it in items:
                v = _to_float(it.get(metric))
                if v is not None:
                    vals.append(v)
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        scored.append({"area": area, "value": avg, "count": len(items)})
    reverse = (order == "desc")
    scored.sort(key=lambda x: (x["value"]) , reverse=reverse)
    return scored[:top_n]

def _area_ranking_answer(q: str) -> Optional[Dict[str, Any]]:
    spec = _detect_area_ranking_intent(q)
    if not spec:
        return None
    _load_data_if_needed()
    if not _DATA_ROWS:
        return {
            "answer": "I couldn't load the dataset to compute this.",
            "sources": [],
        }
    filters = _extract_filters(q)
    rows = _apply_filters(_DATA_ROWS, filters)
    if not rows:
        return {"answer": "No matching properties found for your criteria.", "sources": []}

    ranked = _rank_areas(rows, spec["metric"], spec["order"], spec["top_n"])
    if not ranked:
        return {"answer": "I couldn't compute rankings due to missing values.", "sources": []}

    metric_name = {
        "crime_score_weight": "crime score",
        "flood_risk": "flood risk",
        "price": "average price",
    }.get(spec["metric"], spec["metric"])

    order_phrase = "highest" if spec["order"] == "desc" else "lowest"
    # Header wording: prefer explicit Top/Bottom if user asked, else neutral "Areas ranked by ..."
    if spec.get("n_explicit") and spec.get("mode") == "top":
        header = f"Top {len(ranked)} areas by {order_phrase} {metric_name}:"
    elif spec.get("n_explicit") and spec.get("mode") == "bottom":
        header = f"Bottom {len(ranked)} areas by {order_phrase} {metric_name}:"
    else:
        header = f"Areas ranked by {order_phrase} {metric_name} (showing {len(ranked)}):"

    lines = []
    for i, r in enumerate(ranked, start=1):
        val = r["value"]
        if spec["metric"] == "price":
            v_str = _format_money(val)
        else:
            try:
                v_str = f"{val:.3f}"
            except Exception:
                v_str = str(val)
        lines.append(f"{i}. {r['area']} — {v_str} (n={r['count']})")

    # Pick one representative source row per top area
    area_to_row: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        area = _infer_area_from_address(r.get("address"))
        if area in [x["area"] for x in ranked] and area not in area_to_row:
            area_to_row[area] = r
        if len(area_to_row) >= len(ranked):
            break
    sources = []
    for area in [x["area"] for x in ranked]:
        rr = area_to_row.get(area)
        if rr:
            sources.append({
                "document": _make_doc_from_row(rr),
                "metadata": rr,
                "score": 1.0,
            })

    answer = header + "\n" + "\n".join(lines)
    return {"answer": answer, "sources": sources}


###############################
# Area comparison (deterministic)
###############################

def _normalize_area_token(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).strip().lower()

def _row_matches_area(row: Dict[str, Any], area_token: str) -> bool:
    addr = str(row.get("address", ""))
    if not addr:
        return False
    # Compare postcode outcode if token looks like one
    out = _extract_postcode_outcode(addr)
    if out and _normalize_area_token(out) == _normalize_area_token(area_token):
        return True
    # Fallback: substring match on address
    return _normalize_area_token(area_token) in _normalize_area_token(addr)

def _detect_comparison_intent(q: str) -> Optional[Dict[str, Any]]:
    s = q.strip()
    sl = s.lower()
    # Identify metric by keywords
    metric: Optional[str] = None
    if any(k in sl for k in ["crime", "safer", "unsafe", "dangerous", "safety"]):
        metric = "crime_score_weight"
    elif any(k in sl for k in ["flood", "flooding", "risk"]):
        metric = "flood_risk"
    elif any(k in sl for k in ["price", "cheaper", "expensive", "cost", "affordable"]):
        metric = "price"

    # Detect explicit compare phrasing
    comp_hint = any(k in sl for k in ["compare", "vs", "versus", "than", "or", "and"])
    if not comp_hint:
        return None

    # Extract two or more targets: split on vs/versus/or/and/comma
    text = s
    text = re.sub(r"^(compare|which\s+is|is)\s+", "", text, flags=re.I)
    text = text.strip().rstrip('?')
    parts = re.split(r"\s+(?:vs|versus|than|or|and)\s+|,\s*", text, flags=re.I)
    bad = set(["safer", "cheaper", "more", "most", "less", "highest", "lowest", "for", "crime", "price", "flood", "risk", "area", "areas", "which", "is"])
    cand = [p.strip() for p in parts if p and _normalize_area_token(p) not in bad and len(p.strip()) > 1]
    # Keep distinct order
    seen = set()
    targets: List[str] = []
    for c in cand:
        key = _normalize_area_token(c)
        if key not in seen:
            seen.add(key)
            targets.append(c)
    if len(targets) < 2:
        return None
    targets = targets[:3]
    return {"metric": metric or "price", "targets": targets}

def _compute_area_value(rows: List[Dict[str, Any]], area_token: str, metric: str) -> Tuple[Optional[float], int]:
    vals: List[float] = []
    for r in rows:
        if not _row_matches_area(r, area_token):
            continue
        if metric == "price":
            v = _to_float(r.get("price_num"))
        else:
            v = _to_float(r.get(metric))
        if v is not None:
            vals.append(v)
    if not vals:
        return (None, 0)
    return (sum(vals) / len(vals), len(vals))

def _comparison_answer(q: str) -> Optional[Dict[str, Any]]:
    spec = _detect_comparison_intent(q)
    if not spec:
        return None
    _load_data_if_needed()
    if not _DATA_ROWS:
        return {"answer": "I couldn't load the dataset to compute this.", "sources": []}
    filters = _extract_filters(q)
    rows = _apply_filters(_DATA_ROWS, filters)
    metric = spec["metric"]
    targets = spec["targets"]

    results: List[Tuple[str, Optional[float], int]] = []
    for t in targets:
        val, cnt = _compute_area_value(rows, t, metric)
        results.append((t, val, cnt))

    # If fewer than 2 valid values, fall back to RAG
    valid = [r for r in results if r[1] is not None and r[2] > 0]
    if len(valid) < 2:
        return None

    def fmt_val(v: Optional[float]) -> str:
        if v is None:
            return "?"
        if metric == "price":
            return _format_money(v)
        try:
            return f"{v:.3f}"
        except Exception:
            return str(v)

    metric_name = {
        "crime_score_weight": "crime score",
        "flood_risk": "flood risk",
        "price": "average price",
    }.get(metric, metric)

    lines = [f"Comparison by {metric_name}:"]
    for name, val, cnt in results:
        lines.append(f"- {name}: {fmt_val(val)} (n={cnt})")

    # If exactly two valid, add difference and quick verdict
    if len(valid) == 2:
        (n1, v1, c1), (n2, v2, c2) = valid[0], valid[1]
        diff = (v1 - v2) if (v1 is not None and v2 is not None) else None
        if diff is not None:
            try:
                pd = (diff / v2) * 100 if v2 else None
            except Exception:
                pd = None
            if metric in ("crime_score_weight", "flood_risk"):
                better = n1 if v1 < v2 else n2
                verdict = f"Verdict: {better} appears safer (lower {metric_name})."
            elif metric == "price":
                better = n1 if v1 < v2 else n2
                verdict = f"Verdict: {better} appears cheaper (lower {metric_name})."
            else:
                better = n1 if v1 > v2 else n2
                verdict = f"Verdict: {better} has a higher {metric_name}."
            diff_str = f"Difference: {fmt_val(v1)} vs {fmt_val(v2)}"
            if pd is not None:
                try:
                    diff_str += f" ({pd:+.1f}% vs {n2})"
                except Exception:
                    pass
            lines.append(diff_str)
            lines.append(verdict)

    # Representative sources: pick first matching row for each target
    sources: List[Dict[str, Any]] = []
    for t in targets:
        for r in rows:
            if _row_matches_area(r, t):
                sources.append({
                    "document": _make_doc_from_row(r),
                    "metadata": r,
                    "score": 1.0,
                })
                break

    return {"answer": "\n".join(lines), "sources": sources}
def _try_generate_with_fallback(prompt: str) -> str:
    """Generate with the selected model; on 404/429, try preferred fallbacks quickly."""
    candidates = []
    env_model = os.getenv("GENAI_MODEL")
    if env_model:
        candidates.append(env_model)
    # Ensure primary selected model is first
    if GENERATIVE_MODEL_NAME not in candidates:
        candidates.append(GENERATIVE_MODEL_NAME)
    # Add common light/available fallbacks (dedup while preserving order)
    for m in [
        "models/gemini-1.5-flash",
        "models/gemini-1.5-flash-8b",
        "models/gemini-1.0-pro",
    ]:
        if m not in candidates:
            candidates.append(m)

    last_err = None
    for model_name in candidates:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            return (resp.text or "").strip()
        except Exception as e:
            # On rate limit, try next model; avoid long sleeps in API path
            last_err = e
            continue
    # If all failed, raise the last error
    raise last_err if last_err else RuntimeError("Gemini generation failed with no specific error")


def _embed_query(text: str) -> List[float]:
    emb = genai.embed_content(
        model=EMBED_MODEL_NAME,
        content=text,
        task_type="retrieval_query",
        output_dimensionality=INDEX_DIM,
    )
    if isinstance(emb, dict):
        if "embedding" in emb and isinstance(emb["embedding"], dict) and "values" in emb["embedding"]:
            return _adjust_vec(emb["embedding"]["values"], INDEX_DIM)
        if "embedding" in emb and isinstance(emb["embedding"], list):
            return _adjust_vec(emb["embedding"], INDEX_DIM)
    try:
        vec = getattr(emb, "embedding", [])
        if hasattr(vec, "values"):
            return _adjust_vec(vec.values, INDEX_DIM)
        return _adjust_vec(vec, INDEX_DIM)
    except Exception:
        return [0.0] * INDEX_DIM


def search_pinecone(question: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """Encode the query and search Pinecone. Returns list of {document, metadata, score}."""
    q_vec = _embed_query(question)
    results = index.query(vector=q_vec, top_k=top_k, include_metadata=True)
    items = []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        doc = meta.get("document", "")
        score = match.get("score", 0.0)
        items.append({"document": doc, "metadata": meta, "score": score})
    return items


@app.post("/ask", response_model=None)
async def ask(query: Query, background_tasks: BackgroundTasks):
    """
    🏆 PRODUCTION-GRADE PROPERTY INTELLIGENCE ENDPOINT
    
    Features:
    - Advanced query preprocessing and entity extraction
    - Intent classification and routing
    - Multi-turn conversation tracking
    - Hybrid search with caching
    - Sophisticated prompt engineering
    - Professional response formatting
    - Comprehensive error handling
    - Performance metrics and logging
    """
    start_time = time.time()
    question_raw = query.question.strip()
    session_id = query.session_id or "default"
    
    # Validate input
    if not question_raw:
        if _HAS_ADVANCED_RAG:
            return format_error_response("ambiguous_query", "Please provide a question.")
        else:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Log query
    logger.info(f"📥 Query received [session={session_id}]: {question_raw[:100]}")
    _METRICS["total_queries"] += 1
    
    try:
        # ==========================================
        # STAGE 1: QUERY UNDERSTANDING
        # ==========================================
        
        # Step 1.1: Preprocess query
        question_processed = preprocess_query(question_raw)
        
        # Step 1.2: Extract entities
        entities = extract_entities(question_processed)
        logger.info(f"🔍 Entities extracted: {entities}")
        
        # Step 1.3: Resolve coreferences using conversation context
        conversation_history = get_conversation_context(session_id)
        question_resolved = resolve_coreference(question_processed, conversation_history)
        
        # Step 1.4: Classify intent
        intent = classify_intent(question_resolved, entities)
        logger.info(f"🎯 Intent classified: {intent}")
        _METRICS[f"intent_{intent}"] += 1
        
        # Step 1.5: Check for out-of-scope queries
        if _HAS_ADVANCED_RAG:
            out_of_scope_response = handle_out_of_scope(question_resolved)
            if out_of_scope_response:
                update_conversation_context(session_id, "user", question_raw)
                update_conversation_context(session_id, "assistant", out_of_scope_response["answer"])
                return out_of_scope_response
        
        # Step 1.6: Detect ambiguity and generate clarifying questions
        ambiguity_msg = detect_ambiguity(question_resolved, entities)
        clarifying_qs = None
        if ambiguity_msg and _HAS_ADVANCED_RAG:
            clarifying_qs = generate_clarifying_questions(question_resolved, entities, intent)
        
        if clarifying_qs and len(clarifying_qs) > 0:
            logger.info(f"❓ Ambiguous query detected, requesting clarification")
            return {
                "answer": ambiguity_msg,
                "clarifying_questions": clarifying_qs,
                "intent": intent,
                "needs_clarification": True
            }
        
        # ==========================================
        # STAGE 2: DETERMINISTIC ANALYTICS ROUTING
        # ==========================================
        
        # Route to deterministic analytics if applicable (area comparison, ranking, aggregation)
        comp = _comparison_answer(question_resolved)
        if comp is not None:
            update_conversation_context(session_id, "user", question_raw)
            update_conversation_context(session_id, "assistant", comp.get("answer", ""))
            _METRICS["deterministic_responses"] += 1
            return comp
        
        area_rank = _area_ranking_answer(question_resolved)
        if area_rank is not None:
            update_conversation_context(session_id, "user", question_raw)
            update_conversation_context(session_id, "assistant", area_rank.get("answer", ""))
            _METRICS["deterministic_responses"] += 1
            return area_rank
        
        agg = _aggregate_answer(question_resolved)
        if agg is not None:
            update_conversation_context(session_id, "user", question_raw)
            update_conversation_context(session_id, "assistant", agg.get("answer", ""))
            _METRICS["deterministic_responses"] += 1
            return agg
        
        # ==========================================
        # STAGE 3: INTELLIGENT RETRIEVAL
        # ==========================================
        
        logger.info(f"🔎 Starting hybrid search...")
        
        # Use hybrid search with caching and diversity
        retrieved = hybrid_search(
            question_resolved,
            top_k=15,  # Retrieve more for better re-ranking
            use_expansions=query.use_expansions
        )
        
        if not retrieved or len(retrieved) == 0:
            logger.warning(f"⚠️ No results found for query: {question_resolved}")
            _METRICS["no_results"] += 1
            if _HAS_ADVANCED_RAG:
                return format_error_response("no_results")
            else:
                return {"answer": "No relevant properties found.", "sources": []}
        
        logger.info(f"✅ Retrieved {len(retrieved)} documents")
        
        # ==========================================
        # STAGE 4: RE-RANKING
        # ==========================================
        
        reranked = retrieved
        if _HAS_CROSS_ENCODER:
            try:
                logger.info("🔄 Re-ranking with CrossEncoder...")
                reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                pairs = [(question_resolved, item["document"]) for item in retrieved]
                scores = reranker.predict(pairs)
                for item, s in zip(retrieved, scores):
                    item["rerank_score"] = float(s)
                reranked = sorted(retrieved, key=lambda x: x.get("rerank_score", x.get("score", 0.0)), reverse=True)
                logger.info(f"✅ Re-ranked successfully")
            except Exception as e:
                logger.warning(f"⚠️ Re-ranking failed: {e}")
        
        # Select top K after re-ranking
        top_context = reranked[:8]  # Use top 8 for context
        
        # ==========================================
        # STAGE 5: SOPHISTICATED PROMPT ENGINEERING
        # ==========================================
        
        if _HAS_ADVANCED_RAG:
            prompt = build_sophisticated_prompt(
                context_items=top_context,
                user_question=question_resolved,
                intent=intent,
                entities=entities,
                conversation_history=conversation_history
            )
        else:
            # Fallback to basic prompt if advanced module unavailable
            prompt = build_prompt(top_context, question_resolved)
        
        logger.info(f"📝 Prompt built ({len(prompt)} chars)")
        
        # ==========================================
        # STAGE 6: LLM GENERATION
        # ==========================================
        
        logger.info("🤖 Generating response with Gemini...")
        
        try:
            answer = _try_generate_with_fallback(prompt)
            logger.info(f"✅ Response generated ({len(answer)} chars)")
            _METRICS["successful_generations"] += 1
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            _METRICS["failed_generations"] += 1
            raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")
        
        # ==========================================
        # STAGE 7: RESPONSE FORMATTING
        # ==========================================
        
        # Calculate confidence
        if _HAS_ADVANCED_RAG:
            confidence_score = calculate_response_confidence(question_resolved, top_context, intent)
        else:
            confidence_score = max([item.get("score", 0.0) for item in top_context]) if top_context else 0.5
        
        # Format professional response
        if _HAS_ADVANCED_RAG:
            response = format_professional_response(
                query=question_resolved,
                intent=intent,
                answer=answer,
                sources=top_context,
                confidence=confidence_score,
                entities=entities
            )
        else:
            # Basic response format
            response = {
                "answer": answer,
                "sources": top_context,
                "confidence": {"score": confidence_score, "label": "Moderate"},
                "intent": intent
            }
        
        # ==========================================
        # STAGE 8: CONVERSATION TRACKING & METRICS
        # ==========================================
        
        # Update conversation context
        update_conversation_context(session_id, "user", question_raw)
        update_conversation_context(session_id, "assistant", answer)
        
        # Calculate and log latency
        latency = time.time() - start_time
        _METRICS["total_latency"] += latency
        logger.info(f"⏱️ Total latency: {latency:.2f}s")
        
        # Background task: log detailed metrics
        background_tasks.add_task(
            log_query_metrics,
            query=question_raw,
            intent=intent,
            num_results=len(retrieved),
            confidence=confidence_score,
            latency=latency,
            session_id=session_id
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in /ask endpoint: {e}", exc_info=True)
        _METRICS["errors"] += 1
        if _HAS_ADVANCED_RAG:
            return format_error_response("retrieval_failed", str(e))
        else:
            raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# BACKGROUND TASKS & UTILITIES
# ==========================================

def log_query_metrics(query: str, intent: str, num_results: int, confidence: float, latency: float, session_id: str):
    """Background task to log detailed metrics"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "query": query,
        "intent": intent,
        "num_results": num_results,
        "confidence": confidence,
        "latency_seconds": latency
    }
    
    # Write to metrics log
    metrics_file = ROOT / "query_metrics.jsonl"
    try:
        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write metrics: {e}")


@app.get("/metrics")
def get_metrics():
    """
    Endpoint to retrieve system metrics
    """
    total_queries = _METRICS["total_queries"]
    avg_latency = (_METRICS["total_latency"] / _METRICS["retrieval_count"]) if _METRICS["retrieval_count"] > 0 else 0
    cache_hit_rate = (_METRICS["cache_hits"] / total_queries) if total_queries > 0 else 0
    
    return {
        "total_queries": total_queries,
        "cache_hit_rate": f"{cache_hit_rate * 100:.1f}%",
        "avg_retrieval_latency": f"{avg_latency:.2f}s",
        "successful_generations": _METRICS["successful_generations"],
        "failed_generations": _METRICS["failed_generations"],
        "deterministic_responses": _METRICS["deterministic_responses"],
        "no_results_count": _METRICS["no_results"],
        "errors": _METRICS["errors"],
        "intent_breakdown": {
            k.replace("intent_", ""): v
            for k, v in _METRICS.items()
            if k.startswith("intent_")
        }
    }


@app.post("/clear_cache")
def clear_query_cache():
    """Clear query result cache"""
    _QUERY_CACHE.clear()
    logger.info("🗑️ Query cache cleared")
    return {"message": "Cache cleared successfully"}


@app.post("/reset_conversation")
def reset_conversation(session_id: str = "default"):
    """Reset conversation context for a session"""
    if session_id in _CONVERSATION_CONTEXT:
        del _CONVERSATION_CONTEXT[session_id]
    logger.info(f"🔄 Conversation reset for session: {session_id}")
    return {"message": f"Conversation reset for session {session_id}"}
