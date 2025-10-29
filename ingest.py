"""
Data ingestion script for Property RAG (Pinecone + Gemini).

This script reads ./data/property_data_cleaned.csv, constructs text documents for
each property, generates high-quality embeddings using Google's text-embedding-004
(via google-generativeai SDK), and stores them in a Pinecone index for fast vector search.

Key choices for accuracy:
- Use text-embedding-004 (768-dim) for dense retrieval quality.
- Preserve rich metadata per property to feed the LLM precisely.
- Batch embedding + upsert for speed and reliability.

Usage: python ingest.py

Environment variables (set them in a .env file in the project folder):
- GOOGLE_API_KEY: Gemini API key from Google AI Studio
- PINECONE_API_KEY: Pinecone API key
- PINECONE_CLOUD / PINECONE_REGION: For pinecone-client v3 serverless (defaults: aws/us-east-1)
- PINECONE_ENVIRONMENT: Only for pinecone-client v2 (e.g., "gcp-starter")

Note: Place the CSV file at ./data/property_data_cleaned.csv
"""

from pathlib import Path
import os
import time
import csv
from dotenv import load_dotenv
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

# Google Generative AI (direct embeddings)
import google.generativeai as genai

# Pinecone client v3 (serverless)
from typing import Optional
try:
    from pinecone import Pinecone, ServerlessSpec  # v3 client
    _PC_V3 = True
except Exception:
    Pinecone = None
    ServerlessSpec = None
    _PC_V3 = False


def main():
    repo_root = Path(__file__).parent
    data_path = repo_root / "data" / "property_data_cleaned.csv"

    # Load environment
    load_dotenv(repo_root / ".env", override=True)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")

    if not google_api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set. Create a .env file in the project folder and set your key.")
    if not pinecone_api_key:
        raise EnvironmentError("PINECONE_API_KEY not set. Create a .env file in the project folder and set your key.")
    # PINECONE_CLOUD and PINECONE_REGION have defaults; user can override in .env

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}. Please place your CSV at this location.")

    print(f"Loading data from: {data_path}")
    # Read CSV without pandas for Python 3.13 compatibility (avoid numpy dependency)
    with open(data_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Ensure the expected columns exist; if not, proceed with best-effort
    expected_cols = [
        "type_standardized",
        "bedrooms",
        "bathrooms",
        "price",
        "address",
        "crime_score_weight",
        "flood_risk",
        "is_new_home",
        "property_type_full_description",
        "price_category",
        "text_description",
    ]
    header = rows[0].keys() if rows else []
    missing = [c for c in expected_cols if c not in header]
    if missing:
        print(f"Warning: Missing expected columns in CSV: {missing}. Missing values will be treated as empty strings.")

    # Build documents
    def make_doc(row: dict):
        def get_val(key):
            return row.get(key, "") if row.get(key, "") is not None else ""

        bedrooms_val = 0
        try:
            raw_bed = get_val("bedrooms")
            bedrooms_val = int(raw_bed) if str(raw_bed).strip() != "" else 0
        except Exception:
            bedrooms_val = 0
        bathrooms_val = 0
        try:
            raw_bath = get_val("bathrooms")
            bathrooms_val = int(raw_bath) if str(raw_bath).strip() != "" else 0
        except Exception:
            bathrooms_val = 0

        price_val = get_val("price")
        property_type = get_val("type_standardized")
        address = get_val("address")
        description = get_val("text_description")
        type_full = get_val("property_type_full_description")
        price_cat = get_val("price_category")
        crime_score = get_val("crime_score_weight")
        flood_risk = get_val("flood_risk")
        is_new_home = get_val("is_new_home")

        return (
            f"A {bedrooms_val} bedroom, {bathrooms_val} bathroom {property_type} at {address} priced at £{price_val}. "
            f"New home: {is_new_home}. Flood risk: {flood_risk}. Crime score: {crime_score}. "
            f"Type details: {type_full}. Price category: {price_cat}. "
            f"Description: {description}"
        )

    docs = []
    metadatas = []
    ids = []

    print("Constructing documents for embedding...")
    for idx, row in enumerate(rows):
        doc = make_doc(row)
        docs.append(doc)
        # Store all columns as metadata, sanitizing values for Pinecone
        meta = {}
        for k, v in row.items():
            if v is None:
                continue
            if isinstance(v, str):
                vs = v.strip()
                if vs == "" or vs.lower() in ("na", "n/a", "null", "none"):
                    continue
                meta[k] = vs
            elif isinstance(v, (int, float, bool)):
                meta[k] = v
            elif isinstance(v, list):
                meta[k] = [str(x) for x in v]
            else:
                meta[k] = str(v)
        meta["source_id"] = int(idx)
        metadatas.append(meta)
        ids.append(str(int(idx)))

    # Initialize Gemini embeddings directly via google-generativeai
    print("Configuring Gemini client for embeddings: text-embedding-004")
    genai.configure(api_key=google_api_key)

    # Initialize Pinecone
    # Diagnostics: show pinecone package version and selected client path
    try:
        import pinecone as pine_pkg
        print(f"pinecone-client version: {getattr(pine_pkg, '__version__', 'unknown')} | v3 symbols available: {_PC_V3}")
    except Exception:
        pass

    index_name = os.getenv("PINECONE_INDEX", "properties-index")
    metric = "cosine"
    # Determine embedding dimension from existing index or override via EMBED_DIM
    env_dim = os.getenv("EMBED_DIM")
    try:
        env_dim_int = int(env_dim) if env_dim else None
    except Exception:
        env_dim_int = None
    embed_dim = None

    if _PC_V3:
        print("Connecting to Pinecone (serverless v3)...")
        pc = Pinecone(api_key=pinecone_api_key)
        existing = [idx["name"] for idx in pc.list_indexes()]
        if index_name in existing:
            # Try to fetch index dimension
            try:
                desc = pc.describe_index(index_name)
                embed_dim = getattr(desc, "dimension", None)
                if embed_dim is None and isinstance(desc, dict):
                    embed_dim = desc.get("dimension") or (desc.get("spec", {}) or {}).get("dimension")
                if embed_dim:
                    print(f"Using existing Pinecone index '{index_name}' (dim={embed_dim}).")
                else:
                    print(f"Using existing Pinecone index '{index_name}'.")
            except Exception:
                print(f"Using existing Pinecone index '{index_name}'.")
        else:
            # Create index with desired dimension (default 768)
            embed_dim = env_dim_int or 768
            print(f"Creating Pinecone index '{index_name}' (dim={embed_dim}, metric={metric}, cloud={pinecone_cloud}, region={pinecone_region})...")
            # Prefer ServerlessSpec if available
            spec = ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region) if ServerlessSpec else {"serverless": {"cloud": pinecone_cloud, "region": pinecone_region}}
            pc.create_index(name=index_name, dimension=embed_dim, metric=metric, spec=spec)
            # Wait until index is ready
            while True:
                desc = pc.describe_index(index_name)
                state = getattr(desc, "status", {}).get("ready") or getattr(desc, "status", {}).get("state")
                if state in (True, "Ready", "ready"):
                    break
                print("Waiting for index to be ready...")
                time.sleep(2)

        index = pc.Index(index_name)
        if embed_dim is None:
            embed_dim = env_dim_int or 768
    else:
        # Legacy v2 fallback (requires PINECONE_ENVIRONMENT)
        import pinecone as pine_v2
        pine_env = os.getenv("PINECONE_ENVIRONMENT")
        if not pine_env:
            raise EnvironmentError(
                "Detected pinecone-client v2 but PINECONE_ENVIRONMENT is not set. Either set PINECONE_ENVIRONMENT "
                "(e.g., 'gcp-starter') in .env or upgrade to pinecone-client>=3 to use serverless cloud/region."
            )
        print(f"Connecting to Pinecone v2 (env={pine_env})...")
        pine_v2.init(api_key=pinecone_api_key, environment=pine_env)
        existing = pine_v2.list_indexes()
        if index_name in existing:
            print(f"Using existing Pinecone index '{index_name}'.")
            try:
                desc = pine_v2.describe_index(index_name)
                # v2 returns dict
                embed_dim = (desc.get("dimension") if isinstance(desc, dict) else getattr(desc, "dimension", None)) or env_dim_int or 768
            except Exception:
                embed_dim = env_dim_int or 768
        else:
            embed_dim = env_dim_int or 768
            print(f"Creating Pinecone index '{index_name}' (dim={embed_dim}, metric={metric})...")
            pine_v2.create_index(name=index_name, dimension=embed_dim, metric=metric)
            # No ready loop available consistently; brief wait
            time.sleep(5)
        index = pine_v2.Index(index_name)

    # Compute embeddings in batches and upsert to Pinecone
    batch_size = int(os.getenv("BATCH_SIZE", "128"))
    workers = int(os.getenv("EMBED_WORKERS", "8"))
    embed_timeout = float(os.getenv("EMBED_TIMEOUT_SEC", "30"))
    max_retries = int(os.getenv("EMBED_MAX_RETRIES", "3"))
    print(
        f"Generating embeddings and upserting to Pinecone (batch_size={batch_size}, workers={workers}, timeout={embed_timeout}s)..."
    )

    def _adjust_vec(vec: List[float], dim: int) -> List[float]:
        if not isinstance(vec, list):
            vec = []
        ln = len(vec)
        if ln == dim:
            return vec
        if ln > dim:
            return vec[:dim]
        # pad with zeros
        return vec + [0.0] * (dim - ln)

    def _embed_one(text: str) -> List[float]:
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                emb = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document",
                    output_dimensionality=embed_dim,
                    request_options={"timeout": embed_timeout},
                )
                # Parse embedding from response
                if isinstance(emb, dict):
                    if "embedding" in emb and isinstance(emb["embedding"], dict) and "values" in emb["embedding"]:
                        vec = emb["embedding"]["values"]
                    elif "embedding" in emb and isinstance(emb["embedding"], list):
                        vec = emb["embedding"]
                    else:
                        vec = []
                else:
                    try:
                        vec = getattr(emb, "embedding", [])
                        if hasattr(vec, "values"):
                            vec = vec.values
                    except Exception:
                        vec = []
                return _adjust_vec(vec, embed_dim)
            except Exception as e:
                last_err = e
                # simple backoff
                time.sleep(1.5 * attempt)
        # If all retries failed, return zero vector to keep pipeline moving
        print(f"Warning: embedding failed after {max_retries} attempts: {last_err}")
        return [0.0] * embed_dim

    for i in tqdm(range(0, len(docs), batch_size), desc="Ingesting"):
        batch_docs = docs[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]

        # Parallel embed the current batch
        with ThreadPoolExecutor(max_workers=workers) as ex:
            vecs = list(ex.map(_embed_one, batch_docs))

        vectors = []
        for vid, vec, meta, doc in zip(batch_ids, vecs, batch_meta, batch_docs):
            # Store the original text as part of metadata for later prompt context
            meta = dict(meta)
            meta["document"] = doc
            vectors.append({"id": vid, "values": vec, "metadata": meta})

        index.upsert(vectors=vectors)

    print("✅ Ingestion complete. Vectors stored in Pinecone.")


if __name__ == "__main__":
    main()
