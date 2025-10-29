"""
🏡 PropertyRAG – Professional Real Estate AI Platform
A modern, feature-rich interface for intelligent property search and analysis
Powered by Gemini + Pinecone RAG Architecture
Usage: streamlit run app.py
"""

import streamlit as st
import requests
from requests.exceptions import RequestException
from datetime import datetime
import json
import time
from typing import Dict, List, Any

# ==============================
# ⚙️ CONFIGURATION
# ==============================
BACKEND_URL = "http://127.0.0.1:8000/ask"
STATUS_URL = "http://127.0.0.1:8000/status"

st.set_page_config(
    page_title="PropertyRAG | AI-Powered Real Estate Intelligence",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==============================
# 🎨 SESSION STATE INITIALIZATION
# ==============================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "theme": "dark",
        "history": [],
        "favorites": [],
        "search_history": [],
        "bookmarks": set(),
        "show_onboarding": True if "show_onboarding" not in st.session_state else st.session_state.show_onboarding,
        "comparison_mode": False,
        "comparison_items": [],
        "filters": {
            "price_min": 0,
            "price_max": 10000000,
            "bedrooms": "Any",
            "property_type": "All"
        },
        "settings": {
            "model": "gemini-1.5-flash",
            "response_length": "detailed",
            "show_sources": True,
            "auto_suggestions": True
        }
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==============================
# 🎨 CUSTOM CSS FRAMEWORK
# ==============================
def get_custom_css():
    """Returns comprehensive custom CSS for the entire application - Dark Mode"""
    # Dark mode theme variables
    primary_bg = "#0a0e1a"
    secondary_bg = "#1a1f35"
    card_bg = "#1e2538"
    accent_color = "#6366f1"
    accent_hover = "#818cf8"
    text_primary = "#f8fafc"
    text_secondary = "#94a3b8"
    border_color = "#334155"
    gradient_1 = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    gradient_2 = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
    gradient_3 = "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"
    shadow = "0 10px 40px rgba(0,0,0,0.3)"
    glass_bg = "rgba(30, 37, 56, 0.7)"
    
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* ========== BASE STYLES ========== */
        * {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .main {{
            background: {primary_bg};
            color: {text_primary};
            padding: 0 !important;
        }}
        
        .block-container {{
            padding: 2rem 3rem !important;
            max-width: 1400px;
        }}
        
        /* ========== GLASSMORPHISM ========== */
        .glass {{
            background: {glass_bg};
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid {border_color};
            border-radius: 16px;
            box-shadow: {shadow};
        }}
        
        /* ========== HERO SECTION ========== */
        .hero {{
            background: {gradient_1};
            border-radius: 24px;
            padding: 4rem 3rem;
            text-align: center;
            margin-bottom: 3rem;
            position: relative;
            overflow: hidden;
            box-shadow: {shadow};
        }}
        
        .hero::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: float 20s infinite linear;
        }}
        
        @keyframes float {{
            0%, 100% {{ transform: translate(0, 0) rotate(0deg); }}
            33% {{ transform: translate(30px, -30px) rotate(120deg); }}
            66% {{ transform: translate(-20px, 20px) rotate(240deg); }}
        }}
        
        .hero h1 {{
            font-size: 3.5rem;
            font-weight: 800;
            color: white;
            margin-bottom: 1rem;
            text-shadow: 0 4px 12px rgba(0,0,0,0.3);
            position: relative;
            z-index: 1;
        }}
        
        .hero p {{
            font-size: 1.3rem;
            color: rgba(255,255,255,0.95);
            margin-bottom: 2rem;
            position: relative;
            z-index: 1;
        }}
        
        /* ========== FEATURE CARDS ========== */
        .feature-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }}
        
        .feature-card {{
            background: {card_bg};
            border-radius: 16px;
            padding: 2rem;
            border: 1px solid {border_color};
            position: relative;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .feature-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: {gradient_1};
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }}
        
        .feature-card:hover {{
            transform: translateY(-8px);
            box-shadow: {shadow};
            border-color: {accent_color};
        }}
        
        .feature-card:hover::before {{
            transform: scaleX(1);
        }}
        
        .feature-icon {{
            font-size: 2.5rem;
            margin-bottom: 1rem;
            display: inline-block;
            animation: bounce 2s infinite;
        }}
        
        @keyframes bounce {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-10px); }}
        }}
        
        /* ========== SEARCH BAR ========== */
        .search-container {{
            background: {card_bg};
            border-radius: 16px;
            padding: 1.5rem;
            margin: 2rem 0;
            box-shadow: {shadow};
            border: 2px solid {border_color};
            transition: all 0.3s ease;
        }}
        
        .search-container:focus-within {{
            border-color: {accent_color};
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
        }}
        
        .stTextInput > div > div > input {{
            border-radius: 12px !important;
            border: none !important;
            background: {secondary_bg} !important;
            padding: 1rem 1.5rem !important;
            font-size: 1.1rem !important;
            color: {text_primary} !important;
            transition: all 0.3s ease !important;
        }}
        
        .stTextInput > div > div > input:focus {{
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
            background: {card_bg} !important;
        }}
        
        /* ========== MESSAGE CONTAINERS ========== */
        .message-container {{
            display: flex;
            gap: 1rem;
            margin: 1.5rem 0;
            animation: slideIn 0.4s ease;
        }}
        
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .avatar {{
            width: 48px;
            height: 48px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            flex-shrink: 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        .user-avatar {{
            background: {gradient_2};
        }}
        
        .assistant-avatar {{
            background: {gradient_3};
        }}
        
        .message-content {{
            flex: 1;
            background: {card_bg};
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid {border_color};
            position: relative;
        }}
        
        .user-message {{
            background: {gradient_1};
            color: white;
            border: none;
        }}
        
        .message-actions {{
            display: flex;
            gap: 0.5rem;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid {border_color};
        }}
        
        .action-btn {{
            background: {secondary_bg};
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 0.5rem 1rem;
            cursor: pointer;
            font-size: 0.9rem;
            color: {text_secondary};
            transition: all 0.2s ease;
        }}
        
        .action-btn:hover {{
            background: {accent_color};
            color: white;
            border-color: {accent_color};
            transform: translateY(-2px);
        }}
        
        /* ========== PROPERTY CARDS ========== */
        .property-card {{
            background: {card_bg};
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid {border_color};
            margin: 1rem 0;
            position: relative;
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
        }}
        
        .property-card:hover {{
            transform: translateX(8px);
            box-shadow: {shadow};
            border-color: {accent_color};
        }}
        
        .property-card::before {{
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            width: 4px;
            height: 100%;
            background: {gradient_1};
            transform: scaleY(0);
            transition: transform 0.3s ease;
        }}
        
        .property-card:hover::before {{
            transform: scaleY(1);
        }}
        
        .property-badge {{
            display: inline-block;
            background: {gradient_1};
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-right: 0.5rem;
        }}
        
        /* ========== TYPING INDICATOR ========== */
        .typing-indicator {{
            display: flex;
            gap: 0.3rem;
            padding: 1rem;
            background: {card_bg};
            border-radius: 16px;
            width: fit-content;
        }}
        
        .typing-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: {accent_color};
            animation: typing 1.4s infinite;
        }}
        
        .typing-dot:nth-child(2) {{ animation-delay: 0.2s; }}
        .typing-dot:nth-child(3) {{ animation-delay: 0.4s; }}
        
        @keyframes typing {{
            0%, 60%, 100% {{ transform: translateY(0); opacity: 0.7; }}
            30% {{ transform: translateY(-10px); opacity: 1; }}
        }}
        
        /* ========== TOAST NOTIFICATIONS ========== */
        .toast {{
            position: fixed;
            top: 2rem;
            right: 2rem;
            background: {card_bg};
            border-radius: 12px;
            padding: 1rem 1.5rem;
            box-shadow: {shadow};
            z-index: 9999;
            animation: slideInRight 0.3s ease;
            border-left: 4px solid {accent_color};
        }}
        
        @keyframes slideInRight {{
            from {{ transform: translateX(400px); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}
        
        /* ========== BUTTONS ========== */
        .stButton > button {{
            background: {gradient_1} !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
        }}
        
        /* ========== SIDEBAR ========== */
        .css-1d391kg, [data-testid="stSidebar"] {{
            background: {secondary_bg} !important;
            border-right: 1px solid {border_color} !important;
        }}
        
        .sidebar-section {{
            background: {card_bg};
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid {border_color};
        }}
        
        .sidebar-title {{
            font-size: 1.1rem;
            font-weight: 700;
            color: {text_primary};
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        /* ========== PROGRESS BARS ========== */
        .progress-container {{
            background: {secondary_bg};
            border-radius: 8px;
            height: 8px;
            overflow: hidden;
            margin: 1rem 0;
        }}
        
        .progress-bar {{
            height: 100%;
            background: {gradient_1};
            border-radius: 8px;
            transition: width 0.3s ease;
        }}
        
        /* ========== SKELETON LOADING ========== */
        .skeleton {{
            background: linear-gradient(
                90deg,
                {secondary_bg} 0%,
                {border_color} 50%,
                {secondary_bg} 100%
            );
            background-size: 200% 100%;
            animation: skeleton 1.5s infinite;
            border-radius: 8px;
        }}
        
        @keyframes skeleton {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}
        
        /* ========== TOOLTIPS ========== */
        .tooltip {{
            position: relative;
            display: inline-block;
        }}
        
        .tooltip .tooltiptext {{
            visibility: hidden;
            background: {card_bg};
            color: {text_primary};
            text-align: center;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            border: 1px solid {border_color};
            box-shadow: {shadow};
            white-space: nowrap;
        }}
        
        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}
        
        /* ========== SCROLL TO TOP ========== */
        .scroll-top {{
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: {gradient_1};
            color: white;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: {shadow};
            z-index: 999;
            transition: all 0.3s ease;
        }}
        
        .scroll-top:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
        }}
        
        /* ========== FOOTER ========== */
        .footer {{
            background: {card_bg};
            border-radius: 16px;
            padding: 2rem;
            margin-top: 4rem;
            text-align: center;
            border: 1px solid {border_color};
            color: {text_secondary};
        }}
        
        .footer-links {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 1rem;
            flex-wrap: wrap;
        }}
        
        .footer-link {{
            color: {text_secondary};
            text-decoration: none;
            transition: color 0.2s ease;
        }}
        
        .footer-link:hover {{
            color: {accent_color};
        }}
        
        /* ========== EXPANDER CUSTOMIZATION ========== */
        .streamlit-expanderHeader {{
            background: {card_bg} !important;
            border-radius: 12px !important;
            border: 1px solid {border_color} !important;
            font-weight: 600 !important;
            color: {text_primary} !important;
        }}
        
        .streamlit-expanderHeader:hover {{
            border-color: {accent_color} !important;
        }}
        
        /* ========== COMPARISON VIEW ========== */
        .comparison-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }}
        
        .comparison-card {{
            background: {card_bg};
            border-radius: 16px;
            padding: 2rem;
            border: 2px solid {border_color};
            transition: all 0.3s ease;
        }}
        
        .comparison-card:hover {{
            border-color: {accent_color};
            transform: scale(1.02);
        }}
        
        /* ========== ANIMATIONS ========== */
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        @keyframes slideUp {{
            from {{ transform: translateY(30px); opacity: 0; }}
            to {{ transform: translateY(0); opacity: 1; }}
        }}
        
        .fade-in {{ animation: fadeIn 0.5s ease; }}
        .slide-up {{ animation: slideUp 0.5s ease; }}
        
        /* ========== RESPONSIVE ========== */
        @media (max-width: 768px) {{
            .hero h1 {{ font-size: 2rem; }}
            .hero p {{ font-size: 1rem; }}
            .block-container {{ padding: 1rem !important; }}
            .feature-grid {{ grid-template-columns: 1fr; }}
        }}
        
        /* ========== MISC ========== */
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 0.25rem;
        }}
        
        .badge-success {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
        }}
        
        .badge-info {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        .badge-warning {{
            background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
            color: white;
        }}
        
        /* Hide Streamlit branding */
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        header {{ visibility: hidden; }}
    </style>
    """

# ==============================
# 🔧 UTILITY FUNCTIONS
# ==============================
def add_to_history(role: str, content: str, sources: List[Dict] = None):
    """Add message to chat history"""
    entry = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "sources": sources or []
    }
    st.session_state.history.append(entry)

def add_to_search_history(query: str):
    """Add query to search history"""
    if query not in st.session_state.search_history:
        st.session_state.search_history.insert(0, query)
        st.session_state.search_history = st.session_state.search_history[:10]  # Keep last 10

def toggle_bookmark(item_id: str):
    """Toggle bookmark status"""
    if item_id in st.session_state.bookmarks:
        st.session_state.bookmarks.remove(item_id)
    else:
        st.session_state.bookmarks.add(item_id)

def copy_to_clipboard(text: str):
    """Copy text to clipboard (requires JS injection)"""
    js_code = f"""
    <script>
    navigator.clipboard.writeText(`{text}`);
    </script>
    """
    return js_code

def get_backend_status():
    """Fetch backend status"""
    try:
        resp = requests.get(STATUS_URL, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

# ==============================
# 🎨 INJECT CUSTOM CSS
# ==============================
st.markdown(get_custom_css(), unsafe_allow_html=True)

# ==============================
# 📱 HEADER & NAVIGATION
# ==============================
header_col1, header_col2, header_col3 = st.columns([2, 6, 2])

with header_col1:
    st.markdown("### 🏡 PropertyRAG")

with header_col2:
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    with nav_col1:
        if st.button("🏠 Home", key="nav_home"):
            st.session_state.page = "home"
    with nav_col2:
        if st.button("📊 Analytics", key="nav_analytics"):
            st.session_state.page = "analytics"
    with nav_col3:
        if st.button("⭐ Favorites", key="nav_favorites"):
            st.session_state.page = "favorites"
    with nav_col4:
        if st.button("⚙️ Settings", key="nav_settings"):
            st.session_state.page = "settings"

with header_col3:
    # Dark mode is default - toggle removed
    st.markdown("<div style='height: 42px;'></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================
# 🎯 HERO SECTION
# ==============================
st.markdown("""
<div class="hero fade-in">
    <h1>🏡 AI-Powered Real Estate Intelligence</h1>
    <p>Ask anything about properties, compare areas, analyze trends — powered by advanced RAG technology</p>
    <div style="margin-top: 2rem;">
        <span class="badge badge-success">✓ Smart Retrieval</span>
        <span class="badge badge-info">✓ AI Analysis</span>
        <span class="badge badge-warning">✓ Real-time Data</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================
# 🌟 FEATURE HIGHLIGHTS
# ==============================
st.markdown("""
<div class="feature-grid slide-up">
    <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <h3>Intelligent Search</h3>
        <p>Vector-based semantic search finds the most relevant properties instantly</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon">💡</div>
        <h3>AI Reasoning</h3>
        <p>Gemini-powered insights with context-aware analysis and explanations</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <h3>Advanced Analytics</h3>
        <p>Compare areas, rank by metrics, calculate averages and trends</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🎯</div>
        <h3>Source Transparency</h3>
        <p>Every answer includes exact sources and confidence scores</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================
# 🔍 SEARCH INTERFACE
# ==============================
st.markdown('<div class="search-container">', unsafe_allow_html=True)
st.markdown("### 💬 Ask Your Question")

# Auto-suggestions from search history
if st.session_state.settings["auto_suggestions"] and st.session_state.search_history:
    with st.expander("📜 Recent Searches", expanded=False):
        for idx, past_query in enumerate(st.session_state.search_history[:5]):
            if st.button(f"↩️ {past_query}", key=f"history_{idx}"):
                st.session_state.current_query = past_query

query = st.text_input(
    "Type your question here...",
    placeholder="e.g., Which areas have the most crime? Top 5 | Compare SW1A vs E1 for price",
    key="query_input",
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    ask_button = st.button("✨ Ask Now", use_container_width=True)
with col2:
    if st.button("🔄 Clear History"):
        st.session_state.history = []
        st.rerun()
with col3:
    if st.button("📥 Export"):
        export_data = json.dumps(st.session_state.history, indent=2)
        st.download_button("Download JSON", export_data, "chat_history.json", "application/json")

st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# 🤖 PROCESS QUERY
# ==============================
if ask_button and query.strip():
    add_to_search_history(query)
    add_to_history("user", query)
    
    # Show typing indicator
    with st.container():
        st.markdown("""
        <div class="message-container">
            <div class="avatar assistant-avatar">🤖</div>
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Call backend
    try:
        with st.spinner("🔍 Searching knowledge base..."):
            response = requests.post(
                BACKEND_URL,
                json={"question": query},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer returned.")
                sources = data.get("sources", [])
                
                add_to_history("assistant", answer, sources)
                st.rerun()
            else:
                st.error(f"❌ Backend error: {response.status_code} - {response.text}")
                
    except RequestException as e:
        st.error(f"🚫 Could not reach backend at {BACKEND_URL}. Please ensure FastAPI is running.\n\nError: {e}")

# ==============================
# 💬 CHAT HISTORY DISPLAY
# ==============================
if st.session_state.history:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 💬 Conversation")
    
    for idx, chat in enumerate(reversed(st.session_state.history)):
        if chat["role"] == "user":
            st.markdown(f"""
            <div class="message-container">
                <div class="avatar user-avatar">👤</div>
                <div class="message-content user-message">
                    <strong>You</strong> <span style="opacity:0.7; font-size:0.85rem;">• {chat['timestamp']}</span>
                    <p style="margin-top:0.5rem;">{chat['content']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="message-container">
                <div class="avatar assistant-avatar">🤖</div>
                <div class="message-content">
                    <strong>PropertyRAG AI</strong> <span style="opacity:0.7; font-size:0.85rem;">• {chat['timestamp']}</span>
                    <p style="margin-top:0.5rem; white-space: pre-wrap;">{chat['content']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons
            action_col1, action_col2, action_col3, action_col4 = st.columns([1, 1, 1, 5])
            with action_col1:
                if st.button("📋", key=f"copy_{idx}", help="Copy answer"):
                    st.success("✓ Copied!")
            with action_col2:
                if st.button("👍", key=f"like_{idx}", help="Helpful"):
                    st.success("✓ Thanks for feedback!")
            with action_col3:
                if st.button("👎", key=f"dislike_{idx}", help="Not helpful"):
                    st.info("✓ Feedback recorded")
            
            # Sources
            if chat.get("sources"):
                with st.expander(f"📂 View {len(chat['sources'])} Sources"):
                    for source_idx, source in enumerate(chat["sources"], 1):
                        meta = source.get("metadata", {})
                        st.markdown(f"""
                        <div class="property-card">
                            <div style="display: flex; justify-content: space-between; align-items: start;">
                                <div>
                                    <span class="property-badge">Source {source_idx}</span>
                                    <h4 style="margin: 0.5rem 0;">🏠 {meta.get('address', 'Unknown Address')}</h4>
                                    <p><strong>Type:</strong> {meta.get('type_standardized', 'N/A')} | 
                                       <strong>Price:</strong> £{meta.get('price', 'N/A')} | 
                                       <strong>Beds:</strong> {meta.get('bedrooms', 'N/A')}</p>
                                    <p style="opacity: 0.8; font-size: 0.9rem;">{source.get('document', '')[:200]}...</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

# ==============================
# 📊 SIDEBAR - FILTERS & SETTINGS
# ==============================
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    
    # System Status
    with st.expander("📡 System Status", expanded=True):
        status = get_backend_status()
        if status:
            st.success("✓ Backend Online")
            st.metric("Vectors", status.get("vector_count", "N/A"))
            st.metric("Index", status.get("pinecone_index", "N/A"))
        else:
            st.error("⚠️ Backend Offline")
    
    st.markdown("---")
    
    # Filters
    with st.expander("🔍 Filters", expanded=False):
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        st.session_state.filters["price_min"] = st.number_input(
            "Min Price (£)",
            min_value=0,
            max_value=10000000,
            value=st.session_state.filters["price_min"],
            step=50000
        )
        
        st.session_state.filters["price_max"] = st.number_input(
            "Max Price (£)",
            min_value=0,
            max_value=10000000,
            value=st.session_state.filters["price_max"],
            step=50000
        )
        
        st.session_state.filters["bedrooms"] = st.selectbox(
            "Bedrooms",
            ["Any", "1", "2", "3", "4", "5+"],
            index=0
        )
        
        st.session_state.filters["property_type"] = st.selectbox(
            "Property Type",
            ["All", "Flat", "House", "Studio", "Apartment"],
            index=0
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Advanced Settings
    with st.expander("⚙️ Advanced Settings", expanded=False):
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        st.session_state.settings["response_length"] = st.select_slider(
            "Response Detail",
            options=["concise", "balanced", "detailed"],
            value=st.session_state.settings["response_length"]
        )
        
        st.session_state.settings["show_sources"] = st.checkbox(
            "Show Sources",
            value=st.session_state.settings["show_sources"]
        )
        
        st.session_state.settings["auto_suggestions"] = st.checkbox(
            "Auto-suggestions",
            value=st.session_state.settings["auto_suggestions"]
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Bookmarks
    with st.expander("⭐ Bookmarks", expanded=False):
        if st.session_state.bookmarks:
            for bookmark in list(st.session_state.bookmarks)[:5]:
                st.markdown(f"• {bookmark}")
        else:
            st.info("No bookmarks yet")
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.rerun()
    if st.button("📊 View Analytics", use_container_width=True):
        st.info("Analytics dashboard coming soon!")
    if st.button("💾 Export All Data", use_container_width=True):
        st.info("Export feature coming soon!")

# ==============================
# 🔚 FOOTER
# ==============================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="footer fade-in">
    <h3>🏡 PropertyRAG</h3>
    <p>AI-Powered Real Estate Intelligence Platform</p>
    <div class="footer-links">
        <a href="#" class="footer-link">About</a>
        <a href="#" class="footer-link">Privacy</a>
        <a href="#" class="footer-link">Terms</a>
        <a href="#" class="footer-link">Support</a>
        <a href="#" class="footer-link">API Docs</a>
    </div>
    <p style="margin-top: 1.5rem; opacity: 0.7;">
        Built with ❤️ using Streamlit • FastAPI • Pinecone • Gemini<br>
        © 2025 PropertyRAG. All rights reserved.
    </p>
</div>
""", unsafe_allow_html=True)

# Scroll to top button (placeholder)
st.markdown("""
<div class="scroll-top" onclick="window.scrollTo({top: 0, behavior: 'smooth'})">
    ↑
</div>
""", unsafe_allow_html=True)
