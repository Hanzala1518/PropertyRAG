"""
🏆 ADVANCED RAG SYSTEM - Production-Grade Enhancements
Sophisticated prompt engineering, response formatting, and error handling
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ==============================
# 🎯 SOPHISTICATED PROMPT ENGINEERING
# ==============================

ROLE_BASED_SYSTEM_PROMPT = """You are an expert real estate consultant with 15+ years of experience in the UK property market. You possess deep knowledge of:
- Property valuations, market trends, and investment analysis
- UK neighborhoods, schools, transport links, and local amenities
- Mortgage products, stamp duty, and legal requirements
- Construction quality, property types, and architectural features

Your expertise allows you to provide authoritative, data-driven advice while maintaining professionalism and empathy. You always cite your sources and acknowledge limitations when data is unavailable.

CORE PRINCIPLES:
1. **Accuracy First**: Only use information from provided context. Never fabricate data.
2. **Source Attribution**: Always cite sources with [Source: property_address or document_reference]
3. **Clarity & Structure**: Organize responses with clear sections, bullet points, and tables where appropriate
4. **Professional Tone**: Maintain a helpful, consultative tone while being direct and informative
5. **Risk Awareness**: Include appropriate disclaimers for financial/legal advice

When uncertain, say: "Based on the available data, I cannot provide a definitive answer. Here's what I can tell you..."
"""

INTENT_PROMPTS = {
    "property_search": """
Provide a comprehensive property search response:

**Response Structure**:
1. **Summary**: Brief overview of matching properties found
2. **Property Listings**: Present each property with:
   - Full address and postcode
   - Property type, bedrooms, bathrooms
   - Price (with price per sq ft if available)
   - Key features and amenities
   - Proximity to transport/schools
   - Condition and year built
3. **Market Context**: How these compare to area averages
4. **Recommendation**: Which properties best match the criteria and why
5. **Sources**: Full citations for each property

Use tables for comparing multiple properties. Include confidence scores if multiple interpretations exist.
""",

    "comparison": """
Provide a detailed side-by-side comparison:

**Response Structure**:
1. **Comparison Summary**: Quick verdict on which option is better for different priorities
2. **Detailed Comparison Table**: 
   | Aspect | Option A | Option B | Winner |
   |--------|----------|----------|---------|
   | Price | ... | ... | ... |
   | Location | ... | ... | ... |
   | Features | ... | ... | ... |
3. **Pros & Cons**: Bullet list for each option
4. **Data-Backed Analysis**: Reference specific metrics and sources
5. **Recommendation**: Personalized advice based on typical buyer priorities
6. **Sources**: Full citations

Be objective and data-driven. Highlight tradeoffs clearly.
""",

    "market_trends": """
Provide comprehensive market analysis:

**Response Structure**:
1. **Executive Summary**: Key findings in 2-3 sentences
2. **Trend Analysis**:
   - Average/median prices (with YoY change)
   - Price distribution and range
   - Popular property types
   - Days on market / inventory levels
3. **Neighborhood Insights**: Local factors affecting prices
4. **Comparative Context**: How this area compares to nearby areas
5. **Forward-Looking**: Any emerging trends or factors to watch
6. **Sources & Confidence**: Data provenance and sample sizes

Use percentages, charts descriptions, and specific numbers. Clearly indicate if data is limited.
""",

    "investment_advice": """
Provide structured investment analysis:

**Response Structure**:
1. **Investment Snapshot**: Quick assessment (Strong/Moderate/Weak opportunity)
2. **Financial Analysis**:
   - Estimated rental yield (if applicable)
   - Historical appreciation rate
   - Comparable sales data
   - ROI projections (with assumptions clearly stated)
3. **Risk Assessment**:
   - Market volatility in the area
   - Development plans/infrastructure changes
   - Economic factors
4. **Exit Strategy**: Resale potential and typical holding periods
5. **Recommendation**: Buy/Hold/Pass with rationale
6. **Disclaimer**: Risks and limitations of analysis
7. **Sources**: Full citations for all claims

⚠️ IMPORTANT: Always include: "This analysis is for informational purposes only and does not constitute financial advice. Consult with a qualified financial advisor before making investment decisions."
""",

    "legal_query": """
Provide accurate legal/regulatory information:

**Response Structure**:
1. **Direct Answer**: Clear response to the legal question
2. **Regulatory Context**: Relevant UK laws, regulations, or requirements
3. **Documentation Needed**: If applicable, list required documents
4. **Process Overview**: Step-by-step if asking about procedures
5. **Cost Implications**: Fees, taxes, or charges involved
6. **Expert Referral**: When to consult solicitors/conveyancers
7. **Sources**: Reference to official guidance or regulations

⚠️ CRITICAL DISCLAIMER: "This information is provided for general guidance only and does not constitute legal advice. Property law can be complex and fact-specific. Always consult with a qualified solicitor or licensed conveyancer for your specific situation."
""",

    "amenities_question": """
Provide detailed amenities analysis:

**Response Structure**:
1. **Amenity Overview**: Summary of available facilities
2. **Detailed Breakdown**:
   - On-site amenities (gym, pool, parking, etc.)
   - Nearby facilities (schools, hospitals, shopping)
   - Transport links (tube/train stations, bus routes)
   - Green spaces and recreation
3. **Quality Assessment**: Condition and accessibility of amenities
4. **Comparison**: How this compares to typical offerings in the area
5. **Value Impact**: How amenities affect property value
6. **Sources**: Citations for each amenity claim

Include distances/walking times where available.
""",

    "area_ranking": """
Provide comprehensive area ranking:

**Response Structure**:
1. **Ranking Summary**: Top N areas with key metric
2. **Detailed Rankings Table**:
   | Rank | Area | Metric Value | Sample Size | Notes |
   |------|------|--------------|-------------|-------|
3. **Analysis**: Why these areas rank this way
4. **Context & Caveats**: Limitations of the ranking methodology
5. **Actionable Insights**: What this means for buyers/renters
6. **Sources**: Data provenance and calculation methodology

Be transparent about sample sizes and confidence levels.
""",
}

CHAIN_OF_THOUGHT_TEMPLATE = """
Let's approach this systematically:

**Step 1: Understanding the Question**
{question_analysis}

**Step 2: Analyzing Available Data**
{data_summary}

**Step 3: Key Findings**
{key_findings}

**Step 4: Reasoning & Conclusion**
{reasoning}

**Final Answer**:
{final_answer}
"""

# ==============================
# 📊 PROFESSIONAL RESPONSE FORMATTING
# ==============================

def format_professional_response(
    query: str,
    intent: str,
    answer: str,
    sources: List[Dict[str, Any]],
    confidence: float,
    entities: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Structure response with professional formatting
    """
    # Generate confidence indicator
    if confidence >= 0.9:
        confidence_label = "Very High"
        confidence_emoji = "✅"
    elif confidence >= 0.7:
        confidence_label = "High"
        confidence_emoji = "✓"
    elif confidence >= 0.5:
        confidence_label = "Moderate"
        confidence_emoji = "⚠️"
    else:
        confidence_label = "Low"
        confidence_emoji = "❗"
    
    # Generate follow-up suggestions
    follow_ups = generate_follow_ups(query, intent, entities)
    
    # Structure response
    structured_response = {
        "answer": answer,
        "confidence": {
            "score": confidence,
            "label": confidence_label,
            "emoji": confidence_emoji
        },
        "intent": intent,
        "follow_up_suggestions": follow_ups,
        "sources": format_sources(sources),
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "entities_extracted": entities,
            "response_type": intent
        }
    }
    
    return structured_response

def format_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enhanced source formatting with excerpts, scores, and credibility
    """
    formatted = []
    for idx, source in enumerate(sources, 1):
        meta = source.get("metadata", {})
        doc = source.get("document", "")
        score = source.get("score", 0.0)
        
        # Extract key excerpt (first 150 chars with context)
        excerpt = doc[:150] + "..." if len(doc) > 150 else doc
        
        # Determine document type
        doc_type = "Property Listing"  # Default; extend with actual classification
        
        # Credibility score (simplified; extend with actual scoring)
        credibility = "Official" if meta.get("is_verified") else "Standard"
        
        formatted.append({
            "source_number": idx,
            "title": f"Property at {meta.get('address', 'Unknown')}",
            "excerpt": excerpt,
            "full_document": doc,
            "relevance_score": round(score, 3),
            "document_type": doc_type,
            "credibility": credibility,
            "metadata": {
                "address": meta.get("address", "N/A"),
                "property_type": meta.get("type_standardized", "N/A"),
                "price": meta.get("price", "N/A"),
                "bedrooms": meta.get("bedrooms", "N/A"),
                "bathrooms": meta.get("bathrooms", "N/A"),
                "crime_score": meta.get("crime_score_weight", "N/A"),
                "flood_risk": meta.get("flood_risk", "N/A"),
            },
            "citation": f"[{idx}] {meta.get('address', 'Property Listing')}"
        })
    
    return formatted

def generate_follow_ups(query: str, intent: str, entities: Dict[str, Any]) -> List[str]:
    """
    Generate contextual follow-up questions
    """
    follow_ups = []
    
    if intent == "property_search":
        follow_ups = [
            "Would you like to see properties in nearby areas?",
            "Should I filter by specific amenities (parking, garden, etc.)?",
            "Are you interested in new builds or resale properties?",
            "Would you like a comparison of the top options?"
        ]
    elif intent == "comparison":
        follow_ups = [
            "Would you like detailed mortgage calculations for each?",
            "Should I analyze the investment potential of these properties?",
            "Would you like information about the neighborhoods?",
            "Shall I find similar alternatives in the same area?"
        ]
    elif intent == "market_trends":
        follow_ups = [
            "Would you like to see trends for specific property types?",
            "Should I compare this area to nearby neighborhoods?",
            "Would you like historical price data over a longer period?",
            "Shall I analyze rental yields in this area?"
        ]
    elif intent == "investment_advice":
        follow_ups = [
            "Would you like to see comparable investment opportunities?",
            "Should I calculate potential rental income?",
            "Would you like information about property management services?",
            "Shall I analyze the exit strategy options?"
        ]
    
    # Add entity-based follow-ups
    if entities.get("bedrooms"):
        follow_ups.append(f"Would you also consider {entities['bedrooms']+1} bedroom properties?")
    
    if entities.get("locations"):
        follow_ups.append(f"Should I expand the search to areas near {entities['locations'][0]}?")
    
    return follow_ups[:4]  # Limit to 4 suggestions

def build_sophisticated_prompt(
    context_items: List[Dict[str, Any]],
    user_question: str,
    intent: str,
    entities: Dict[str, Any],
    conversation_history: List[Dict[str, str]] = None
) -> str:
    """
    Build a sophisticated, intent-specific prompt with role-based instructions
    """
    # Select intent-specific template
    intent_instruction = INTENT_PROMPTS.get(intent, INTENT_PROMPTS["property_search"])
    
    # Build context block with enhanced formatting
    context_sections = []
    max_chars = 1200
    for i, item in enumerate(context_items, start=1):
        metadata = item.get("metadata", {})
        document = item.get("document", "")
        score = item.get("score", 0.0)
        
        if isinstance(document, str) and len(document) > max_chars:
            document = document[:max_chars] + " [...]"
        
        # Format metadata clearly
        meta_lines = [
            f"Property Type: {metadata.get('type_standardized', 'N/A')}",
            f"Address: {metadata.get('address', 'N/A')}",
            f"Price: £{metadata.get('price', 'N/A')}",
            f"Bedrooms: {metadata.get('bedrooms', 'N/A')}",
            f"Bathrooms: {metadata.get('bathrooms', 'N/A')}",
            f"Crime Score: {metadata.get('crime_score_weight', 'N/A')}",
            f"Flood Risk: {metadata.get('flood_risk', 'N/A')}",
            f"Relevance Score: {score:.3f}"
        ]
        
        context_sections.append(
            f"[Source {i}]\n" +
            f"Description: {document}\n" +
            f"Property Details:\n" +
            "\n".join(f"  - {line}" for line in meta_lines)
        )
    
    context_block = "\n\n".join(context_sections) if context_sections else "(No relevant sources found)"
    
    # Add conversation context if available
    history_block = ""
    if conversation_history and len(conversation_history) > 0:
        history_lines = []
        for turn in conversation_history[-4:]:  # Last 4 turns
            role = "User" if turn["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {turn['content'][:200]}")
        history_block = "\n--- CONVERSATION HISTORY ---\n" + "\n".join(history_lines) + "\n\n"
    
    # Construct full prompt
    prompt = f"""{ROLE_BASED_SYSTEM_PROMPT}

{intent_instruction}

{history_block}--- RETRIEVED CONTEXT (Use ONLY this information) ---
{context_block}

--- USER QUESTION ---
{user_question}

--- EXTRACTED ENTITIES ---
{json.dumps(entities, indent=2)}

--- YOUR RESPONSE ---
Provide a comprehensive, well-structured answer following the response structure above. Remember to:
✓ Cite sources explicitly with [Source N]
✓ Use tables/bullets for clarity
✓ Include disclaimers where appropriate
✓ Acknowledge limitations if data is incomplete
✓ Maintain professional, consultative tone

Begin your response:
"""
    
    return prompt

# ==============================
# ⚠️ ERROR HANDLING & CLARIFICATION
# ==============================

def generate_clarifying_questions(query: str, entities: Dict[str, Any], intent: str) -> Optional[List[str]]:
    """
    Generate clarifying questions when query is ambiguous
    """
    questions = []
    
    # Missing location
    if intent in ["property_search", "market_trends"] and not entities.get("locations"):
        questions.append("📍 Which area or location are you interested in? (e.g., Chelsea, SW1A, South London)")
    
    # Missing budget
    if intent == "property_search" and not entities.get("price_range"):
        questions.append("💰 What's your budget range? (e.g., £300k-£400k, under £250k)")
    
    # Vague property type
    if intent == "property_search" and not entities.get("property_types") and not entities.get("bedrooms"):
        questions.append("🏠 What type of property are you looking for? (e.g., 2-bed flat, 3-bed house, studio)")
    
    # Ambiguous comparison
    if intent == "comparison" and len(entities.get("locations", [])) < 2:
        questions.append("🔄 Which two areas or properties would you like me to compare?")
    
    # Timeframe for trends
    if intent == "market_trends":
        questions.append("📅 What time period are you interested in? (e.g., last 6 months, yearly trends)")
    
    return questions if questions else None

def handle_out_of_scope(query: str) -> Dict[str, Any]:
    """
    Handle queries outside the system's scope
    """
    out_of_scope_keywords = [
        "weather", "restaurant", "movie", "recipe", "sports", "politics",
        "health diagnosis", "medical advice", "stock price", "cryptocurrency"
    ]
    
    if any(kw in query.lower() for kw in out_of_scope_keywords):
        return {
            "answer": "I apologize, but I'm specialized in UK property and real estate matters. "
                     "I can help you with property searches, market analysis, area comparisons, "
                     "investment advice, and legal/regulatory questions about real estate. "
                     "\n\nHow can I assist you with property-related questions?",
            "is_out_of_scope": True,
            "suggested_topics": [
                "Search for properties in specific areas",
                "Compare neighborhoods or properties",
                "Analyze market trends and prices",
                "Get investment advice for properties",
                "Understand property legal requirements"
            ]
        }
    
    return None

def format_error_response(error_type: str, details: str = "") -> Dict[str, Any]:
    """
    Format user-friendly error messages
    """
    error_messages = {
        "no_results": {
            "title": "No Matching Properties Found",
            "message": "I couldn't find any properties matching your criteria. This could be because:\n"
                      "• The area might not have available listings\n"
                      "• The price range or requirements might be too specific\n"
                      "• There might be a typo in the location name",
            "suggestions": [
                "Try broadening your search criteria",
                "Check the spelling of location names",
                "Expand your price range",
                "Consider nearby areas"
            ]
        },
        "retrieval_failed": {
            "title": "Search Temporarily Unavailable",
            "message": "I'm having trouble accessing the property database right now. Please try again in a moment.",
            "suggestions": [
                "Refresh the page and try again",
                "Simplify your query",
                "Contact support if the issue persists"
            ]
        },
        "ambiguous_query": {
            "title": "Need More Information",
            "message": f"Your query needs a bit more detail for me to provide accurate results. {details}",
            "suggestions": []
        }
    }
    
    error_info = error_messages.get(error_type, {
        "title": "Unexpected Error",
        "message": "Something went wrong. Please try rephrasing your question.",
        "suggestions": []
    })
    
    return {
        "error": True,
        "error_type": error_type,
        **error_info
    }

# ==============================
# 📈 RESPONSE QUALITY METRICS
# ==============================

def calculate_response_confidence(
    query: str,
    retrieved_items: List[Dict[str, Any]],
    intent: str
) -> float:
    """
    Calculate confidence score for the response
    """
    if not retrieved_items:
        return 0.0
    
    # Factors:
    # 1. Top score of retrieved items
    top_score = max([item.get("score", 0.0) for item in retrieved_items])
    
    # 2. Number of relevant items (more is better, up to a point)
    num_items = len(retrieved_items)
    coverage_score = min(num_items / 5.0, 1.0)  # Optimal is 5+ items
    
    # 3. Score distribution (tight clustering = higher confidence)
    scores = [item.get("score", 0.0) for item in retrieved_items[:5]]
    if len(scores) > 1:
        score_std = sum((s - sum(scores)/len(scores))**2 for s in scores) ** 0.5
        consistency_score = max(0, 1.0 - score_std)
    else:
        consistency_score = 0.5
    
    # 4. Intent-specific adjustments
    intent_weight = {
        "property_search": 1.0,
        "comparison": 0.9,  # Slightly lower as requires exact matches
        "market_trends": 1.1,  # Can work with broader data
        "specific_property": 1.2,  # High confidence when found
    }.get(intent, 1.0)
    
    # Weighted combination
    confidence = (
        0.5 * top_score +
        0.2 * coverage_score +
        0.3 * consistency_score
    ) * intent_weight
    
    return min(confidence, 1.0)

import json
