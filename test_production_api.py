"""
🧪 Test Script for Production-Grade PropertyRAG API
"""

import requests
import json
from datetime import datetime

API_BASE = "http://127.0.0.1:8000"

def print_section(title):
    print(f"\n{'='*70}")
    print(f"🧪 {title}")
    print('='*70)

def test_query(question, session_id="test-session", use_expansions=True):
    """Test a single query"""
    print(f"\n📝 Query: {question}")
    
    response = requests.post(
        f"{API_BASE}/ask",
        json={
            "question": question,
            "session_id": session_id,
            "use_caching": True,
            "use_expansions": use_expansions
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Status: {response.status_code}")
        print(f"🎯 Intent: {data.get('intent', 'N/A')}")
        
        # Confidence
        confidence = data.get('confidence', {})
        if isinstance(confidence, dict):
            print(f"📊 Confidence: {confidence.get('label', 'N/A')} ({confidence.get('score', 0):.2f})")
        
        # Answer
        answer = data.get('answer', '')
        print(f"\n💬 Answer:\n{answer[:500]}..." if len(answer) > 500 else f"\n💬 Answer:\n{answer}")
        
        # Follow-ups
        follow_ups = data.get('follow_ups', [])
        if follow_ups:
            print(f"\n💡 Follow-up Suggestions:")
            for i, fu in enumerate(follow_ups[:3], 1):
                print(f"   {i}. {fu}")
        
        # Sources
        sources = data.get('sources', [])
        if sources:
            print(f"\n📚 Sources: {len(sources)} documents retrieved")
            if len(sources) > 0:
                print(f"   Top score: {sources[0].get('score', 0):.3f}")
        
        # Clarification
        if data.get('needs_clarification'):
            clarifying = data.get('clarifying_questions', [])
            print(f"\n❓ Needs Clarification:")
            for i, q in enumerate(clarifying, 1):
                print(f"   {i}. {q}")
        
        return data
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(f"   {response.text}")
        return None

def test_metrics():
    """Test metrics endpoint"""
    print_section("SYSTEM METRICS")
    
    response = requests.get(f"{API_BASE}/metrics")
    if response.status_code == 200:
        metrics = response.json()
        print(f"\n📊 System Metrics:")
        print(f"   Total Queries: {metrics.get('total_queries', 0)}")
        print(f"   Cache Hit Rate: {metrics.get('cache_hit_rate', 'N/A')}")
        print(f"   Avg Latency: {metrics.get('avg_retrieval_latency', 'N/A')}")
        print(f"   Successful Generations: {metrics.get('successful_generations', 0)}")
        print(f"   Failed Generations: {metrics.get('failed_generations', 0)}")
        print(f"   Deterministic Responses: {metrics.get('deterministic_responses', 0)}")
        
        intent_breakdown = metrics.get('intent_breakdown', {})
        if intent_breakdown:
            print(f"\n🎯 Intent Breakdown:")
            for intent, count in intent_breakdown.items():
                print(f"   {intent}: {count}")
    else:
        print(f"❌ Error: {response.status_code}")

def main():
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║   🚀 PropertyRAG Production-Grade API Test Suite                 ║
║   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                      ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Test 1: Simple property search (with abbreviations)
    print_section("TEST 1: Simple Property Search (with abbreviations)")
    test_query("Show me 2 BHK flats in Mumbai under 50L")
    
    # Test 2: Comparison query
    print_section("TEST 2: Area Comparison")
    test_query("Compare crime rates in SW1A vs E1")
    
    # Test 3: Ambiguous query (should request clarification)
    print_section("TEST 3: Ambiguous Query Detection")
    test_query("I want to buy a property")
    
    # Test 4: Out-of-scope query
    print_section("TEST 4: Out-of-Scope Detection")
    test_query("What's the weather like today?")
    
    # Test 5: Market trends
    print_section("TEST 5: Market Trends Query")
    test_query("What are the property price trends in London?")
    
    # Test 6: Multi-turn conversation
    print_section("TEST 6: Multi-Turn Conversation")
    session_id = "conversation-test"
    test_query("Tell me about properties in Westminster", session_id=session_id)
    print("\n" + "─"*70)
    test_query("What about the prices there?", session_id=session_id)
    
    # Test 7: Caching (same query twice)
    print_section("TEST 7: Query Caching")
    print("First query (cache miss):")
    test_query("Show me 3 bedroom houses in Camden")
    print("\n" + "─"*70)
    print("Second query (cache hit expected):")
    test_query("Show me 3 bedroom houses in Camden")
    
    # Test 8: System metrics
    test_metrics()
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║   ✅ Test Suite Complete                                         ║
║   Check the logs at: project/rag_system.log                      ║
║   Check query metrics at: project/query_metrics.jsonl           ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    main()
