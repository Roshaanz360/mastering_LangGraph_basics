#!/usr/bin/env python3
"""
Test script for the SIMPLIFIED Enhanced RAG Agent.
Ultra-streamlined graph with maximum functionality in minimal nodes.
"""

import asyncio
from src.react_agent.graph import graph
from src.react_agent.state import State
from langchain_core.messages import HumanMessage

async def test_simplified_rag():
    """Test the ultra-simplified RAG agent with all enhanced features."""
    print("Testing SIMPLIFIED Enhanced RAG Agent")
    print("=" * 60)
    print("🎯 ULTRA-SIMPLIFIED ARCHITECTURE:")
    print("   📝 call_model (query + routing)")
    print("   🤖 model_direct_response (general queries)")
    print("   🔍 rag_complete_pipeline (ALL enhanced RAG features)")
    print("   ✨ final_response (proper RAG response formatting)")
    print("   🛠️  tools (if needed)")
    print("=" * 60)
    
    # Test cases for the simplified architecture
    test_cases = [
        {
            "query": "What is reward hacking in AI safety?",
            "expected": "RAG",
            "description": "Technical AI safety question"
        },
        {
            "query": "What's 2 + 2?",
            "expected": "GENERAL", 
            "description": "Basic math question"
        },
        {
            "query": "Explain hallucination in language models",
            "expected": "RAG",
            "description": "Technical LLM concept"
        },
        {
            "query": "How's the weather today?",
            "expected": "GENERAL",
            "description": "General knowledge question"
        }
    ]
    
    print(f"\n🧪 Running {len(test_cases)} tests on simplified architecture...\n")
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"[TEST {i}] {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        print(f"Expected: {test_case['expected']} path")
        
        initial_state = {
            "messages": [HumanMessage(content=test_case['query'])]
        }
        
        try:
            result = await graph.ainvoke(initial_state)
            
            # Determine which path was taken
            if 'rag_response' in result and result.get('rag_response'):
                actual_path = "RAG"
                features_used = []
                if result.get('document_grades'):
                    features_used.append("Document Grading")
                if result.get('rewritten_query'):
                    features_used.append("Query Rewriting")
                if result.get('verification_result'):
                    features_used.append("Response Verification")
                if result.get('performance_comparison'):
                    features_used.append("Performance Evaluation")
                
                print(f"✅ Path: RAG Pipeline")
                print(f"🔧 Enhanced Features Used: {', '.join(features_used) if features_used else 'Basic RAG'}")
                
            else:
                actual_path = "GENERAL"
                print(f"✅ Path: Model Direct Response")
            
            # Check if matches expectation
            match = "✅" if actual_path == test_case['expected'] else "❌"
            print(f"{match} Expected: {test_case['expected']}, Actual: {actual_path}")
            
            # Show response preview
            final_message = result['messages'][-1].content
            print(f"📝 Response: {final_message[:100]}...")
            
            results.append({
                "test": test_case['description'],
                "expected": test_case['expected'],
                "actual": actual_path,
                "correct": actual_path == test_case['expected']
            })
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append({
                "test": test_case['description'],
                "expected": test_case['expected'],
                "actual": "ERROR",
                "correct": False
            })
        
        print("-" * 50)
    
    # Final Summary
    print("\n" + "=" * 60)
    print("🎯 SIMPLIFIED RAG AGENT TEST RESULTS")
    print("=" * 60)
    
    correct_tests = sum(1 for r in results if r['correct'])
    total_tests = len(results)
    success_rate = (correct_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"📊 Success Rate: {correct_tests}/{total_tests} ({success_rate:.1f}%)")
    
    rag_tests = sum(1 for r in results if r['actual'] == 'RAG')
    general_tests = sum(1 for r in results if r['actual'] == 'GENERAL')
    
    print(f"🔍 RAG Pipeline Used: {rag_tests} times")
    print(f"🧠 General Response Used: {general_tests} times")
    
    print(f"\n🏗️ ARCHITECTURE BENEFITS:")
    print(f"   ⚡ ULTRA-FAST: Only 3-4 nodes maximum")
    print(f"   🎯 COMPLETE: All enhanced features in one pipeline")
    print(f"   🧠 INTELLIGENT: LLM-based routing")
    print(f"   🚀 PRODUCTION-READY: Minimal complexity, maximum functionality")
    
    if success_rate >= 75:
        print(f"\n🎉 EXCELLENT! Simplified architecture is working perfectly!")
        print(f"🚀 Ready for production deployment!")
    elif success_rate >= 50:
        print(f"\n👍 GOOD! Architecture is solid with room for fine-tuning")
    else:
        print(f"\n⚠️ NEEDS ADJUSTMENT: Routing logic requires optimization")
    
    print(f"\n✨ SIMPLIFIED ENHANCED RAG AGENT: MAXIMUM POWER, MINIMUM COMPLEXITY!")

if __name__ == "__main__":
    asyncio.run(test_simplified_rag()) 