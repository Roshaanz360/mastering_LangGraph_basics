#!/usr/bin/env python3
"""
Test script to verify startup document loading works correctly
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_startup_loading():
    print("=== Testing Startup Document Loading ===")
    
    # Check API keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    print(f"GEMINI_API_KEY present: {bool(gemini_key)}")
    print(f"GROQ_API_KEY present: {bool(groq_key)}")
    
    if not gemini_key or not groq_key:
        print("ERROR: Both GEMINI_API_KEY and GROQ_API_KEY are required!")
        return
    
    # Import the graph module (this will trigger startup document loading)
    print("\n--- Importing graph module (triggers startup loading) ---")
    from src.react_agent.graph import rag_processor
    
    # Check if all documents are loaded
    print("\n--- Checking Document Loading Results ---")
    if not rag_processor.vectorstore:
        print("❌ ERROR: No vectorstore created!")
        return
    
    try:
        # Check document count
        collection = rag_processor.vectorstore._collection
        count = collection.count()
        print(f"✅ Vectorstore created with {count} documents")
        
        # Check document sources
        sample_docs = rag_processor.vectorstore.similarity_search("the", k=20)
        sources = set()
        for doc in sample_docs:
            source = doc.metadata.get('source', '')
            if source:
                filename = source.split('\\')[-1].split('/')[-1]
                sources.add(filename)
        
        expected_files = {'diff.txt', 'hallu.txt', 'reHack.txt', 'sample_data.txt'}
        print(f"📚 Document sources found: {sources}")
        
        if expected_files.issubset(sources):
            print("✅ SUCCESS: All expected documents are loaded!")
            
            # Test a quick query
            print("\n--- Testing Quick Query ---")
            docs = rag_processor.retrieve("reward hacking", k=2)
            if docs:
                print(f"✅ Query test passed: Retrieved {len(docs)} documents")
                for i, doc in enumerate(docs):
                    source = doc.get('metadata', {}).get('source', 'unknown')
                    filename = source.split('\\')[-1].split('/')[-1]
                    print(f"  Doc {i+1}: {filename}")
            else:
                print("❌ Query test failed: No documents retrieved")
        else:
            missing = expected_files - sources
            print(f"❌ ERROR: Missing documents: {missing}")
            
    except Exception as e:
        print(f"❌ ERROR checking vectorstore: {e}")

if __name__ == "__main__":
    test_startup_loading() 