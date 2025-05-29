#!/usr/bin/env python3
"""
Simple test script to verify RAG functionality
"""

import os
from dotenv import load_dotenv
from src.react_agent.rag import RAGProcessor

# Load environment variables
load_dotenv()

async def test_rag():
    print("=== Testing RAG System ===")
    
    # Check API keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    print(f"GEMINI_API_KEY present: {bool(gemini_key)}")
    print(f"GROQ_API_KEY present: {bool(groq_key)}")
    
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY is required for embeddings!")
        return
    
    if not groq_key:
        print("ERROR: GROQ_API_KEY is required for text generation!")
        return
    
    # Check if documents directory exists
    docs_dir = "documents/"
    if not os.path.exists(docs_dir):
        print(f"ERROR: Documents directory '{docs_dir}' does not exist!")
        return
    
    # Initialize RAG processor
    rag = RAGProcessor()
    
    # Check if we need to reload documents
    print("\n--- Checking RAG System ---")
    if not rag.vectorstore:
        print("No existing vectorstore found. Loading documents...")
        rag.force_reload_documents("documents/")
    else:
        # Quick check if all documents are loaded
        try:
            collection = rag.vectorstore._collection
            count = collection.count()
            print(f"Found existing vectorstore with {count} documents")
            
            # Check if we have documents from all sources
            sample_docs = rag.vectorstore.similarity_search("the", k=10)
            sources = set()
            for doc in sample_docs:
                source = doc.metadata.get('source', 'unknown')
                sources.add(source.split('\\')[-1])  # Just filename
            
            print(f"Document sources: {', '.join(sources)}")
            
            # If we don't have all 4 files, reload
            expected_files = {'diff.txt', 'hallu.txt', 'reHack.txt', 'sample_data.txt'}
            if not expected_files.issubset(sources):
                print("Missing some documents. Reloading...")
                rag.force_reload_documents("documents/")
            else:
                print("All documents present. Ready for queries!")
                
        except Exception as e:
            print(f"Error checking vectorstore: {e}")
            print("Reloading documents...")
            rag.force_reload_documents("documents/")
    
    # Test queries
    test_queries = [
        "What is reward hacking?",
        "What causes hallucinations in LLMs?", 
        "What is machine learning?",
        "What is diffusion model?"
    ]
    
    print("\n--- Testing Queries ---")
    for query in test_queries:
        print(f"\n" + "="*60)
        print(f"❓ QUERY: {query}")
        print("="*60)
        
        # Test retrieval
        docs = rag.retrieve(query, k=3)
        
        if docs:
            # Test generation
            try:
                response = await rag.generate_response(query, docs)
                print(f"\n💡 ANSWER:")
                print("-" * 40)
                print(response)
                print("-" * 40)
                
                # Show source information
                sources = set()
                for doc in docs:
                    source = doc.get('metadata', {}).get('source', 'unknown')
                    sources.add(source.split('\\')[-1])  # Just filename
                
                print(f"\n📚 Sources: {', '.join(sources)}")
                
            except Exception as e:
                print(f"\n❌ Generation error: {e}")
        else:
            print(f"\n❌ No documents retrieved!")
        
        print("="*60)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_rag()) 