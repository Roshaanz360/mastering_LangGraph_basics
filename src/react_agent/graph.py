from datetime import datetime, timezone
from typing import Dict, List, Literal, cast

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model, log_step
from react_agent.rag import RAGProcessor
from react_agent.evaluator import RAGEvaluator

# Load environment variables
load_dotenv()

# Initialize RAG components
rag_processor = RAGProcessor()

# Ensure all documents are loaded at startup
def ensure_all_documents_loaded():
    """Ensure all expected documents are loaded in the vectorstore."""
    expected_files = {'diff.txt', 'hallu.txt', 'reHack.txt', 'sample_data.txt'}
    
    if not rag_processor.vectorstore:
        print("[STARTUP] No vectorstore found. Loading all documents...")
        rag_processor.force_reload_documents("documents/")
        return
    
    try:
        # Check what documents are currently in the vectorstore
        sample_docs = rag_processor.vectorstore.similarity_search("the", k=20)
        current_sources = set()
        for doc in sample_docs:
            source = doc.metadata.get('source', '')
            if source:
                filename = source.split('\\')[-1].split('/')[-1]  # Handle both Windows and Unix paths
                current_sources.add(filename)
        
        print(f"[STARTUP] Current document sources: {current_sources}")
        
        # If we don't have all expected files, reload
        if not expected_files.issubset(current_sources):
            missing = expected_files - current_sources
            print(f"[STARTUP] Missing documents: {missing}. Reloading all documents...")
            rag_processor.force_reload_documents("documents/")
        else:
            print("[STARTUP] All expected documents are present in vectorstore.")
            
    except Exception as e:
        print(f"[STARTUP] Error checking vectorstore contents: {e}")
        print("[STARTUP] Reloading documents to be safe...")
        rag_processor.force_reload_documents("documents/")

# Load all documents at startup
ensure_all_documents_loaded()

rag_evaluator = RAGEvaluator()

# === Enhanced LangGraph Nodes ===

# Step 1: Model call
async def call_model(state: State) -> dict:
    log_step(state, "call_model")

    # Extract the current query from the last human message
    last_human_msgs = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    current_query = last_human_msgs[-1].content if last_human_msgs else ""

    print(f"[GRAPH] Extracted query: '{current_query}' - determining relevance...")

    if not current_query:
        return {
            "current_query": current_query,
            "original_query": current_query,
            "is_rag_relevant": False,
            "relevance_reason": "No query provided"
        }

    # Use LLM to assess if the query requires specific/domain knowledge
    configuration = Configuration.from_context()
    
    try:
        model = load_chat_model(configuration.model)
        
        assessment_prompt = f"""You are an AI assistant that determines whether a user query can be answered with general knowledge or requires specific domain documents.

User Query: "{current_query}"

Available Documents Context: The system has access to documents about AI safety, machine learning, reward hacking, hallucination in AI, and related technical topics.

Analyze the query and determine:
1. Can this query be answered well with general AI knowledge?
2. Does this query require specific technical details, research findings, or domain-specific information that would be better served by retrieving relevant documents?

Consider these factors:
- General questions about common topics → Use general knowledge
- Specific technical questions about AI safety, ML research, specific algorithms → Use RAG
- Questions asking for specific examples, case studies, or detailed explanations → Use RAG
- Questions about weather, general facts, basic concepts → Use general knowledge
- Questions about specific papers, studies, or technical implementations → Use RAG

Respond with ONLY one of these options:
- "GENERAL" if the query can be answered well with general knowledge
- "RAG" if the query requires specific domain documents for a better answer

Response:"""

        response = await model.ainvoke(assessment_prompt)
        decision = response.content.strip().upper()
        
        if "RAG" in decision:
            print(f"[GRAPH] LLM determined query requires RAG: '{current_query}'")
            return {
                "current_query": current_query,
                "original_query": current_query,
                "is_rag_relevant": True,
                "relevance_reason": "LLM determined this query requires specific domain knowledge from documents"
            }
        else:
            print(f"[GRAPH] LLM determined query can be answered with general knowledge: '{current_query}'")
            return {
                "current_query": current_query,
                "original_query": current_query,
                "is_rag_relevant": False,
                "relevance_reason": "LLM determined this query can be answered with general knowledge"
            }
            
    except Exception as e:
        print(f"[GRAPH] Error in LLM assessment: {e} - falling back to document check")
        
        # Fallback to simple document check if LLM assessment fails
        if not rag_processor.vectorstore:
            return {
                "current_query": current_query,
                "original_query": current_query,
                "is_rag_relevant": False,
                "relevance_reason": "No documents available and LLM assessment failed"
            }
        
        try:
            quick_docs = rag_processor.retrieve(current_query, k=2)
            if not quick_docs:
                return {
                    "current_query": current_query,
                    "original_query": current_query,
                    "is_rag_relevant": False,
                    "relevance_reason": "No relevant documents found (fallback check)"
                }
            else:
                return {
                    "current_query": current_query,
                    "original_query": current_query,
                    "is_rag_relevant": True,
                    "relevance_reason": f"Found {len(quick_docs)} potentially relevant documents (fallback check)"
                }
        except Exception as fallback_error:
            print(f"[GRAPH] Fallback check also failed: {fallback_error}")
            return {
                "current_query": current_query,
                "original_query": current_query,
                "is_rag_relevant": False,
                "relevance_reason": "Both LLM assessment and document check failed"
            }

# Step 3: Tool calling logic
async def tools_wrapper_node(state: State):
    log_step(state, "tools")
    return ToolNode(TOOLS).invoke(state)

# Enhanced RAG: Complete Pipeline in One Node
async def rag_complete_pipeline(state: State) -> dict:
    """Complete enhanced RAG pipeline in a single node - handles all advanced features."""
    log_step(state, "rag_complete_pipeline")

    if not state.current_query:
        raise ValueError("Missing query for RAG processing.")

    print(f"[RAG] Starting complete enhanced RAG pipeline for: '{state.current_query}'")

    # Ensure documents are loaded
    if not rag_processor.vectorstore:
        print("[RAG] Loading documents...")
        rag_processor.force_reload_documents("documents/")
    
    # Reset cost tracking
    rag_processor.reset_cost_tracking()
    
    # STEP 1: Initial Retrieval
    print("[RAG] Step 1: Initial document retrieval...")
    retrieved_docs = rag_processor.retrieve(state.current_query)
    print(f"[RAG] Retrieved {len(retrieved_docs)} documents")
    
    # STEP 2: Grade Documents
    print("[RAG] Step 2: Grading document relevance...")
    graded_docs = await rag_processor.grade_documents(state.current_query, retrieved_docs)
    relevant_docs = [doc_grade["document"] for doc_grade in graded_docs if doc_grade["grade"] == "relevant"]
    retrieval_quality = await rag_processor.assess_retrieval_quality(state.current_query, graded_docs)
    print(f"[RAG] Graded {len(graded_docs)} documents, {len(relevant_docs)} relevant, quality: {retrieval_quality}")
    
    # STEP 3: Query Rewriting & Re-retrieval (if needed)
    rewritten_query = None
    if retrieval_quality in ["weak", "poor"]:
        print("[RAG] Step 3: Query rewriting and re-retrieval...")
        retrieval_context = f"Previous retrieval found {len(relevant_docs)} relevant documents out of {len(retrieved_docs)} total. Quality: {retrieval_quality}"
        rewritten_query = await rag_processor.rewrite_query(state.original_query, retrieval_context)
        print(f"[RAG] Query rewritten to: '{rewritten_query}'")
        
        # Re-retrieve with rewritten query
        new_retrieved_docs = rag_processor.retrieve(rewritten_query)
        new_graded_docs = await rag_processor.grade_documents(rewritten_query, new_retrieved_docs)
        new_relevant_docs = [doc_grade["document"] for doc_grade in new_graded_docs if doc_grade["grade"] == "relevant"]
        new_retrieval_quality = await rag_processor.assess_retrieval_quality(rewritten_query, new_graded_docs)
        
        # Use new results if better
        if new_retrieval_quality != "poor" and len(new_relevant_docs) > len(relevant_docs):
            retrieved_docs = new_retrieved_docs
            graded_docs = new_graded_docs
            relevant_docs = new_relevant_docs
            retrieval_quality = new_retrieval_quality
            current_query = rewritten_query
        else:
            current_query = state.current_query
    else:
        current_query = state.current_query
    
    # STEP 4: Response Generation
    print("[RAG] Step 4: Generating response...")
    docs_for_generation = relevant_docs if relevant_docs else retrieved_docs
    if not docs_for_generation:
        rag_response = "Sorry, I could not find relevant information in the documents."
    else:
        rag_response = await rag_processor.generate_response(current_query, docs_for_generation)
    print(f"[RAG] Generated response: {rag_response[:100]}...")
    
    # STEP 5: Response Verification
    print("[RAG] Step 5: Verifying response...")
    if rag_response and docs_for_generation:
        verification_result = await rag_processor.verify_response(current_query, rag_response, docs_for_generation)
    else:
        verification_result = {"is_grounded": False, "confidence": 0.0, "issues": ["No response or documents to verify"]}
    print(f"[RAG] Verification: grounded={verification_result.get('is_grounded')}, confidence={verification_result.get('confidence', 0):.2f}")
    
    # STEP 6: Enhanced Evaluation
    print("[RAG] Step 6: Enhanced evaluation...")
    baseline_docs = retrieved_docs[:4] if retrieved_docs else []
    baseline_response = await rag_processor.generate_response(state.original_query, baseline_docs) if baseline_docs else "No baseline response available"
    
    cost_data = rag_processor.get_cost_summary()
    enhanced_metrics = rag_evaluator.evaluate_enhanced_rag(
        query=current_query,
        baseline_response=baseline_response,
        enhanced_response=rag_response,
        graded_documents=graded_docs,
        verification_result=verification_result,
        cost_data=cost_data,
        rewrite_used=rewritten_query is not None,
        original_query=state.original_query
    )
    
    basic_metrics = rag_evaluator.evaluate_response(
        query=current_query,
        retrieved_docs=relevant_docs if relevant_docs else retrieved_docs,
        response=rag_response
    )
    
    print(f"[RAG] Enhanced evaluation complete. Improvement: {enhanced_metrics.improvement:.2f}")
    
    # Return all the data but let final_response handle message formatting
    print(f"[RAG] ✅ Complete enhanced RAG pipeline finished successfully!")
    
    return {
        "retrieved_documents": retrieved_docs,
        "document_grades": graded_docs,
        "relevant_documents": relevant_docs,
        "rewritten_query": rewritten_query,
        "current_query": current_query,
        "rag_response": rag_response,
        "verification_result": verification_result,
        "retrieval_quality": retrieval_quality,
        "evaluation_metrics": basic_metrics.__dict__,
        "cost_tracking": cost_data,
        "performance_comparison": {
            "baseline_score": enhanced_metrics.baseline_score,
            "enhanced_score": enhanced_metrics.enhanced_score,
            "improvement": enhanced_metrics.improvement,
            "cost_effective": enhanced_metrics.cost_analysis.get("is_cost_effective", False)
        }
    }

# Final response node to properly format both RAG and general responses
async def final_response(state: State) -> dict:
    """Final response formatting for both RAG and general responses."""
    log_step(state, "final_response")
    
    if state.rag_response:
        # Format RAG response with enhanced information
        response_text = state.rag_response
        
        # Add verification status if verification failed
        if state.verification_result and not state.verification_result.get("is_grounded", True):
            response_text += "\n\n[Note: This response may not be fully grounded in the source documents]"
        
        # Add performance information if available
        if state.performance_comparison and state.performance_comparison.get("improvement", 0) > 0:
            improvement = state.performance_comparison["improvement"]
            response_text += f"\n\n[Enhanced RAG Performance: {improvement:.1%} improvement over baseline]"
        
        print(f"[FINAL] Formatting RAG response with enhanced information")
        final_message = AIMessage(content=response_text)
        return {"messages": [final_message]}
    else:
        # Handle case where no RAG response (fallback)
        print(f"[FINAL] No RAG response found, using existing messages")
        # Messages should already be set by model_direct_response
        return {"messages": []}  # No new messages needed

# === Enhanced Routing Logic ===

async def model_direct_response(state: State) -> dict:
    """Generate a direct model response for non-RAG queries."""
    log_step(state, "model_direct_response")
    
    configuration = Configuration.from_context()
    
    try:
        model = load_chat_model(configuration.model).bind_tools(TOOLS)
        system_message_text = configuration.system_prompt.format(system_time=datetime.now(tz=timezone.utc).isoformat())
        
        # Add context about why we're not using RAG
        system_message_text += f"\n\nNote: This query was determined to be answerable with general knowledge. Reason: {state.relevance_reason}. Please provide a helpful direct response or suggest using tools if appropriate."

        from langchain_core.messages import SystemMessage
        system_msg = SystemMessage(content=system_message_text)
        messages_for_model = [system_msg] + list(state.messages)

        response = cast(AIMessage, await model.ainvoke(messages_for_model))
        
    except Exception as e:
        # Fallback without tools
        if "function" in str(e).lower() or "tool" in str(e).lower():
            print(f"Tool calling failed, falling back to text-only mode: {e}")
            model = load_chat_model(configuration.model)
            system_message_text = configuration.system_prompt.format(system_time=datetime.now(tz=timezone.utc).isoformat())
            system_message_text += f"\n\nNote: This query can be answered with general knowledge. Reason: {state.relevance_reason}. Tool calling is not available. Please provide a direct text response."

            from langchain_core.messages import SystemMessage
            system_msg = SystemMessage(content=system_message_text)
            messages_for_model = [system_msg] + list(state.messages)

            response = cast(AIMessage, await model.ainvoke(messages_for_model))
        else:
            raise e

    return {"messages": [response]}

def route_after_call_model(state: State) -> Literal["rag_complete_pipeline", "model_direct_response"]:
    """Route based on query relevance determined in call_model."""
    if state.is_rag_relevant:
        return "rag_complete_pipeline"  # Go directly to RAG for relevant queries
    else:
        return "model_direct_response"  # Use model for non-relevant queries

def route_after_model_response(state: State) -> Literal["__end__", "tools"]:
    """Route after model direct response - check for tool calls."""
    last_message = state.messages[-1]
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    else:
        return "__end__"

def route_after_tools(state: State) -> Literal["model_direct_response"]:
    """After tools are executed, go back to model for final response."""
    return "model_direct_response"

# === Build the Enhanced LangGraph ===

builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Core agent steps
builder.add_node("call_model", call_model)
builder.add_node("model_direct_response", model_direct_response)
builder.add_node("tools", tools_wrapper_node)

# Enhanced RAG steps
builder.add_node("rag_complete_pipeline", rag_complete_pipeline)
builder.add_node("final_response", final_response)

# Flow definition
builder.add_edge("__start__", "call_model")
builder.add_conditional_edges("call_model", route_after_call_model)
builder.add_conditional_edges("model_direct_response", route_after_model_response)
builder.add_conditional_edges("tools", route_after_tools)

# Enhanced RAG flow
builder.add_edge("rag_complete_pipeline", "final_response")
builder.add_edge("final_response", "__end__")

graph = builder.compile(name="Enhanced ReAct Agent with Advanced RAG")
