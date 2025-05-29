#!/usr/bin/env python3
"""
Test script for Enhanced RAG Agent with document grading, query rewriting, and verification.
This script demonstrates the enhanced capabilities and generates evaluation reports.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Any

from src.react_agent.rag import RAGProcessor
from src.react_agent.evaluator import RAGEvaluator, EnhancedRAGMetrics
from src.react_agent.graph import graph
from src.react_agent.state import State
from langchain_core.messages import HumanMessage

class EnhancedRAGTester:
    def __init__(self):
        self.rag_processor = RAGProcessor()
        self.evaluator = RAGEvaluator()
        self.test_results = []
        
        # Ensure documents are loaded
        if not self.rag_processor.vectorstore:
            print("[TEST] Loading documents for testing...")
            self.rag_processor.force_reload_documents("documents/")
    
    async def test_enhanced_rag_workflow(self, query: str, expected_answer: str = None) -> Dict[str, Any]:
        """Test the complete enhanced RAG workflow."""
        print(f"\n{'='*60}")
        print(f"Testing Enhanced RAG with query: '{query}'")
        print(f"{'='*60}")
        
        # Initialize state
        initial_state = State(
            messages=[HumanMessage(content=query)],
            current_query=query,
            original_query=query
        )
        
        # Run the enhanced RAG workflow
        try:
            # Step 1: Initial retrieval
            print("\n[STEP 1] Initial document retrieval...")
            retrieval_result = self.rag_processor.retrieve(query)
            print(f"Retrieved {len(retrieval_result)} documents")
            
            # Step 2: Grade documents
            print("\n[STEP 2] Grading document relevance...")
            graded_docs = await self.rag_processor.grade_documents(query, retrieval_result)
            relevant_docs = [doc for doc in graded_docs if doc["grade"] == "relevant"]
            print(f"Graded {len(graded_docs)} documents, {len(relevant_docs)} marked as relevant")
            
            # Step 3: Assess retrieval quality
            print("\n[STEP 3] Assessing retrieval quality...")
            retrieval_quality = await self.rag_processor.assess_retrieval_quality(query, graded_docs)
            print(f"Retrieval quality: {retrieval_quality}")
            
            # Step 4: Query rewriting (if needed)
            rewritten_query = None
            if retrieval_quality in ["weak", "poor"]:
                print("\n[STEP 4] Rewriting query for better retrieval...")
                retrieval_context = f"Previous retrieval found {len(relevant_docs)} relevant documents out of {len(retrieval_result)} total. Quality: {retrieval_quality}"
                rewritten_query = await self.rag_processor.rewrite_query(query, retrieval_context)
                print(f"Query rewritten to: '{rewritten_query}'")
                
                # Step 5: Re-retrieval with rewritten query
                print("\n[STEP 5] Re-retrieving with rewritten query...")
                new_retrieval_result = self.rag_processor.retrieve(rewritten_query)
                new_graded_docs = await self.rag_processor.grade_documents(rewritten_query, new_retrieval_result)
                new_relevant_docs = [doc for doc in new_graded_docs if doc["grade"] == "relevant"]
                new_retrieval_quality = await self.rag_processor.assess_retrieval_quality(rewritten_query, new_graded_docs)
                
                print(f"Re-retrieved {len(new_retrieval_result)} documents, {len(new_relevant_docs)} relevant")
                print(f"New retrieval quality: {new_retrieval_quality}")
                
                # Use new results if better
                if new_retrieval_quality != "poor" and len(new_relevant_docs) > len(relevant_docs):
                    retrieval_result = new_retrieval_result
                    graded_docs = new_graded_docs
                    relevant_docs = new_relevant_docs
                    retrieval_quality = new_retrieval_quality
                    query = rewritten_query  # Update query for generation
            
            # Step 6: Generate response
            print("\n[STEP 6] Generating response...")
            docs_for_generation = [doc["document"] for doc in relevant_docs] if relevant_docs else retrieval_result
            enhanced_response = await self.rag_processor.generate_response(query, docs_for_generation)
            print(f"Generated response: {enhanced_response[:200]}...")
            
            # Step 7: Verify response
            print("\n[STEP 7] Verifying response grounding...")
            verification_result = await self.rag_processor.verify_response(
                query, enhanced_response, docs_for_generation
            )
            print(f"Verification: grounded={verification_result.get('is_grounded')}, "
                  f"confidence={verification_result.get('confidence', 0):.2f}")
            
            # Step 8: Generate baseline for comparison
            print("\n[STEP 8] Generating baseline response for comparison...")
            baseline_docs = retrieval_result[:4]  # Simple top-4 retrieval
            baseline_response = await self.rag_processor.generate_response(query, baseline_docs)
            
            # Step 9: Enhanced evaluation
            print("\n[STEP 9] Performing enhanced evaluation...")
            cost_data = self.rag_processor.get_cost_summary()
            
            enhanced_metrics = self.evaluator.evaluate_enhanced_rag(
                query=query,
                baseline_response=baseline_response,
                enhanced_response=enhanced_response,
                graded_documents=graded_docs,
                verification_result=verification_result,
                cost_data=cost_data,
                rewrite_used=rewritten_query is not None,
                original_query=query,
                ground_truth=expected_answer
            )
            
            # Compile results
            test_result = {
                "query": query,
                "original_query": query,
                "rewritten_query": rewritten_query,
                "retrieval_quality": retrieval_quality,
                "baseline_response": baseline_response,
                "enhanced_response": enhanced_response,
                "verification_result": verification_result,
                "enhanced_metrics": enhanced_metrics,
                "cost_data": cost_data,
                "timestamp": datetime.now().isoformat()
            }
            
            self.test_results.append(test_result)
            
            print(f"\n[RESULTS] Enhanced RAG Test Complete!")
            print(f"Improvement: {enhanced_metrics.improvement:.2f}")
            print(f"Cost effective: {enhanced_metrics.cost_analysis.get('is_cost_effective', False)}")
            
            return test_result
            
        except Exception as e:
            print(f"[ERROR] Test failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "query": query}
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive evaluation report."""
        if not self.test_results:
            return "No test results available for report generation."
        
        # Calculate aggregate metrics
        total_tests = len(self.test_results)
        successful_tests = [r for r in self.test_results if "error" not in r]
        
        if not successful_tests:
            return "No successful tests to report on."
        
        improvements = [r["enhanced_metrics"].improvement for r in successful_tests]
        avg_improvement = sum(improvements) / len(improvements)
        
        cost_effective_count = sum(1 for r in successful_tests 
                                 if r["enhanced_metrics"].cost_analysis.get("is_cost_effective", False))
        
        total_cost = sum(r["cost_data"].get("estimated_cost_usd", 0) for r in successful_tests)
        
        # Generate report
        report = f"""
Enhanced RAG Agent Evaluation Report
===================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

QUANTITATIVE RESULTS:
--------------------
Total Tests: {total_tests}
Successful Tests: {len(successful_tests)}
Average Improvement: {avg_improvement:.3f} ({avg_improvement*100:.1f}%)
Cost Effective Tests: {cost_effective_count}/{len(successful_tests)} ({cost_effective_count/len(successful_tests)*100:.1f}%)
Total Cost: ${total_cost:.4f}

COMPONENT PERFORMANCE:
---------------------
"""
        
        # Aggregate component performance
        grading_accuracies = [r["enhanced_metrics"].grading_performance.get("accuracy", 0) for r in successful_tests]
        rewriting_effectiveness = [r["enhanced_metrics"].rewriting_performance.get("effectiveness", 0) for r in successful_tests if r["rewritten_query"]]
        verification_confidences = [r["enhanced_metrics"].verification_performance.get("confidence", 0) for r in successful_tests]
        
        report += f"Document Grading:\n"
        report += f"  - Average Accuracy: {sum(grading_accuracies)/len(grading_accuracies):.3f}\n"
        report += f"  - Performed Well (>0.8): {sum(1 for acc in grading_accuracies if acc > 0.8)}/{len(grading_accuracies)}\n\n"
        
        if rewriting_effectiveness:
            report += f"Query Rewriting:\n"
            report += f"  - Average Effectiveness: {sum(rewriting_effectiveness)/len(rewriting_effectiveness):.3f}\n"
            report += f"  - Queries Rewritten: {len(rewriting_effectiveness)}/{len(successful_tests)}\n\n"
        
        report += f"Response Verification:\n"
        report += f"  - Average Confidence: {sum(verification_confidences)/len(verification_confidences):.3f}\n"
        report += f"  - High Confidence (>0.8): {sum(1 for conf in verification_confidences if conf > 0.8)}/{len(verification_confidences)}\n\n"
        
        # Qualitative observations
        report += "QUALITATIVE OBSERVATIONS:\n"
        report += "------------------------\n"
        
        all_observations = []
        for result in successful_tests:
            all_observations.extend(result["enhanced_metrics"].qualitative_observations)
        
        # Count common observations
        observation_counts = {}
        for obs in all_observations:
            observation_counts[obs] = observation_counts.get(obs, 0) + 1
        
        for obs, count in sorted(observation_counts.items(), key=lambda x: x[1], reverse=True):
            report += f"- {obs} (observed in {count}/{len(successful_tests)} tests)\n"
        
        # Individual test summaries
        report += f"\nINDIVIDUAL TEST RESULTS:\n"
        report += f"------------------------\n"
        
        for i, result in enumerate(successful_tests, 1):
            metrics = result["enhanced_metrics"]
            report += f"\nTest {i}: {result['query'][:50]}...\n"
            report += f"  Improvement: {metrics.improvement:.3f}\n"
            report += f"  Retrieval Quality: {result['retrieval_quality']}\n"
            report += f"  Query Rewritten: {'Yes' if result['rewritten_query'] else 'No'}\n"
            report += f"  Verification Grounded: {result['verification_result'].get('is_grounded', 'Unknown')}\n"
            report += f"  Cost: ${result['cost_data'].get('estimated_cost_usd', 0):.4f}\n"
        
        # Performance comparison summary
        report += f"\nPERFORMANCE COMPARISON:\n"
        report += f"----------------------\n"
        report += f"Enhanced RAG vs Baseline:\n"
        
        positive_improvements = [imp for imp in improvements if imp > 0]
        negative_improvements = [imp for imp in improvements if imp <= 0]
        
        report += f"  - Positive Improvements: {len(positive_improvements)}/{len(improvements)} tests\n"
        avg_positive = sum(positive_improvements)/len(positive_improvements) if positive_improvements else 0
        report += f"  - Average Positive Improvement: {avg_positive:.3f}\n"
        report += f"  - Tests with No Improvement: {len(negative_improvements)}\n"
        
        # Cost analysis
        report += f"\nCOST ANALYSIS:\n"
        report += f"-------------\n"
        report += f"Total Token Usage: {sum(r['cost_data'].get('total_tokens', 0) for r in successful_tests):,}\n"
        report += f"Average Cost per Query: ${total_cost/len(successful_tests):.4f}\n"
        
        token_breakdown = {}
        for result in successful_tests:
            for token_type, count in result["cost_data"].get("token_breakdown", {}).items():
                token_breakdown[token_type] = token_breakdown.get(token_type, 0) + count
        
        report += f"Token Breakdown:\n"
        for token_type, count in token_breakdown.items():
            report += f"  - {token_type}: {count:,} tokens\n"
        
        # Recommendations
        report += f"\nRECOMMENDATIONS:\n"
        report += f"---------------\n"
        
        if avg_improvement > 0.1:
            report += "- Enhanced RAG shows significant improvement and should be deployed\n"
        elif avg_improvement > 0:
            report += "- Enhanced RAG shows modest improvement, consider cost-benefit analysis\n"
        else:
            report += "- Enhanced RAG does not improve performance, investigate issues\n"
        
        if cost_effective_count / len(successful_tests) > 0.7:
            report += "- Cost efficiency is good, enhanced features provide value\n"
        else:
            report += "- Consider optimizing costs or reducing enhanced features\n"
        
        if sum(grading_accuracies) / len(grading_accuracies) < 0.7:
            report += "- Document grading needs improvement, consider tuning prompts\n"
        
        return report
    
    def export_results(self, filename: str = None) -> str:
        """Export test results to JSON file."""
        if not filename:
            filename = f"enhanced_rag_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Convert dataclasses to dicts for JSON serialization
            exportable_results = []
            for result in self.test_results:
                if "enhanced_metrics" in result:
                    # Convert EnhancedRAGMetrics to dict
                    metrics = result["enhanced_metrics"]
                    result["enhanced_metrics"] = {
                        "baseline_score": metrics.baseline_score,
                        "enhanced_score": metrics.enhanced_score,
                        "improvement": metrics.improvement,
                        "grading_performance": metrics.grading_performance,
                        "rewriting_performance": metrics.rewriting_performance,
                        "verification_performance": metrics.verification_performance,
                        "cost_analysis": metrics.cost_analysis,
                        "qualitative_observations": metrics.qualitative_observations
                    }
                exportable_results.append(result)
            
            with open(filename, 'w') as f:
                json.dump(exportable_results, f, indent=2, default=str)
            
            print(f"[EXPORT] Test results exported to {filename}")
            return filename
        except Exception as e:
            print(f"[EXPORT] Error exporting results: {e}")
            return ""

async def main():
    """Main test function."""
    print("Enhanced RAG Agent Testing Suite")
    print("=" * 50)
    
    tester = EnhancedRAGTester()
    
    # Test queries covering different scenarios
    test_queries = [
        {
            "query": "What is reward hacking in AI?",
            "expected": "Reward hacking occurs when an AI system finds unexpected ways to maximize its reward function"
        },
        {
            "query": "How can we prevent hallucination in language models?",
            "expected": "Techniques include grounding responses in source documents and verification"
        },
        {
            "query": "What are the main challenges in machine learning?",
            "expected": "Challenges include data quality, overfitting, and generalization"
        },
        {
            "query": "Explain the concept of artificial intelligence safety",
            "expected": "AI safety focuses on ensuring AI systems behave safely and as intended"
        }
    ]
    
    print(f"Running {len(test_queries)} test queries...\n")
    
    # Run tests
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n{'='*20} TEST {i}/{len(test_queries)} {'='*20}")
        await tester.test_enhanced_rag_workflow(
            test_case["query"], 
            test_case.get("expected")
        )
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    # Generate and save comprehensive report
    print(f"\n{'='*60}")
    print("GENERATING COMPREHENSIVE EVALUATION REPORT")
    print(f"{'='*60}")
    
    report = tester.generate_comprehensive_report()
    
    # Save report to file
    report_filename = f"enhanced_rag_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {report_filename}")
    print("\nReport Preview:")
    print("-" * 50)
    print(report[:1000] + "..." if len(report) > 1000 else report)
    
    # Export detailed results
    results_filename = tester.export_results()
    print(f"\nDetailed results exported to: {results_filename}")
    
    print(f"\n{'='*60}")
    print("ENHANCED RAG TESTING COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main()) 