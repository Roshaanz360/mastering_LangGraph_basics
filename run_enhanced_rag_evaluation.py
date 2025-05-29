#!/usr/bin/env python3
"""
Comprehensive Enhanced RAG Agent Evaluation Script

This script:
1. Loads predefined questions and ground truths from JSON
2. Runs the enhanced RAG agent with grading, rewriting, and verification
3. Compares performance against baseline RAG
4. Generates detailed quantitative and qualitative reports
5. Tracks cost and performance metrics
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from react_agent.rag import RAGProcessor
from react_agent.evaluator import RAGEvaluator, EnhancedRAGMetrics
from react_agent.graph import graph
from react_agent.state import State
from langchain_core.messages import HumanMessage

class ComprehensiveRAGEvaluator:
    def __init__(self, questions_file: str = "evaluation_questions.json"):
        self.rag_processor = RAGProcessor()
        self.evaluator = RAGEvaluator()
        self.questions_file = questions_file
        self.evaluation_results = []
        self.baseline_results = []
        
        # Load evaluation questions
        self.questions = self._load_questions()
        
        # Ensure documents are loaded
        if not self.rag_processor.vectorstore:
            print("[EVAL] Loading documents for evaluation...")
            self.rag_processor.force_reload_documents("documents/")
    
    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load evaluation questions from JSON file."""
        try:
            with open(self.questions_file, 'r') as f:
                data = json.load(f)
            questions = data.get("evaluation_dataset", {}).get("questions", [])
            print(f"[EVAL] Loaded {len(questions)} evaluation questions")
            return questions
        except FileNotFoundError:
            print(f"[EVAL] Warning: Questions file {self.questions_file} not found. Using default questions.")
            return self._get_default_questions()
        except Exception as e:
            print(f"[EVAL] Error loading questions: {e}. Using default questions.")
            return self._get_default_questions()
    
    def _get_default_questions(self) -> List[Dict[str, Any]]:
        """Fallback default questions if JSON file is not available."""
        return [
            {
                "id": 1,
                "question": "What is reward hacking in AI?",
                "ground_truth": "Reward hacking occurs when an AI system finds unexpected ways to maximize its reward function",
                "category": "AI Safety",
                "difficulty": "medium"
            },
            {
                "id": 2,
                "question": "How can we prevent hallucination in language models?",
                "ground_truth": "Techniques include grounding responses in source documents and verification",
                "category": "AI Safety", 
                "difficulty": "medium"
            }
        ]
    
    async def run_baseline_rag(self, query: str) -> Dict[str, Any]:
        """Run baseline RAG without enhanced features."""
        print(f"[BASELINE] Running baseline RAG for: '{query[:50]}...'")
        
        # Simple retrieval without grading
        retrieved_docs = self.rag_processor.retrieve(query)
        
        # Use top 4 documents without grading
        docs_for_generation = retrieved_docs[:4] if retrieved_docs else []
        
        # Generate response
        if docs_for_generation:
            response = await self.rag_processor.generate_response(query, docs_for_generation)
        else:
            response = "Sorry, I couldn't find relevant information to answer your question."
        
        return {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "response": response,
            "num_docs_used": len(docs_for_generation)
        }
    
    async def run_enhanced_rag(self, query: str, ground_truth: str = None) -> Dict[str, Any]:
        """Run enhanced RAG with all features."""
        print(f"[ENHANCED] Running enhanced RAG for: '{query[:50]}...'")
        
        # Reset cost tracking
        self.rag_processor.reset_cost_tracking()
        
        # Step 1: Initial retrieval
        retrieved_docs = self.rag_processor.retrieve(query)
        
        # Step 2: Grade documents
        graded_docs = await self.rag_processor.grade_documents(query, retrieved_docs)
        relevant_docs = [doc for doc in graded_docs if doc["grade"] == "relevant"]
        
        # Step 3: Assess retrieval quality
        retrieval_quality = await self.rag_processor.assess_retrieval_quality(query, graded_docs)
        
        # Step 4: Query rewriting if needed
        rewritten_query = None
        final_query = query
        if retrieval_quality in ["weak", "poor"]:
            retrieval_context = f"Previous retrieval found {len(relevant_docs)} relevant documents out of {len(retrieved_docs)} total. Quality: {retrieval_quality}"
            rewritten_query = await self.rag_processor.rewrite_query(query, retrieval_context)
            
            # Re-retrieve with rewritten query
            new_retrieved_docs = self.rag_processor.retrieve(rewritten_query)
            new_graded_docs = await self.rag_processor.grade_documents(rewritten_query, new_retrieved_docs)
            new_relevant_docs = [doc for doc in new_graded_docs if doc["grade"] == "relevant"]
            new_retrieval_quality = await self.rag_processor.assess_retrieval_quality(rewritten_query, new_graded_docs)
            
            # Use new results if better
            if new_retrieval_quality != "poor" and len(new_relevant_docs) >= len(relevant_docs):
                retrieved_docs = new_retrieved_docs
                graded_docs = new_graded_docs
                relevant_docs = new_relevant_docs
                retrieval_quality = new_retrieval_quality
                final_query = rewritten_query
        
        # Step 5: Generate response
        docs_for_generation = [doc["document"] for doc in relevant_docs] if relevant_docs else retrieved_docs
        enhanced_response = await self.rag_processor.generate_response(final_query, docs_for_generation)
        
        # Step 6: Verify response
        verification_result = await self.rag_processor.verify_response(
            final_query, enhanced_response, docs_for_generation
        )
        
        # Step 7: Get cost data
        cost_data = self.rag_processor.get_cost_summary()
        
        return {
            "query": query,
            "final_query": final_query,
            "rewritten_query": rewritten_query,
            "retrieved_docs": retrieved_docs,
            "graded_docs": graded_docs,
            "relevant_docs": relevant_docs,
            "retrieval_quality": retrieval_quality,
            "response": enhanced_response,
            "verification_result": verification_result,
            "cost_data": cost_data,
            "num_docs_used": len(docs_for_generation)
        }
    
    async def evaluate_single_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single question with both baseline and enhanced RAG."""
        question = question_data["question"]
        ground_truth = question_data.get("ground_truth", "")
        
        print(f"\n{'='*80}")
        print(f"Evaluating Question {question_data.get('id', '?')}: {question}")
        print(f"Category: {question_data.get('category', 'Unknown')} | Difficulty: {question_data.get('difficulty', 'Unknown')}")
        print(f"{'='*80}")
        
        # Run baseline RAG
        baseline_result = await self.run_baseline_rag(question)
        
        # Run enhanced RAG
        enhanced_result = await self.run_enhanced_rag(question, ground_truth)
        
        # Evaluate with enhanced metrics
        enhanced_metrics = self.evaluator.evaluate_enhanced_rag(
            query=question,
            baseline_response=baseline_result["response"],
            enhanced_response=enhanced_result["response"],
            graded_documents=enhanced_result["graded_docs"],
            verification_result=enhanced_result["verification_result"],
            cost_data=enhanced_result["cost_data"],
            rewrite_used=enhanced_result["rewritten_query"] is not None,
            original_query=question,
            ground_truth=ground_truth
        )
        
        # Compile evaluation result
        evaluation_result = {
            "question_data": question_data,
            "baseline_result": baseline_result,
            "enhanced_result": enhanced_result,
            "enhanced_metrics": enhanced_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # Print summary
        print(f"\n[SUMMARY] Question {question_data.get('id', '?')}:")
        print(f"  Baseline Response: {baseline_result['response'][:100]}...")
        print(f"  Enhanced Response: {enhanced_result['response'][:100]}...")
        print(f"  Improvement: {enhanced_metrics.improvement:.3f}")
        print(f"  Retrieval Quality: {enhanced_result['retrieval_quality']}")
        print(f"  Query Rewritten: {'Yes' if enhanced_result['rewritten_query'] else 'No'}")
        print(f"  Verification Grounded: {enhanced_result['verification_result'].get('is_grounded', 'Unknown')}")
        print(f"  Cost: ${enhanced_result['cost_data'].get('estimated_cost_usd', 0):.4f}")
        
        return evaluation_result
    
    async def run_full_evaluation(self) -> Dict[str, Any]:
        """Run evaluation on all questions."""
        print(f"\n{'='*100}")
        print(f"STARTING COMPREHENSIVE ENHANCED RAG EVALUATION")
        print(f"Total Questions: {len(self.questions)}")
        print(f"{'='*100}")
        
        results = []
        
        for i, question_data in enumerate(self.questions, 1):
            print(f"\n[PROGRESS] Question {i}/{len(self.questions)}")
            
            try:
                result = await self.evaluate_single_question(question_data)
                results.append(result)
                
                # Small delay between questions
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"[ERROR] Failed to evaluate question {question_data.get('id', '?')}: {e}")
                import traceback
                traceback.print_exc()
                
                # Add error result
                results.append({
                    "question_data": question_data,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        self.evaluation_results = results
        return self._generate_final_report()
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final evaluation report."""
        successful_results = [r for r in self.evaluation_results if "error" not in r]
        failed_results = [r for r in self.evaluation_results if "error" in r]
        
        if not successful_results:
            return {"error": "No successful evaluations to report on"}
        
        # Calculate aggregate metrics
        improvements = [r["enhanced_metrics"].improvement for r in successful_results]
        avg_improvement = sum(improvements) / len(improvements)
        
        cost_effective_count = sum(1 for r in successful_results 
                                 if r["enhanced_metrics"].cost_analysis.get("is_cost_effective", False))
        
        total_cost = sum(r["enhanced_result"]["cost_data"].get("estimated_cost_usd", 0) for r in successful_results)
        
        # Component performance
        grading_accuracies = [r["enhanced_metrics"].grading_performance.get("accuracy", 0) for r in successful_results]
        avg_grading_accuracy = sum(grading_accuracies) / len(grading_accuracies)
        
        rewriting_used = [r for r in successful_results if r["enhanced_result"]["rewritten_query"]]
        rewriting_effectiveness = [r["enhanced_metrics"].rewriting_performance.get("effectiveness", 0) for r in rewriting_used]
        avg_rewriting_effectiveness = sum(rewriting_effectiveness) / len(rewriting_effectiveness) if rewriting_effectiveness else 0
        
        verification_confidences = [r["enhanced_metrics"].verification_performance.get("confidence", 0) for r in successful_results]
        avg_verification_confidence = sum(verification_confidences) / len(verification_confidences)
        
        # Performance by category and difficulty
        category_performance = {}
        difficulty_performance = {}
        
        for result in successful_results:
            category = result["question_data"].get("category", "Unknown")
            difficulty = result["question_data"].get("difficulty", "Unknown")
            improvement = result["enhanced_metrics"].improvement
            
            if category not in category_performance:
                category_performance[category] = []
            category_performance[category].append(improvement)
            
            if difficulty not in difficulty_performance:
                difficulty_performance[difficulty] = []
            difficulty_performance[difficulty].append(improvement)
        
        # Calculate averages
        for category in category_performance:
            category_performance[category] = sum(category_performance[category]) / len(category_performance[category])
        
        for difficulty in difficulty_performance:
            difficulty_performance[difficulty] = sum(difficulty_performance[difficulty]) / len(difficulty_performance[difficulty])
        
        # Compile final report
        final_report = {
            "evaluation_summary": {
                "total_questions": len(self.evaluation_results),
                "successful_evaluations": len(successful_results),
                "failed_evaluations": len(failed_results),
                "success_rate": len(successful_results) / len(self.evaluation_results),
                "timestamp": datetime.now().isoformat()
            },
            "quantitative_results": {
                "average_improvement": avg_improvement,
                "improvement_percentage": avg_improvement * 100,
                "positive_improvements": len([imp for imp in improvements if imp > 0]),
                "cost_effective_tests": cost_effective_count,
                "cost_effectiveness_rate": cost_effective_count / len(successful_results),
                "total_cost": total_cost,
                "average_cost_per_query": total_cost / len(successful_results)
            },
            "component_performance": {
                "document_grading": {
                    "average_accuracy": avg_grading_accuracy,
                    "high_accuracy_count": len([acc for acc in grading_accuracies if acc > 0.8])
                },
                "query_rewriting": {
                    "queries_rewritten": len(rewriting_used),
                    "rewrite_rate": len(rewriting_used) / len(successful_results),
                    "average_effectiveness": avg_rewriting_effectiveness
                },
                "response_verification": {
                    "average_confidence": avg_verification_confidence,
                    "high_confidence_count": len([conf for conf in verification_confidences if conf > 0.8])
                }
            },
            "performance_by_category": category_performance,
            "performance_by_difficulty": difficulty_performance,
            "detailed_results": self.evaluation_results
        }
        
        return final_report
    
    def save_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """Save the evaluation report to a file."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"enhanced_rag_evaluation_report_{timestamp}.json"
        
        try:
            # Convert any dataclass objects to dicts for JSON serialization
            def convert_for_json(obj):
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                return str(obj)
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=convert_for_json)
            
            print(f"[SAVE] Evaluation report saved to: {filename}")
            return filename
        except Exception as e:
            print(f"[SAVE] Error saving report: {e}")
            return ""
    
    def generate_human_readable_report(self, report: Dict[str, Any]) -> str:
        """Generate a human-readable text report."""
        summary = report["evaluation_summary"]
        quant = report["quantitative_results"]
        comp = report["component_performance"]
        
        text_report = f"""
Enhanced RAG Agent Comprehensive Evaluation Report
=================================================
Generated: {summary['timestamp']}

EXECUTIVE SUMMARY:
-----------------
Total Questions Evaluated: {summary['total_questions']}
Successful Evaluations: {summary['successful_evaluations']} ({summary['success_rate']*100:.1f}%)
Average Performance Improvement: {quant['improvement_percentage']:.1f}%
Cost Effectiveness Rate: {quant['cost_effectiveness_rate']*100:.1f}%

QUANTITATIVE RESULTS:
--------------------
Performance Metrics:
  • Average Improvement: {quant['average_improvement']:.3f} ({quant['improvement_percentage']:.1f}%)
  • Questions with Positive Improvement: {quant['positive_improvements']}/{summary['successful_evaluations']}
  • Cost Effective Evaluations: {quant['cost_effective_tests']}/{summary['successful_evaluations']}

Cost Analysis:
  • Total Cost: ${quant['total_cost']:.4f}
  • Average Cost per Query: ${quant['average_cost_per_query']:.4f}

COMPONENT PERFORMANCE:
---------------------
Document Grading:
  • Average Accuracy: {comp['document_grading']['average_accuracy']:.3f}
  • High Accuracy (>0.8): {comp['document_grading']['high_accuracy_count']}/{summary['successful_evaluations']}

Query Rewriting:
  • Queries Rewritten: {comp['query_rewriting']['queries_rewritten']}/{summary['successful_evaluations']} ({comp['query_rewriting']['rewrite_rate']*100:.1f}%)
  • Average Effectiveness: {comp['query_rewriting']['average_effectiveness']:.3f}

Response Verification:
  • Average Confidence: {comp['response_verification']['average_confidence']:.3f}
  • High Confidence (>0.8): {comp['response_verification']['high_confidence_count']}/{summary['successful_evaluations']}

PERFORMANCE BY CATEGORY:
-----------------------
"""
        
        for category, performance in report["performance_by_category"].items():
            text_report += f"  • {category}: {performance:.3f} ({performance*100:.1f}% improvement)\n"
        
        text_report += f"""
PERFORMANCE BY DIFFICULTY:
-------------------------
"""
        
        for difficulty, performance in report["performance_by_difficulty"].items():
            text_report += f"  • {difficulty}: {performance:.3f} ({performance*100:.1f}% improvement)\n"
        
        # Add qualitative observations
        all_observations = []
        for result in report["detailed_results"]:
            if "enhanced_metrics" in result:
                all_observations.extend(result["enhanced_metrics"].qualitative_observations)
        
        observation_counts = {}
        for obs in all_observations:
            observation_counts[obs] = observation_counts.get(obs, 0) + 1
        
        text_report += f"""
QUALITATIVE OBSERVATIONS:
------------------------
"""
        
        for obs, count in sorted(observation_counts.items(), key=lambda x: x[1], reverse=True):
            text_report += f"  • {obs} (observed in {count} evaluations)\n"
        
        # Add recommendations
        text_report += f"""
RECOMMENDATIONS:
---------------
"""
        
        if quant['improvement_percentage'] > 10:
            text_report += "  • Enhanced RAG shows significant improvement and should be deployed\n"
        elif quant['improvement_percentage'] > 0:
            text_report += "  • Enhanced RAG shows modest improvement, consider cost-benefit analysis\n"
        else:
            text_report += "  • Enhanced RAG does not improve performance, investigate issues\n"
        
        if quant['cost_effectiveness_rate'] > 0.7:
            text_report += "  • Cost efficiency is good, enhanced features provide value\n"
        else:
            text_report += "  • Consider optimizing costs or reducing enhanced features\n"
        
        if comp['document_grading']['average_accuracy'] < 0.7:
            text_report += "  • Document grading needs improvement, consider tuning prompts\n"
        
        if comp['query_rewriting']['rewrite_rate'] > 0.5:
            text_report += "  • High query rewriting rate suggests initial retrieval needs improvement\n"
        
        return text_report

async def main():
    """Main evaluation function."""
    print("Enhanced RAG Agent Comprehensive Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ComprehensiveRAGEvaluator()
    
    # Run full evaluation
    final_report = await evaluator.run_full_evaluation()
    
    if "error" in final_report:
        print(f"[ERROR] Evaluation failed: {final_report['error']}")
        return
    
    # Save detailed JSON report
    json_filename = evaluator.save_report(final_report)
    
    # Generate and save human-readable report
    text_report = evaluator.generate_human_readable_report(final_report)
    text_filename = f"enhanced_rag_evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(text_filename, 'w') as f:
        f.write(text_report)
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Detailed JSON Report: {json_filename}")
    print(f"Human-Readable Summary: {text_filename}")
    
    print("\nEVALUATION SUMMARY:")
    print("-" * 40)
    print(text_report[:1500] + "..." if len(text_report) > 1500 else text_report)

if __name__ == "__main__":
    asyncio.run(main()) 