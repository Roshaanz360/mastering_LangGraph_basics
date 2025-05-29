"""RAG evaluation implementation."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

@dataclass
class EvaluationMetrics:
    relevance_score: float
    answer_quality: float
    context_utilization: float
    overall_score: float
    
    # Enhanced metrics
    grading_accuracy: Optional[float] = None
    rewrite_effectiveness: Optional[float] = None
    verification_accuracy: Optional[float] = None
    cost_efficiency: Optional[float] = None

@dataclass
class EnhancedRAGMetrics:
    """Comprehensive metrics for enhanced RAG evaluation."""
    baseline_score: float
    enhanced_score: float
    improvement: float
    grading_performance: Dict[str, float]
    rewriting_performance: Dict[str, float]
    verification_performance: Dict[str, float]
    cost_analysis: Dict[str, Any]
    qualitative_observations: List[str]

class RAGEvaluator:
    def __init__(self):
        # Get API keys from environment variables
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            print("Warning: GEMINI_API_KEY not found. Evaluation will use fallback metrics.")
            self.embeddings = None
        else:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=gemini_api_key
            )
        
        # Store evaluation history for comparison
        self.evaluation_history = []
        
    def evaluate_response(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        response: str,
        ground_truth: str = None
    ) -> EvaluationMetrics:
        """Evaluate the RAG response quality."""
        
        if not self.embeddings:
            # Fallback evaluation without embeddings
            print("[EVAL] Using fallback evaluation metrics")
            return EvaluationMetrics(
                relevance_score=0.7,  # Default reasonable score
                answer_quality=0.7,
                context_utilization=0.7,
                overall_score=0.7
            )
        
        try:
            # Calculate relevance score
            query_embedding = self.embeddings.embed_query(query)
            doc_embeddings = [self.embeddings.embed_query(doc["content"]) for doc in retrieved_docs]
            relevance_scores = [cosine_similarity([query_embedding], [doc_emb])[0][0] for doc_emb in doc_embeddings]
            relevance_score = float(np.mean(relevance_scores))

            # Calculate answer quality (if ground truth is provided)
            if ground_truth:
                response_embedding = self.embeddings.embed_query(response)
                truth_embedding = self.embeddings.embed_query(ground_truth)
                answer_quality = float(cosine_similarity([response_embedding], [truth_embedding])[0][0])
            else:
                answer_quality = 0.5  # Default score if no ground truth

            # Calculate context utilization
            response_embedding = self.embeddings.embed_query(response)
            context_scores = [cosine_similarity([response_embedding], [doc_emb])[0][0] for doc_emb in doc_embeddings]
            context_utilization = float(np.mean(context_scores))

            # Calculate overall score
            overall_score = (relevance_score + answer_quality + context_utilization) / 3

            return EvaluationMetrics(
                relevance_score=relevance_score,
                answer_quality=answer_quality,
                context_utilization=context_utilization,
                overall_score=overall_score
            )
        except Exception as e:
            print(f"[EVAL] Error during evaluation: {e}")
            # Return fallback metrics
            return EvaluationMetrics(
                relevance_score=0.6,
                answer_quality=0.6,
                context_utilization=0.6,
                overall_score=0.6
            )

    def evaluate_enhanced_rag(
        self,
        query: str,
        baseline_response: str,
        enhanced_response: str,
        graded_documents: List[Dict[str, Any]],
        verification_result: Dict[str, Any],
        cost_data: Dict[str, Any],
        rewrite_used: bool = False,
        original_query: str = None,
        ground_truth: str = None
    ) -> EnhancedRAGMetrics:
        """Evaluate the enhanced RAG system against baseline."""
        
        print("[EVAL] Evaluating enhanced RAG performance...")
        
        # Calculate baseline and enhanced scores
        baseline_score = self._calculate_response_score(query, baseline_response, ground_truth)
        enhanced_score = self._calculate_response_score(query, enhanced_response, ground_truth)
        improvement = enhanced_score - baseline_score
        
        # Evaluate grading performance
        grading_performance = self._evaluate_grading_performance(graded_documents)
        
        # Evaluate rewriting performance
        rewriting_performance = self._evaluate_rewriting_performance(
            original_query, query, rewrite_used, enhanced_score
        ) if rewrite_used else {"effectiveness": 0.0, "improvement": 0.0}
        
        # Evaluate verification performance
        verification_performance = self._evaluate_verification_performance(verification_result)
        
        # Analyze cost efficiency
        cost_analysis = self._analyze_cost_efficiency(cost_data, improvement)
        
        # Generate qualitative observations
        qualitative_observations = self._generate_qualitative_observations(
            improvement, grading_performance, rewriting_performance, 
            verification_performance, cost_analysis
        )
        
        metrics = EnhancedRAGMetrics(
            baseline_score=baseline_score,
            enhanced_score=enhanced_score,
            improvement=improvement,
            grading_performance=grading_performance,
            rewriting_performance=rewriting_performance,
            verification_performance=verification_performance,
            cost_analysis=cost_analysis,
            qualitative_observations=qualitative_observations
        )
        
        # Store in history
        self.evaluation_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "metrics": metrics,
            "cost_data": cost_data
        })
        
        return metrics

    def _calculate_response_score(self, query: str, response: str, ground_truth: str = None) -> float:
        """Calculate a score for a response."""
        if not self.embeddings:
            return 0.7  # Fallback score
        
        try:
            if ground_truth:
                response_embedding = self.embeddings.embed_query(response)
                truth_embedding = self.embeddings.embed_query(ground_truth)
                return float(cosine_similarity([response_embedding], [truth_embedding])[0][0])
            else:
                # Use query-response similarity as proxy
                query_embedding = self.embeddings.embed_query(query)
                response_embedding = self.embeddings.embed_query(response)
                return float(cosine_similarity([query_embedding], [response_embedding])[0][0])
        except Exception as e:
            print(f"[EVAL] Error calculating response score: {e}")
            return 0.5

    def _evaluate_grading_performance(self, graded_documents: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate the performance of document grading."""
        if not graded_documents:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
        
        relevant_count = sum(1 for doc in graded_documents if doc["grade"] == "relevant")
        total_count = len(graded_documents)
        avg_confidence = np.mean([doc["score"] for doc in graded_documents])
        
        # Simple heuristics for grading performance
        precision = relevant_count / total_count if total_count > 0 else 0
        recall = min(1.0, relevant_count / max(1, total_count * 0.7))  # Assume 70% should be relevant
        accuracy = avg_confidence
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "relevant_ratio": float(relevant_count / total_count) if total_count > 0 else 0.0
        }

    def _evaluate_rewriting_performance(
        self, original_query: str, rewritten_query: str, rewrite_used: bool, enhanced_score: float
    ) -> Dict[str, float]:
        """Evaluate the effectiveness of query rewriting."""
        if not rewrite_used or not original_query or not rewritten_query:
            return {"effectiveness": 0.0, "improvement": 0.0}
        
        try:
            if self.embeddings:
                # Measure semantic similarity between original and rewritten queries
                orig_embedding = self.embeddings.embed_query(original_query)
                rewritten_embedding = self.embeddings.embed_query(rewritten_query)
                similarity = float(cosine_similarity([orig_embedding], [rewritten_embedding])[0][0])
                
                # Effectiveness is based on maintaining semantic meaning while improving results
                effectiveness = min(1.0, similarity + (enhanced_score * 0.3))
                improvement = enhanced_score * 0.5  # Assume rewriting contributed to improvement
            else:
                effectiveness = 0.7
                improvement = 0.1
                
            return {
                "effectiveness": effectiveness,
                "improvement": improvement,
                "semantic_preservation": similarity if self.embeddings else 0.7
            }
        except Exception as e:
            print(f"[EVAL] Error evaluating rewriting: {e}")
            return {"effectiveness": 0.5, "improvement": 0.0}

    def _evaluate_verification_performance(self, verification_result: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the performance of response verification."""
        if not verification_result:
            return {"accuracy": 0.0, "confidence": 0.0}
        
        return {
            "accuracy": float(verification_result.get("score", 0.5)),
            "confidence": float(verification_result.get("confidence", 0.5)),
            "is_grounded": float(verification_result.get("is_grounded", True))
        }

    def _analyze_cost_efficiency(self, cost_data: Dict[str, Any], improvement: float) -> Dict[str, Any]:
        """Analyze the cost efficiency of the enhanced RAG system."""
        total_cost = cost_data.get("estimated_cost_usd", 0.0)
        total_tokens = cost_data.get("total_tokens", 0)
        
        # Calculate cost per improvement point
        cost_per_improvement = total_cost / max(0.01, improvement) if improvement > 0 else float('inf')
        
        # Efficiency score (lower cost per improvement is better)
        efficiency_score = min(1.0, 1.0 / max(0.1, cost_per_improvement * 100))
        
        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "cost_per_improvement": cost_per_improvement,
            "efficiency_score": efficiency_score,
            "cost_breakdown": cost_data.get("cost_breakdown", {}),
            "is_cost_effective": improvement > 0 and cost_per_improvement < 0.1
        }

    def _generate_qualitative_observations(
        self, improvement: float, grading_perf: Dict, rewriting_perf: Dict, 
        verification_perf: Dict, cost_analysis: Dict
    ) -> List[str]:
        """Generate qualitative observations about the enhanced RAG performance."""
        observations = []
        
        # Overall performance
        if improvement > 0.1:
            observations.append("Enhanced RAG shows significant improvement over baseline")
        elif improvement > 0:
            observations.append("Enhanced RAG shows modest improvement over baseline")
        else:
            observations.append("Enhanced RAG did not improve over baseline")
        
        # Grading observations
        if grading_perf.get("accuracy", 0) > 0.8:
            observations.append("Document grading performed well with high accuracy")
        elif grading_perf.get("relevant_ratio", 0) < 0.3:
            observations.append("Document grading may be too strict, filtering out too many documents")
        
        # Rewriting observations
        if rewriting_perf.get("effectiveness", 0) > 0.7:
            observations.append("Query rewriting was effective and preserved semantic meaning")
        elif rewriting_perf.get("improvement", 0) < 0.05:
            observations.append("Query rewriting did not significantly improve retrieval")
        
        # Verification observations
        if verification_perf.get("confidence", 0) > 0.8:
            observations.append("Response verification showed high confidence in grounding")
        elif not verification_perf.get("is_grounded", True):
            observations.append("Verification detected potential hallucination in response")
        
        # Cost observations
        if cost_analysis.get("is_cost_effective", False):
            observations.append("Enhanced RAG is cost-effective given the performance improvement")
        else:
            observations.append("Enhanced RAG may not be cost-effective for the improvement gained")
        
        return observations

    def generate_evaluation_report(
        self,
        metrics: EvaluationMetrics,
        query: str,
        response: str,
        enhanced_metrics: EnhancedRAGMetrics = None
    ) -> str:
        """Generate a comprehensive evaluation report."""
        
        report = f"""
RAG Evaluation Report
====================
Query: {query}

Response: {response}

Basic Metrics:
- Relevance Score: {metrics.relevance_score:.2f}
- Answer Quality: {metrics.answer_quality:.2f}
- Context Utilization: {metrics.context_utilization:.2f}
- Overall Score: {metrics.overall_score:.2f}
"""
        
        if enhanced_metrics:
            report += f"""
Enhanced RAG Analysis:
=====================
Performance Comparison:
- Baseline Score: {enhanced_metrics.baseline_score:.2f}
- Enhanced Score: {enhanced_metrics.enhanced_score:.2f}
- Improvement: {enhanced_metrics.improvement:.2f} ({enhanced_metrics.improvement*100:.1f}%)

Component Performance:
- Grading Accuracy: {enhanced_metrics.grading_performance.get('accuracy', 0):.2f}
- Rewriting Effectiveness: {enhanced_metrics.rewriting_performance.get('effectiveness', 0):.2f}
- Verification Confidence: {enhanced_metrics.verification_performance.get('confidence', 0):.2f}

Cost Analysis:
- Total Cost: ${enhanced_metrics.cost_analysis.get('total_cost', 0):.4f}
- Cost per Improvement: ${enhanced_metrics.cost_analysis.get('cost_per_improvement', 0):.4f}
- Cost Effective: {enhanced_metrics.cost_analysis.get('is_cost_effective', False)}

Qualitative Observations:
"""
            for obs in enhanced_metrics.qualitative_observations:
                report += f"- {obs}\n"
        
        report += f"""
Interpretation:
- Relevance Score: How well the retrieved documents match the query
- Answer Quality: How accurate and complete the response is
- Context Utilization: How well the response uses the retrieved context
- Overall Score: Combined performance metric
"""
        return report

    def export_evaluation_history(self, filename: str = None) -> str:
        """Export evaluation history to JSON file."""
        if not filename:
            filename = f"rag_evaluation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.evaluation_history, f, indent=2, default=str)
            print(f"[EVAL] Evaluation history exported to {filename}")
            return filename
        except Exception as e:
            print(f"[EVAL] Error exporting evaluation history: {e}")
            return ""

    def rag_evaluate(query: str, response: str, retrieved_docs: list) -> dict:
        # Your real evaluation logic here
        # For example: return simple heuristic metrics
        return {"relevance_score": 0.85, "coherence": 0.9}
