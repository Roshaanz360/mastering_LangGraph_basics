from dataclasses import dataclass, field
from typing import Sequence, List, Dict, Any, Optional
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated

@dataclass
class InputState:
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)

@dataclass
class State(InputState):
    is_last_step: IsLastStep = field(default=False)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_metrics: Dict[str, float] = field(default_factory=dict)
    current_query: str = field(default="")
    rag_response: str = field(default="")
    
    # Enhanced RAG fields
    original_query: str = field(default="")
    rewritten_query: Optional[str] = field(default=None)
    document_grades: List[Dict[str, Any]] = field(default_factory=list)
    relevant_documents: List[Dict[str, Any]] = field(default_factory=list)
    verification_result: Dict[str, Any] = field(default_factory=dict)
    retrieval_quality: str = field(default="")  # "good", "weak", "poor"
    needs_rewrite: bool = field(default=False)
    cost_tracking: Dict[str, Any] = field(default_factory=dict)
    performance_comparison: Dict[str, Any] = field(default_factory=dict)
    
    # Query relevance fields
    is_rag_relevant: bool = field(default=False)
    relevance_reason: str = field(default="")
