# Enhanced RAG Agent with Document Grading, Query Rewriting, and Verification

This implementation provides an advanced Retrieval-Augmented Generation (RAG) agent that goes beyond basic document retrieval to include intelligent document grading, query rewriting, and response verification capabilities.

## 🚀 Key Features

### Enhanced RAG Capabilities

1. **Document Grading**: Automatically evaluates the relevance of retrieved documents to the user query
2. **Query Rewriting**: Intelligently rewrites queries when initial retrieval quality is poor
3. **Response Verification**: Verifies that generated responses are grounded in source documents
4. **Cost Tracking**: Monitors token usage and estimated costs for each component
5. **Performance Comparison**: Compares enhanced RAG against baseline RAG performance

### Evaluation Framework

- **Quantitative Metrics**: Accuracy, similarity scores, improvement measurements
- **Qualitative Observations**: Automated analysis of system behavior and performance
- **Cost Analysis**: Token usage breakdown and cost-effectiveness evaluation
- **Comprehensive Reporting**: Detailed JSON and human-readable reports

## 📋 System Architecture

```
User Query
    ↓
Initial Retrieval
    ↓
Document Grading ← LLM evaluates relevance
    ↓
Quality Assessment
    ↓
Query Rewriting? ← If retrieval quality is weak/poor
    ↓
Re-retrieval (if needed)
    ↓
Response Generation ← Using only relevant documents
    ↓
Response Verification ← LLM checks grounding
    ↓
Enhanced Evaluation ← Compare vs baseline
    ↓
Final Response + Metrics
```

## 🛠️ Installation and Setup

### Prerequisites

1. Python 3.8+
2. Required API keys:
   - `GEMINI_API_KEY` (for embeddings)
   - `GROQ_API_KEY` (for text generation)

### Environment Setup

```bash
# Create .env file with your API keys
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
echo "GROQ_API_KEY=your_groq_api_key_here" >> .env

# Install dependencies
pip install -r requirements.txt
```

### Document Preparation

Place your documents in the `documents/` directory. The system expects `.txt` files:

```
documents/
├── diff.txt
├── hallu.txt
├── reHack.txt
└── sample_data.txt
```

## 🔧 Usage

### Quick Test

Run the enhanced RAG test script:

```bash
python test_enhanced_rag.py
```

This will:
- Test the enhanced RAG workflow with sample queries
- Generate performance reports
- Compare against baseline RAG
- Export detailed results

### Comprehensive Evaluation

Run the full evaluation suite:

```bash
python run_enhanced_rag_evaluation.py
```

This will:
- Load questions from `evaluation_questions.json`
- Run both baseline and enhanced RAG on all questions
- Generate comprehensive reports with quantitative and qualitative analysis
- Export results in JSON and human-readable formats

### Custom Evaluation

Create your own evaluation questions in JSON format:

```json
{
  "evaluation_dataset": {
    "questions": [
      {
        "id": 1,
        "question": "Your question here",
        "ground_truth": "Expected answer",
        "category": "Category",
        "difficulty": "easy/medium/hard"
      }
    ]
  }
}
```

## 📊 Understanding the Results

### Quantitative Metrics

- **Improvement Score**: How much the enhanced RAG improves over baseline (-1.0 to 1.0)
- **Cost Effectiveness**: Whether the improvement justifies the additional cost
- **Component Accuracy**: Performance of grading, rewriting, and verification

### Qualitative Observations

The system automatically generates observations such as:
- "Enhanced RAG shows significant improvement over baseline"
- "Document grading performed well with high accuracy"
- "Query rewriting was effective and preserved semantic meaning"
- "Verification detected potential hallucination in response"

### Report Files

After evaluation, you'll get:

1. **JSON Report**: `enhanced_rag_evaluation_report_YYYYMMDD_HHMMSS.json`
   - Detailed metrics and raw data
   - Machine-readable format for further analysis

2. **Summary Report**: `enhanced_rag_evaluation_summary_YYYYMMDD_HHMMSS.txt`
   - Human-readable executive summary
   - Key findings and recommendations

## 🔍 Component Details

### Document Grading

The grading system evaluates each retrieved document using an LLM with the prompt:
- Assesses relevance to the user question
- Provides binary classification (relevant/not_relevant)
- Includes confidence score and reasoning

### Query Rewriting

Triggered when retrieval quality is assessed as "weak" or "poor":
- Analyzes why initial retrieval failed
- Rewrites query to be more specific and detailed
- Maintains original intent while improving retrieval likelihood

### Response Verification

Checks if the generated response is grounded in source documents:
- Verifies factual accuracy against provided context
- Detects potential hallucinations
- Provides confidence score and identifies issues

### Cost Tracking

Monitors token usage across all components:
- Grading tokens
- Rewriting tokens  
- Verification tokens
- Generation tokens
- Estimated costs in USD

## 📈 Performance Analysis

### When Enhanced RAG Helps

- **Complex queries** that benefit from query refinement
- **Ambiguous questions** where document grading improves precision
- **Scenarios requiring high accuracy** where verification prevents hallucinations

### When It May Not Help

- **Simple, well-defined queries** where basic RAG already works well
- **Cost-sensitive applications** where the improvement doesn't justify additional expense
- **Real-time applications** where latency is critical

## 🎯 Customization

### Adjusting Grading Criteria

Modify the grading prompt in `src/react_agent/rag.py`:

```python
grading_prompt = f"""Your custom grading instructions here...
Document: {doc['content']}
Question: {query}
..."""
```

### Tuning Rewriting Logic

Adjust when rewriting is triggered in `assess_retrieval_quality()`:

```python
if relevance_ratio >= 0.7 and avg_score >= 0.7:
    return "good"
elif relevance_ratio >= 0.3 and avg_score >= 0.5:  # Adjust these thresholds
    return "weak"
else:
    return "poor"
```

### Custom Verification

Modify verification criteria in `verify_response()`:

```python
verification_prompt = f"""Your custom verification instructions...
Source: {source_context}
Response: {response}
..."""
```

## 🚨 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure `GEMINI_API_KEY` and `GROQ_API_KEY` are set in `.env`
   - Check API key validity and quotas

2. **Document Loading Issues**
   - Verify documents are in `documents/` directory
   - Check file encoding (UTF-8 recommended)
   - Ensure files have `.txt` extension

3. **Memory Issues**
   - Large document collections may require more RAM
   - Consider chunking documents or reducing batch sizes

4. **Performance Issues**
   - Enhanced RAG uses more API calls than baseline
   - Consider caching or reducing evaluation frequency for cost optimization

### Debug Mode

Enable detailed logging by setting environment variable:

```bash
export RAG_DEBUG=1
python run_enhanced_rag_evaluation.py
```

## 📝 Example Output

```
Enhanced RAG Agent Comprehensive Evaluation Report
=================================================

EXECUTIVE SUMMARY:
-----------------
Total Questions Evaluated: 10
Successful Evaluations: 10 (100.0%)
Average Performance Improvement: 15.2%
Cost Effectiveness Rate: 80.0%

QUANTITATIVE RESULTS:
--------------------
Performance Metrics:
  • Average Improvement: 0.152 (15.2%)
  • Questions with Positive Improvement: 8/10
  • Cost Effective Evaluations: 8/10

COMPONENT PERFORMANCE:
---------------------
Document Grading:
  • Average Accuracy: 0.847
  • High Accuracy (>0.8): 7/10

Query Rewriting:
  • Queries Rewritten: 3/10 (30.0%)
  • Average Effectiveness: 0.782

Response Verification:
  • Average Confidence: 0.891
  • High Confidence (>0.8): 9/10

RECOMMENDATIONS:
---------------
  • Enhanced RAG shows significant improvement and should be deployed
  • Cost efficiency is good, enhanced features provide value
```

## 🤝 Contributing

To extend the enhanced RAG system:

1. Add new evaluation metrics in `src/react_agent/evaluator.py`
2. Implement additional grading strategies in `src/react_agent/rag.py`
3. Create new test scenarios in the evaluation scripts
4. Update documentation and examples

## 📄 License

This enhanced RAG implementation is part of the react-agent project. See the main project LICENSE file for details.

---

For questions or issues, please refer to the main project documentation or create an issue in the repository. 