# RAG Text Search - LLM Zoomcamp FAQ Chatbot

This project implements a **Retrieval-Augmented Generation (RAG)** system for answering questions about the LLM Zoomcamp course. The system combines text search capabilities with Large Language Model (LLM) generation to provide accurate, context-aware responses to user questions.

## Overview

The RAG Text Search system consists of four main components:

1. **Document Indexing** (`minsearch.py`) - Creates searchable indexes using ML-based text similarity
2. **Search Utilities** (`searchutil.py`) - Provides optimized search functions with field boosting
3. **LLM Integration** (`llm.py`) - Handles prompt creation and OpenAI API interactions
4. **Main Chatbot** (`llm-zoomcamp-faq-chatbot.py`) - Orchestrates the entire RAG pipeline

## How It Works

### 1. Document Processing
The system loads FAQ documents from `../documents.json` containing course-related questions and answers. Each document includes:
- `question`: The original question
- `text`: The detailed answer
- `section`: The topic category
- `course`: The course identifier

### 2. Text Search with MinSearch

https://github.com/alexeygrigorev/minsearch

The core search functionality is powered by **MinSearch**, a lightweight text search engine that implements several machine learning approaches:

#### TF-IDF Vectorization
- **Term Frequency-Inverse Document Frequency (TF-IDF)** converts text into numerical vectors
- Each document field (`question`, `text`, `section`) is transformed into a high-dimensional vector space
- TF-IDF captures the importance of words by considering both their frequency in a document and their rarity across the corpus

#### Cosine Similarity
- **Cosine similarity** measures the angle between query and document vectors
- This metric is ideal for text comparison as it's normalized and focuses on content similarity rather than document length
- Values range from 0 (completely dissimilar) to 1 (identical content)

#### Field Boosting Strategy
The search system applies different importance weights to different fields:
```python
boost_dict = {'question': 3.0, 'section': 0.5}
```
- **Questions** get 3x boost (most important for matching user intent)
- **Sections** get 0.5x boost (less important, used for categorization)
- **Text** gets default 1.0x weight (standard importance)

#### Filtering and Ranking
- **Keyword filtering** ensures results match specific criteria (e.g., course type)
- **Score combination** aggregates similarity scores across all text fields
- **Top-k selection** uses `np.argpartition` for efficient retrieval of best matches

### 3. RAG Pipeline Execution

When a user asks a question, the system:

1. **Retrieves** relevant documents using MinSearch (typically top 5 results)
2. **Augments** the query with retrieved context documents
3. **Generates** a response using GPT-4o through OpenAI API

The prompt template ensures the LLM:
- Acts as a course teaching assistant
- Uses only information from the retrieved context
- Admits when there's insufficient information to answer

## Usage

### Prerequisites
- Python 3.8+
- OpenAI API key (set as environment variable)
- Required packages: `pandas`, `scikit-learn`, `numpy`, `openai`

### Running the Chatbot
```bash
cd ~/LLMZoomcamp/llm-zoomcamp/rag-text-search
python llm-zoomcamp-faq-chatbot.py
```

The chatbot will:
1. Load and index all FAQ documents
2. Start an interactive session
3. Process questions and return contextual answers

### Example Interaction
```
Ask question: When does the course start?
Answer: The course will start on 15th Jan 2024 at 17h00, beginning with the first "Office Hours" live session.
```
