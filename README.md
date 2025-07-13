# WhatsApp Chat RAG System v2

A  Retrieval-Augmented Generation (RAG) system for analyzing WhatsApp chat data with support for Tamil-English (Tanglish) conversations. This system combines multiple search methodologies with an LLM assistant to provide intelligent chat analysis and insights.

## Features

### Search Capabilities
- **Semantic Search**: Find messages with similar meaning using sentence transformers
- **Keyword Search**: Fast exact and partial keyword matching with fuzzy similarity
- **TF-IDF Search**: Term frequency-based document retrieval
- **Emotion Similarity**: Find emotionally similar messages using emotion embeddings
- **Combined Search**: Hybrid approach combining all methods with parallel processing

### Comprehensive Analytics
- **Chat Statistics**: Message counts, user activity, date ranges
- **Sentiment Analysis**: Emotion classification and sentiment scoring
- **User Behavior**: Response times, activity patterns, dominance metrics
- **Content Analysis**: Word frequency, emoji usage, message patterns

### LLM Assistant Integration
- **LLM Integration**: Works with Ollama models (DeepSeek-R1 optimized)
- **Function Calling**: Intelligent function selection based on query type
- **Natural Language**: Ask questions in plain English about your chat data

### Performance Optimizations
- **Caching System**: Persistent storage of computed embeddings and indexes
- **Batch Processing**: Efficient handling of large datasets
- **Parallel Search**: Multi-threaded search execution

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/whatsapp-chat-rag.git
cd whatsapp-chat-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama and download DeepSeek model:
```bash
# Install Ollama (https://ollama.ai/)
ollama pull deepseek-r1:8b
```

4. Download NLTK data:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

## Quick Start

### 1. Prepare Your Chat Data
Export your WhatsApp chat as a `.txt` file and place it in your project directory.

### 2. Initialize the System
```python
from utils.parser import parse_chat_log
from utils.message_reader import reader
from analysis.rag import TanglishChatRAG
from chat.assistant import ChatRAGAssistant

# Load and parse chat data
text = reader("path/to/your/chat.txt")
df = parse_chat_log(text)

# Initialize RAG system
chat_rag = TanglishChatRAG(df=df)

# Create AI assistant
assistant = ChatRAGAssistant(chat_rag, "deepseek-r1:8b")
```

### 3. Query Your Chat
```python
# Ask natural language questions
answer = assistant.query("Were there any fights in the chat?")
print(answer)

# Find similar messages
answer = assistant.query("Find messages similar to 'good morning everyone'")
print(answer)

# Get user statistics
answer = assistant.query("Show me statistics about user activity")
print(answer)
```

## Usage Examples

### Interactive Chat Analysis
```python
# Run the interactive assistant
python app.py
```

### Programmatic Usage
```python
# Direct search methods
results = chat_rag.semantic_search_optimized("birthday party", top_k=10)
results = chat_rag.keyword_search_optimized(["happy", "celebration"], top_k=15)
results = chat_rag.combined_search_optimized("fight argument", top_k=20)

# Get user messages
user_messages = chat_rag.get_user_messages("John", limit=5)

# Get recent activity
recent = chat_rag.get_recent_messages(hours=24, limit=10)

# Get comprehensive stats
stats = chat_rag.get_stats()
```

## Architecture

### Core Components
- **TanglishChatRAG**: Main RAG system with optimized search methods
- **ChatRAGAssistant**: AI assistant with function calling capabilities
- **Message Parser**: WhatsApp chat format parser
- **Caching System**: Persistent storage for embeddings and indexes

### Search Methods
1. **Semantic Search**: Uses `sentence-transformers` for meaning-based retrieval
2. **Keyword Search**: Optimized exact and fuzzy matching with word indexes
3. **TF-IDF**: Statistical term frequency analysis
4. **Emotion Similarity**: Emotion-based message clustering
5. **Combined Search**: Parallel execution with weighted scoring

### LLM Integration
- **Function Calling**: Automatic function selection based on query intent
- **Context Awareness**: Maintains conversation context for better responses
- **Multi-language Support**: Handles Tamil-English code-switching

### Optimizations
- Precomputed embeddings with persistent caching
- Vectorized operations using NumPy
- Batch processing for large datasets
- Memory-efficient data structures

## Configuration

### Model Selection
```python
# Use different models
assistant = ChatRAGAssistant(chat_rag, "llama2:7b")
assistant = ChatRAGAssistant(chat_rag, "mistral:7b")
```

### Search Parameters
```python
# Adjust search parameters
results = chat_rag.semantic_search_optimized(
    query="birthday",
    top_k=20  # Number of results
)
```

### Caching
```python
# Clear cache for fresh rebuild
chat_rag.clear_cache()

# Custom cache directory
chat_rag = TanglishChatRAG(df=df, cache_dir="./custom_cache")
```

## Dependencies

### Core Libraries
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning utilities
- `sentence-transformers` - Semantic embeddings
- `torch` - Deep learning framework
- `nltk` - Natural language processing

### LLM Integration
- `ollama` - Local LLM inference
- `emoji` - Emoji processing

### Optional
- `matplotlib` - Data visualization
- `seaborn` - Statistical plots
- `wordcloud` - Word cloud generation


**Note**: This system is designed for personal use with your own chat data. Ensure you have proper permissions before analyzing any chat conversations.

**Made with ❤️ NiceGuy**
