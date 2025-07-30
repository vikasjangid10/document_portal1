# 📚 Document Portal - AI Analysis

A comprehensive AI-powered document analysis and chat system built with Python, LangChain, and Streamlit. This project provides multiple capabilities for working with PDF documents including analysis, chat, and evaluation.

## 🚀 Features

### 📊 Document Analysis
- **Metadata Extraction**: Automatically extract title, author, publisher, dates, and more
- **Content Summarization**: Generate comprehensive summaries of documents
- **Sentiment Analysis**: Analyze the tone and sentiment of document content
- **Structured Output**: Get results in structured JSON format

### 💬 Single Document Chat
- **AI-Powered Chat**: Chat with individual documents using natural language
- **Contextual Retrieval**: Intelligent document retrieval based on user queries
- **Real-time Evaluation**: Get metrics on chat quality and relevance
- **Session Management**: Organized session-based document processing

### 📚 Multi-Document Chat
- **Cross-Document Analysis**: Chat across multiple documents simultaneously
- **Source Tracking**: Track which documents contribute to responses
- **Ensemble Retrieval**: Use multiple retrieval strategies for better results
- **MMR (Maximum Marginal Relevance)**: Ensure diverse and relevant document retrieval
- **Advanced Evaluation**: Multi-dimensional evaluation including source diversity and cross-document coherence

### 🔍 Document Comparison (Planned)
- **Content Similarity Analysis**: Compare documents for similarities
- **Key Differences Identification**: Highlight differences between documents
- **Topic Overlap Analysis**: Analyze overlapping topics and themes
- **Comparative Summaries**: Generate comparative summaries

## 🏗️ Architecture

### Core Components

1. **Document Analyzer** (`src/document_analyzer/`)
   - `data_ingestion.py`: PDF processing and text extraction
   - `data_analysis.py`: AI-powered document analysis and metadata extraction

2. **Single Document Chat** (`src/single_document_chat/`)
   - `data_ingestion.py`: PDF processing and text chunking
   - `retrieval.py`: Vector store creation and document retrieval
   - `evaluation.py`: Chat quality evaluation and metrics

3. **Multi-Document Chat** (`src/multi_document_chat/`)
   - `data_ingestion.py`: Multi-PDF processing with source tracking
   - `retrieval.py`: Ensemble retrieval and source-separated stores
   - `mmr.py`: Maximum Marginal Relevance for diverse retrieval
   - `evaluation.py`: Multi-dimensional evaluation metrics

4. **Infrastructure**
   - **Logging**: Structured JSON logging with `structlog`
   - **Exception Handling**: Custom exception classes
   - **Configuration**: YAML-based configuration management
   - **Model Loading**: Dynamic LLM/embedding model loading

### Key Technologies

- **LangChain**: Core AI framework for document processing and chat
- **Streamlit**: Modern web interface
- **FAISS**: Vector database for document retrieval
- **PyMuPDF**: PDF text extraction
- **Groq/Google Gemini**: LLM providers
- **Pydantic**: Data validation and serialization

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.10 or higher
- Conda or virtual environment
- API keys for LLM providers

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/document_portal.git
cd document_portal
```

### 2. Create Environment

```bash
# Create conda environment
conda create -p ./env python=3.10 -y
conda activate ./env

# Or create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Setup

Create a `.env` file in the project root:

```env
# Required API Keys
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Optional: Set LLM provider (default: groq)
LLM_PROVIDER=groq

# Optional: Set data storage path
DATA_STORAGE_PATH=./data
```

### 5. Get API Keys

#### Groq API Key (Free)
- Visit [Groq Console](https://console.groq.com/keys)
- Create an account and get your API key
- [Groq Documentation](https://console.groq.com/docs/overview)

#### Google Gemini API Key (15 Days Free)
- Visit [Google AI Studio](https://aistudio.google.com/apikey)
- Create an account and get your API key
- [Gemini Documentation](https://ai.google.dev/gemini-api/docs/models)

## 🚀 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Features

#### 1. Document Analysis
1. Navigate to "Document Analysis" in the sidebar
2. Upload a PDF document
3. View extracted metadata, summary, and sentiment analysis

#### 2. Single Document Chat
1. Navigate to "Single Document Chat" in the sidebar
2. Upload a PDF document
3. Start chatting with your document using natural language
4. View evaluation metrics for each interaction

#### 3. Multi-Document Chat
1. Navigate to "Multi-Document Chat" in the sidebar
2. Upload multiple PDF documents
3. Chat across all documents simultaneously
4. View advanced evaluation metrics including source diversity

## 📊 Evaluation Metrics

### Single Document Chat
- **Context Relevance**: How relevant retrieved context is to the query
- **Response Quality**: Quality of AI responses based on accuracy and helpfulness
- **Response Time**: Average time for AI responses
- **Success Rate**: Percentage of successful queries

### Multi-Document Chat
- **Context Relevance**: Relevance of retrieved context across documents
- **Response Quality**: Quality of cross-document synthesis
- **Source Diversity**: Diversity of sources used in responses
- **Cross-Document Coherence**: How well information is synthesized across documents
- **Average Sources per Query**: Average number of sources used per query

## 🔧 Configuration

### Model Configuration (`config/config.yaml`)

```yaml
faiss_db:
  collection_name: "document_portal"

embedding_model:
  provider: "google"
  model_name: "models/text-embedding-004"

retriever:
  top_k: 10

llm:
  groq:
    provider: "groq"
    model_name: "deepseek-r1-distill-llama-70b"
    temperature: 0
    max_output_tokens: 2048

  google:
    provider: "google"
    model_name: "gemini-2.0-flash"
    temperature: 0
    max_output_tokens: 2048
```

### Supported LLM Providers

- **Groq** (Recommended - Free tier available)
- **Google Gemini** (15-day free access)
- **OpenAI** (Paid)
- **Claude** (Paid)
- **Hugging Face** (Free)
- **Ollama** (Local setup)

## 📁 Project Structure

```
document_portal/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── config/
│   └── config.yaml                # Configuration file
├── data/                          # Data storage
│   ├── document_analysis/         # Document analysis sessions
│   ├── single_document_chat/      # Single document chat sessions
│   ├── multi_document_chat/       # Multi-document chat sessions
│   ├── vector_stores/             # FAISS vector stores
│   └── evaluations/               # Evaluation data
├── src/
│   ├── document_analyzer/         # Document analysis functionality
│   ├── single_document_chat/      # Single document chat functionality
│   ├── multi_document_chat/       # Multi-document chat functionality
│   └── document_compare/          # Document comparison (planned)
├── utils/                         # Utility functions
├── logger/                        # Logging configuration
├── exception/                     # Custom exceptions
├── model/                         # Data models
└── prompt/                        # Prompt templates
```

## 🧪 Testing

### Running Tests

```bash
# Test document analysis
python -c "from src.document_analyzer.data_analysis import DocumentAnalyzer; print('Document analyzer test passed')"

# Test single document chat
python -c "from src.single_document_chat.retrieval import SingleDocumentRetrieval; print('Single document chat test passed')"

# Test multi-document chat
python -c "from src.multi_document_chat.mmr import MMRRetrieval; print('Multi-document chat test passed')"
```

## 📈 Performance

### Optimization Features

- **Session-based Processing**: Efficient document organization
- **Vector Store Caching**: Cached vector stores for faster retrieval
- **Contextual Compression**: Intelligent document compression
- **MMR Retrieval**: Diverse and relevant document retrieval
- **Evaluation Metrics**: Real-time performance monitoring

### Scalability

- **Modular Architecture**: Easy to extend and modify
- **Multiple LLM Support**: Switch between different providers
- **Configurable Parameters**: Adjustable retrieval and evaluation parameters
- **Session Management**: Organized data storage and retrieval

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: For the excellent AI framework
- **Streamlit**: For the beautiful web interface
- **Groq**: For fast and free LLM access
- **Google**: For Gemini AI capabilities
- **FAISS**: For efficient vector similarity search

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/document_portal/issues) page
2. Create a new issue with detailed information
3. Include your environment details and error logs

## 🔮 Future Enhancements

- [ ] Document comparison functionality
- [ ] Support for more document formats (DOCX, TXT, etc.)
- [ ] Advanced visualization of document relationships
- [ ] Integration with cloud storage providers
- [ ] Real-time collaboration features
- [ ] Advanced analytics dashboard
- [ ] Mobile application
- [ ] API endpoints for programmatic access

---

**Happy Document Analysis! 📚✨**


