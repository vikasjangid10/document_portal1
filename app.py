import streamlit as st
import os
import time
from datetime import datetime
from typing import List, Dict, Any

# Import our modules
from src.document_analyzer.data_ingestion import DocumentHandler
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.single_document_chat.data_ingestion import SingleDocumentIngestion
from src.single_document_chat.retrieval import SingleDocumentRetrieval
from src.single_document_chat.evaluation import SingleDocumentEvaluation
from src.multi_document_chat.data_ingestion import MultiDocumentIngestion
from src.multi_document_chat.retrieval import MultiDocumentRetrieval
from src.multi_document_chat.mmr import MMRRetrieval
from src.multi_document_chat.evaluation import MultiDocumentEvaluation
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

# Initialize logger
logger = CustomLogger().get_logger(__name__)

def main():
    st.set_page_config(
        page_title="Document Portal - AI Analysis",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">📚 Document Portal - AI Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a feature:",
        ["Document Analysis", "Single Document Chat", "Multi-Document Chat", "Document Comparison"]
    )
    
    if page == "Document Analysis":
        document_analysis_page()
    elif page == "Single Document Chat":
        single_document_chat_page()
    elif page == "Multi-Document Chat":
        multi_document_chat_page()
    elif page == "Document Comparison":
        document_comparison_page()

def document_analysis_page():
    st.header("📊 Document Analysis")
    st.write("Upload a PDF document to extract metadata and generate summaries.")
    
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file:
        try:
            with st.spinner("Processing document..."):
                # Initialize handlers
                handler = DocumentHandler()
                analyzer = DocumentAnalyzer()
                
                # Process document
                file_path = handler.save_pdf(uploaded_file)
                text = handler.read_pdf(file_path)
                
                # Analyze document
                result = analyzer.analyze_document(text)
                
                st.success(f"✅ Successfully processed {uploaded_file.name}")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📋 Document Metadata")
                    st.write(f"**Title:** {result.get('Title', 'N/A')}")
                    st.write(f"**Author:** {result.get('Author', 'N/A')}")
                    st.write(f"**Publisher:** {result.get('Publisher', 'N/A')}")
                    st.write(f"**Language:** {result.get('Language', 'N/A')}")
                    st.write(f"**Page Count:** {result.get('PageCount', 'N/A')}")
                    st.write(f"**Sentiment:** {result.get('SentimentTone', 'N/A')}")
                
                with col2:
                    st.subheader("📅 Document Dates")
                    st.write(f"**Created:** {result.get('DateCreated', 'N/A')}")
                    st.write(f"**Modified:** {result.get('LastModifiedDate', 'N/A')}")
                
                st.subheader("📝 Summary")
                if result.get('Summary'):
                    for i, summary_point in enumerate(result['Summary'], 1):
                        st.write(f"{i}. {summary_point}")
                
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
            logger.error(f"Document analysis error: {e}")

def single_document_chat_page():
    st.header("💬 Single Document Chat")
    st.write("Upload a PDF and chat with it using AI-powered retrieval.")
    
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], key="single_chat")
    
    if uploaded_file:
        try:
            with st.spinner("Processing document for chat..."):
                # Initialize components
                ingestion = SingleDocumentIngestion()
                documents, pdf_path = ingestion.process_pdf(uploaded_file)
                
                # Create vector store and retriever
                retrieval = SingleDocumentRetrieval(ingestion.session_id)
                vector_store = retrieval.create_vector_store(documents)
                retriever = retrieval.create_retriever(vector_store)
                
                # Initialize evaluation
                evaluation = SingleDocumentEvaluation(ingestion.session_id)
                
                st.success(f"✅ Document ready for chat! Session: {ingestion.session_id}")
                
                # Chat interface
                st.subheader("🤖 Chat with your document")
                
                # Initialize chat history
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                
                # Display chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Chat input
                if prompt := st.chat_input("Ask a question about your document..."):
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Generate response
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            start_time = time.time()
                            
                            # Get relevant context
                            context = retrieval.get_relevant_context(retriever, prompt)
                            
                            # Generate response using LLM
                            model_loader = ModelLoader()
                            llm = model_loader.load_llm()
                            
                            response_prompt = f"""
                            You are a helpful AI assistant. Answer the user's question based on the provided context from their document.
                            
                            Context from document:
                            {context}
                            
                            User question: {prompt}
                            
                            Provide a clear, accurate, and helpful response based on the context. If the context doesn't contain relevant information, say so.
                            """
                            
                            response = llm.invoke(response_prompt)
                            response_time = time.time() - start_time
                            
                            st.markdown(response.content)
                            
                            # Add assistant message to chat history
                            st.session_state.messages.append({"role": "assistant", "content": response.content})
                            
                            # Evaluate the interaction
                            retrieved_docs = retrieval.retrieve_documents(retriever, prompt)
                            evaluation.add_chat_interaction(
                                prompt, response.content, context, response_time, retrieved_docs
                            )
                            
                            # Show evaluation metrics
                            with st.expander("📊 Evaluation Metrics"):
                                relevance_score = evaluation.evaluate_context_relevance(prompt, context)
                                quality_score = evaluation.evaluate_response_quality(prompt, response.content, context)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Context Relevance", f"{relevance_score:.2f}")
                                with col2:
                                    st.metric("Response Quality", f"{quality_score:.2f}")
                
                # Show evaluation summary
                if st.button("📊 Show Chat Evaluation Summary"):
                    summary = evaluation.get_evaluation_summary()
                    st.json(summary)
                    
        except Exception as e:
            st.error(f"❌ Error setting up chat: {str(e)}")
            logger.error(f"Single document chat error: {e}")

def multi_document_chat_page():
    st.header("📚 Multi-Document Chat")
    st.write("Upload multiple PDFs and chat across all documents using AI-powered retrieval.")
    
    uploaded_files = st.file_uploader(
        "Upload multiple PDF documents", 
        type=["pdf"], 
        accept_multiple_files=True,
        key="multi_chat"
    )
    
    if uploaded_files:
        try:
            with st.spinner("Processing multiple documents..."):
                # Initialize components
                ingestion = MultiDocumentIngestion()
                documents, pdf_paths = ingestion.process_multiple_pdfs(uploaded_files)
                
                # Create vector stores
                retrieval = MultiDocumentRetrieval(ingestion.session_id)
                source_stores = retrieval.create_source_separated_stores(documents)
                
                # Create ensemble retriever
                ensemble_retriever = retrieval.create_ensemble_retriever(source_stores)
                
                # Initialize MMR retrieval
                mmr_retrieval = MMRRetrieval(ingestion.session_id)
                mmr_retrievers = mmr_retrieval.create_source_aware_mmr_retriever(source_stores)
                
                # Initialize evaluation
                evaluation = MultiDocumentEvaluation(ingestion.session_id)
                
                st.success(f"✅ {len(uploaded_files)} documents ready for chat! Session: {ingestion.session_id}")
                
                # Show document summary
                summary = ingestion.get_document_summary(documents)
                with st.expander("📋 Document Summary"):
                    st.write(f"**Total Documents:** {summary['total_documents']}")
                    st.write(f"**Total Sources:** {summary['total_sources']}")
                    st.write(f"**Average Chunks per Source:** {summary['average_chunks_per_source']}")
                    st.write("**Sources:**")
                    for source in summary['sources']:
                        st.write(f"- {source}")
                
                # Chat interface
                st.subheader("🤖 Chat across all documents")
                
                # Initialize chat history
                if "multi_messages" not in st.session_state:
                    st.session_state.multi_messages = []
                
                # Display chat history
                for message in st.session_state.multi_messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Chat input
                if prompt := st.chat_input("Ask a question across all documents..."):
                    # Add user message to chat history
                    st.session_state.multi_messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Generate response
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing across documents..."):
                            start_time = time.time()
                            
                            # Get relevant context with sources
                            context_result = retrieval.get_relevant_context_with_sources(ensemble_retriever, prompt)
                            context = context_result["context"]
                            sources_used = context_result["sources_used"]
                            
                            # Generate response using LLM
                            model_loader = ModelLoader()
                            llm = model_loader.load_llm()
                            
                            response_prompt = f"""
                            You are a helpful AI assistant. Answer the user's question based on the provided context from multiple documents.
                            
                            Context from documents:
                            {context}
                            
                            Sources used: {', '.join(sources_used)}
                            
                            User question: {prompt}
                            
                            Provide a comprehensive response that synthesizes information from multiple sources. 
                            If information from different sources conflicts, acknowledge this in your response.
                            """
                            
                            response = llm.invoke(response_prompt)
                            response_time = time.time() - start_time
                            
                            st.markdown(response.content)
                            
                            # Add assistant message to chat history
                            st.session_state.multi_messages.append({"role": "assistant", "content": response.content})
                            
                            # Evaluate the interaction
                            retrieved_docs = retrieval.retrieve_documents(ensemble_retriever, prompt)
                            evaluation.add_chat_interaction(
                                prompt, response.content, context, response_time, retrieved_docs, sources_used
                            )
                            
                            # Show evaluation metrics
                            with st.expander("📊 Multi-Document Evaluation Metrics"):
                                relevance_score = evaluation.evaluate_context_relevance(prompt, context)
                                quality_score = evaluation.evaluate_response_quality(prompt, response.content, context)
                                diversity_score = evaluation.evaluate_source_diversity(sources_used, len(summary['sources']))
                                coherence_score = evaluation.evaluate_cross_document_coherence(response.content, context_result.get("context_by_source", {}))
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Context Relevance", f"{relevance_score:.2f}")
                                    st.metric("Response Quality", f"{quality_score:.2f}")
                                with col2:
                                    st.metric("Source Diversity", f"{diversity_score:.2f}")
                                    st.metric("Cross-Document Coherence", f"{coherence_score:.2f}")
                
                # Show evaluation summary
                if st.button("📊 Show Multi-Document Evaluation Summary"):
                    summary = evaluation.get_evaluation_summary()
                    st.json(summary)
                    
        except Exception as e:
            st.error(f"❌ Error setting up multi-document chat: {str(e)}")
            logger.error(f"Multi-document chat error: {e}")

def document_comparison_page():
    st.header("🔍 Document Comparison")
    st.write("Upload two PDFs to compare their content and identify similarities and differences.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        doc1 = st.file_uploader("Upload first PDF", type=["pdf"], key="doc1")
    
    with col2:
        doc2 = st.file_uploader("Upload second PDF", type=["pdf"], key="doc2")
    
    if doc1 and doc2:
        from src.document_analyzer.data_ingestion import DocumentHandler
        from src.document_compare.retrieval import compare_documents
        handler1 = DocumentHandler()
        handler2 = DocumentHandler()
        path1 = handler1.save_pdf(doc1)
        path2 = handler2.save_pdf(doc2)
        text1 = handler1.read_pdf(path1)
        text2 = handler2.read_pdf(path2)
        result = compare_documents(text1, text2)
        st.write(f"Similarity Ratio: {result['similarity']:.2f}")
        st.write("Summary of Differences:")
        for line in result['differences']:
            st.write(line)

if __name__ == "__main__":
    main()
