import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

class SingleDocumentRetrieval:
    """
    Handles vector store creation and document retrieval for single document chat.
    """
    def __init__(self, session_id: str):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.session_id = session_id
            self.model_loader = ModelLoader()
            self.embeddings = self.model_loader.load_embeddings()
            self.llm = self.model_loader.load_llm()
            
            # Initialize compressor for contextual compression
            self.compressor = LLMChainExtractor.from_llm(self.llm)
            
            self.log.info("SingleDocumentRetrieval initialized", session_id=session_id)
            
        except Exception as e:
            self.log.error(f"Error initializing SingleDocumentRetrieval: {e}")
            raise DocumentPortalException("Error initializing SingleDocumentRetrieval", e) from e
    
    def create_vector_store(self, documents: List[Document], store_name: str = None) -> FAISS:
        """Create FAISS vector store from documents"""
        try:
            if not documents:
                raise DocumentPortalException("No documents provided for vector store creation")
            
            # Create vector store
            vector_store = FAISS.from_documents(documents, self.embeddings)
            
            # Save vector store if name provided
            if store_name:
                store_path = os.path.join("data", "vector_stores", self.session_id, store_name)
                os.makedirs(os.path.dirname(store_path), exist_ok=True)
                vector_store.save_local(store_path)
                self.log.info("Vector store saved", store_path=store_path)
            
            self.log.info("Vector store created successfully", 
                         num_docs=len(documents), 
                         session_id=self.session_id)
            return vector_store
            
        except Exception as e:
            self.log.error(f"Error creating vector store: {e}")
            raise DocumentPortalException("Error creating vector store", e) from e
    
    def load_vector_store(self, store_path: str) -> FAISS:
        """Load existing vector store"""
        try:
            vector_store = FAISS.load_local(store_path, self.embeddings)
            self.log.info("Vector store loaded successfully", store_path=store_path)
            return vector_store
            
        except Exception as e:
            self.log.error(f"Error loading vector store: {e}")
            raise DocumentPortalException("Error loading vector store", e) from e
    
    def create_retriever(self, vector_store: FAISS, use_compression: bool = True, top_k: int = 5):
        """Create retriever with optional contextual compression"""
        try:
            if use_compression:
                # Create base retriever
                base_retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": top_k}
                )
                
                # Create contextual compression retriever
                retriever = ContextualCompressionRetriever(
                    base_compressor=self.compressor,
                    base_retriever=base_retriever
                )
                
                self.log.info("Contextual compression retriever created", top_k=top_k)
            else:
                # Create simple retriever
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": top_k}
                )
                
                self.log.info("Simple retriever created", top_k=top_k)
            
            return retriever
            
        except Exception as e:
            self.log.error(f"Error creating retriever: {e}")
            raise DocumentPortalException("Error creating retriever", e) from e
    
    def retrieve_documents(self, retriever, query: str) -> List[Document]:
        """Retrieve relevant documents for a query"""
        try:
            documents = retriever.get_relevant_documents(query)
            
            self.log.info("Documents retrieved successfully", 
                         query=query, 
                         num_docs=len(documents),
                         session_id=self.session_id)
            
            return documents
            
        except Exception as e:
            self.log.error(f"Error retrieving documents: {e}")
            raise DocumentPortalException("Error retrieving documents", e) from e
    
    def get_relevant_context(self, retriever, query: str, max_chars: int = 4000) -> str:
        """Get relevant context from retrieved documents"""
        try:
            documents = self.retrieve_documents(retriever, query)
            
            # Combine document content
            context_parts = []
            total_chars = 0
            
            for doc in documents:
                content = doc.page_content
                if total_chars + len(content) <= max_chars:
                    context_parts.append(content)
                    total_chars += len(content)
                else:
                    # Add partial content if it fits
                    remaining_chars = max_chars - total_chars
                    if remaining_chars > 100:  # Only add if we have meaningful content
                        context_parts.append(content[:remaining_chars])
                    break
            
            context = "\n\n".join(context_parts)
            
            self.log.info("Context extracted successfully", 
                         query=query, 
                         context_length=len(context),
                         num_docs_used=len(context_parts))
            
            return context
            
        except Exception as e:
            self.log.error(f"Error getting relevant context: {e}")
            raise DocumentPortalException("Error getting relevant context", e) from e
