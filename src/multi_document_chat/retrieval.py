import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers import EnsembleRetriever
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from datetime import datetime

class MultiDocumentRetrieval:
    """
    Handles vector store creation and document retrieval for multi-document chat.
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
            
            self.log.info("MultiDocumentRetrieval initialized", session_id=session_id)
            
        except Exception as e:
            self.log.error(f"Error initializing MultiDocumentRetrieval: {e}")
            raise DocumentPortalException("Error initializing MultiDocumentRetrieval", e) from e
    
    def create_unified_vector_store(self, documents: List[Document], store_name: str = None) -> FAISS:
        """Create unified FAISS vector store from all documents"""
        try:
            if not documents:
                raise DocumentPortalException("No documents provided for vector store creation")
            
            # Create unified vector store
            vector_store = FAISS.from_documents(documents, self.embeddings)
            
            # Save vector store if name provided
            if store_name:
                store_path = os.path.join("data", "vector_stores", self.session_id, store_name)
                os.makedirs(os.path.dirname(store_path), exist_ok=True)
                vector_store.save_local(store_path)
                self.log.info("Unified vector store saved", store_path=store_path)
            
            self.log.info("Unified vector store created successfully", 
                         num_docs=len(documents), 
                         session_id=self.session_id)
            return vector_store
            
        except Exception as e:
            self.log.error(f"Error creating unified vector store: {e}")
            raise DocumentPortalException("Error creating unified vector store", e) from e
    
    def create_source_separated_stores(self, documents: List[Document]) -> Dict[str, FAISS]:
        """Create separate vector stores for each source document"""
        try:
            # Group documents by source
            source_groups = {}
            for doc in documents:
                source = doc.metadata.get("source_file", "unknown")
                if source not in source_groups:
                    source_groups[source] = []
                source_groups[source].append(doc)
            
            # Create vector store for each source
            source_stores = {}
            for source, docs in source_groups.items():
                vector_store = FAISS.from_documents(docs, self.embeddings)
                source_stores[source] = vector_store
                
                # Save individual source store
                store_path = os.path.join("data", "vector_stores", self.session_id, f"source_{source}")
                os.makedirs(os.path.dirname(store_path), exist_ok=True)
                vector_store.save_local(store_path)
                
                self.log.info("Source vector store created", 
                             source=source, 
                             num_docs=len(docs))
            
            return source_stores
            
        except Exception as e:
            self.log.error(f"Error creating source-separated stores: {e}")
            raise DocumentPortalException("Error creating source-separated stores", e) from e
    
    def create_ensemble_retriever(self, source_stores: Dict[str, FAISS], 
                                top_k: int = 5) -> EnsembleRetriever:
        """Create ensemble retriever from multiple source stores"""
        try:
            retrievers = []
            
            for source, store in source_stores.items():
                retriever = store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": top_k}
                )
                retrievers.append(retriever)
            
            # Create ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=retrievers,
                weights=[1.0] * len(retrievers)  # Equal weights for all sources
            )
            
            self.log.info("Ensemble retriever created", 
                         num_sources=len(source_stores),
                         top_k=top_k)
            
            return ensemble_retriever
            
        except Exception as e:
            self.log.error(f"Error creating ensemble retriever: {e}")
            raise DocumentPortalException("Error creating ensemble retriever", e) from e
    
    def create_compression_retriever(self, vector_store: FAISS, top_k: int = 5):
        """Create retriever with contextual compression"""
        try:
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
            
            self.log.info("Compression retriever created", top_k=top_k)
            return retriever
            
        except Exception as e:
            self.log.error(f"Error creating compression retriever: {e}")
            raise DocumentPortalException("Error creating compression retriever", e) from e
    
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
    
    def get_relevant_context_with_sources(self, retriever, query: str, 
                                       max_chars: int = 4000) -> Dict[str, Any]:
        """Get relevant context with source information"""
        try:
            documents = self.retrieve_documents(retriever, query)
            
            # Group by source and combine content
            source_groups = {}
            total_chars = 0
            
            for doc in documents:
                source = doc.metadata.get("source_file", "unknown")
                content = doc.page_content
                
                if source not in source_groups:
                    source_groups[source] = []
                
                # Check if adding this content would exceed max_chars
                if total_chars + len(content) <= max_chars:
                    source_groups[source].append(content)
                    total_chars += len(content)
                else:
                    # Add partial content if it fits
                    remaining_chars = max_chars - total_chars
                    if remaining_chars > 100:  # Only add if we have meaningful content
                        source_groups[source].append(content[:remaining_chars])
                    break
            
            # Combine content by source
            context_by_source = {}
            for source, contents in source_groups.items():
                context_by_source[source] = "\n\n".join(contents)
            
            # Combine all context
            all_context = "\n\n".join(context_by_source.values())
            
            result = {
                "context": all_context,
                "context_by_source": context_by_source,
                "sources_used": list(context_by_source.keys()),
                "total_chars": len(all_context),
                "num_sources": len(context_by_source)
            }
            
            self.log.info("Context extracted with sources", 
                         query=query, 
                         context_length=len(all_context),
                         num_sources=len(context_by_source),
                         sources_used=list(context_by_source.keys()))
            
            return result
            
        except Exception as e:
            self.log.error(f"Error getting relevant context with sources: {e}")
            raise DocumentPortalException("Error getting relevant context with sources", e) from e
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent to determine retrieval strategy"""
        try:
            intent_prompt = f"""
            Analyze the following query and determine:
            1. What type of information is being sought
            2. Which documents/sources might be most relevant
            3. Whether this is a comparison, synthesis, or specific information request
            
            Query: {query}
            
            Provide your analysis in a structured format.
            """
            
            response = self.llm.invoke(intent_prompt)
            
            # Extract key information from response
            analysis = {
                "query": query,
                "analysis": response.content,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.log.info("Query intent analyzed", query=query)
            return analysis
            
        except Exception as e:
            self.log.error(f"Error analyzing query intent: {e}")
            return {"query": query, "analysis": "Unable to analyze query intent", "error": str(e)}
    
    def get_retrieval_statistics(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about the retrieval process"""
        try:
            if not documents:
                return {"message": "No documents retrieved"}
            
            # Group by source
            source_counts = {}
            total_chunks = len(documents)
            
            for doc in documents:
                source = doc.metadata.get("source_file", "unknown")
                source_counts[source] = source_counts.get(source, 0) + 1
            
            # Calculate statistics
            num_sources = len(source_counts)
            avg_chunks_per_source = total_chunks / num_sources if num_sources > 0 else 0
            
            stats = {
                "total_documents_retrieved": total_chunks,
                "number_of_sources": num_sources,
                "average_chunks_per_source": round(avg_chunks_per_source, 2),
                "source_distribution": source_counts,
                "session_id": self.session_id,
                "retrieval_timestamp": datetime.utcnow().isoformat()
            }
            
            self.log.info("Retrieval statistics generated", 
                         total_docs=total_chunks,
                         num_sources=num_sources)
            
            return stats
            
        except Exception as e:
            self.log.error(f"Error generating retrieval statistics: {e}")
            raise DocumentPortalException("Error generating retrieval statistics", e) from e
