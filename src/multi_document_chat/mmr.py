import os
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
# MMRRetriever is not available in current LangChain version, using custom implementation
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from datetime import datetime

class MMRRetrieval:
    """
    Implements Maximum Marginal Relevance (MMR) for diverse document retrieval in multi-document chat.
    """
    def __init__(self, session_id: str):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.session_id = session_id
            self.model_loader = ModelLoader()
            self.embeddings = self.model_loader.load_embeddings()
            self.llm = self.model_loader.load_llm()
            
            self.log.info("MMRRetrieval initialized", session_id=session_id)
            
        except Exception as e:
            self.log.error(f"Error initializing MMRRetrieval: {e}")
            raise DocumentPortalException("Error initializing MMRRetrieval", e) from e
    
    def create_mmr_retriever(self, vector_store: FAISS, 
                            fetch_k: int = 20, 
                            lambda_mult: float = 0.5,
                            k: int = 5) -> Dict[str, Any]:
        """
        Create custom MMR retriever for diverse document retrieval.
        
        Args:
            vector_store: FAISS vector store
            fetch_k: Number of documents to fetch before applying MMR
            lambda_mult: Diversity parameter (0.0 = max diversity, 1.0 = max relevance)
            k: Number of documents to return after MMR
        """
        try:
            # Store retriever configuration
            mmr_config = {
                "vector_store": vector_store,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult,
                "k": k,
                "embeddings": self.embeddings
            }
            
            self.log.info("Custom MMR retriever created successfully", 
                         fetch_k=fetch_k,
                         lambda_mult=lambda_mult,
                         k=k,
                         session_id=self.session_id)
            
            return mmr_config
            
        except Exception as e:
            self.log.error(f"Error creating MMR retriever: {e}")
            raise DocumentPortalException("Error creating MMR retriever", e) from e
    
    def create_source_aware_mmr_retriever(self, source_stores: Dict[str, FAISS],
                                        fetch_k: int = 20,
                                        lambda_mult: float = 0.5,
                                        k: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Create MMR retrievers for each source document.
        
        Args:
            source_stores: Dictionary of source-specific vector stores
            fetch_k: Number of documents to fetch before applying MMR
            lambda_mult: Diversity parameter
            k: Number of documents to return after MMR
        """
        try:
            mmr_retrievers = {}
            
            for source, store in source_stores.items():
                mmr_retriever = self.create_mmr_retriever(
                    store, 
                    fetch_k=fetch_k, 
                    lambda_mult=lambda_mult, 
                    k=k
                )
                mmr_retrievers[source] = mmr_retriever
                
                self.log.info("Source-specific MMR retriever created", 
                             source=source,
                             fetch_k=fetch_k,
                             lambda_mult=lambda_mult)
            
            return mmr_retrievers
            
        except Exception as e:
            self.log.error(f"Error creating source-aware MMR retrievers: {e}")
            raise DocumentPortalException("Error creating source-aware MMR retrievers", e) from e
    
    def retrieve_diverse_documents(self, mmr_config: Dict[str, Any], 
                                 query: str) -> List[Document]:
        """Retrieve diverse documents using custom MMR implementation"""
        try:
            vector_store = mmr_config["vector_store"]
            fetch_k = mmr_config["fetch_k"]
            lambda_mult = mmr_config["lambda_mult"]
            k = mmr_config["k"]
            embeddings = mmr_config["embeddings"]
            
            # Get query embedding
            query_embedding = embeddings.embed_query(query)
            
            # Get initial candidates using similarity search
            candidates = vector_store.similarity_search_with_score(
                query, k=fetch_k
            )
            
            # Apply MMR algorithm
            selected_docs = self._apply_mmr_algorithm(
                candidates, query_embedding, lambda_mult, k, embeddings
            )
            
            self.log.info("Diverse documents retrieved using custom MMR", 
                         query=query, 
                         num_docs=len(selected_docs),
                         session_id=self.session_id)
            
            return selected_docs
            
        except Exception as e:
            self.log.error(f"Error retrieving diverse documents: {e}")
            raise DocumentPortalException("Error retrieving diverse documents", e) from e
    
    def retrieve_from_multiple_sources(self, mmr_configs: Dict[str, Dict[str, Any]],
                                     query: str,
                                     max_docs_per_source: int = 3) -> Dict[str, List[Document]]:
        """
        Retrieve diverse documents from multiple sources using MMR.
        
        Args:
            mmr_retrievers: Dictionary of source-specific MMR retrievers
            query: User query
            max_docs_per_source: Maximum documents to retrieve per source
        """
        try:
            source_documents = {}
            
            for source, config in mmr_configs.items():
                # Update config parameters for per-source limit
                config["k"] = max_docs_per_source
                
                documents = self.retrieve_diverse_documents(config, query)
                source_documents[source] = documents
                
                self.log.info("Documents retrieved from source", 
                             source=source,
                             num_docs=len(documents))
            
            return source_documents
            
        except Exception as e:
            self.log.error(f"Error retrieving from multiple sources: {e}")
            raise DocumentPortalException("Error retrieving from multiple sources", e) from e
    
    def get_diverse_context(self, mmr_config: Dict[str, Any], 
                          query: str, 
                          max_chars: int = 4000) -> Dict[str, Any]:
        """Get diverse context using MMR retrieval"""
        try:
            documents = self.retrieve_diverse_documents(mmr_config, query)
            
            # Combine document content with diversity tracking
            context_parts = []
            total_chars = 0
            diversity_scores = []
            
            for i, doc in enumerate(documents):
                content = doc.page_content
                
                if total_chars + len(content) <= max_chars:
                    context_parts.append(content)
                    total_chars += len(content)
                    
                    # Calculate diversity score (simplified - based on position)
                    diversity_score = 1.0 - (i / len(documents))  # Higher score for earlier docs
                    diversity_scores.append(diversity_score)
                else:
                    # Add partial content if it fits
                    remaining_chars = max_chars - total_chars
                    if remaining_chars > 100:
                        context_parts.append(content[:remaining_chars])
                        diversity_scores.append(0.5)  # Medium diversity for partial content
                    break
            
            context = "\n\n".join(context_parts)
            avg_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0
            
            result = {
                "context": context,
                "total_chars": len(context),
                "num_documents": len(context_parts),
                "average_diversity_score": round(avg_diversity, 3),
                "diversity_scores": diversity_scores,
                "session_id": self.session_id
            }
            
            self.log.info("Diverse context extracted using MMR", 
                         query=query, 
                         context_length=len(context),
                         avg_diversity=avg_diversity,
                         num_docs=len(context_parts))
            
            return result
            
        except Exception as e:
            self.log.error(f"Error getting diverse context: {e}")
            raise DocumentPortalException("Error getting diverse context", e) from e
    
    def get_multi_source_diverse_context(self, mmr_configs: Dict[str, Dict[str, Any]],
                                       query: str,
                                       max_chars: int = 4000,
                                       max_docs_per_source: int = 3) -> Dict[str, Any]:
        """Get diverse context from multiple sources using MMR"""
        try:
            source_documents = self.retrieve_from_multiple_sources(
                mmr_configs, query, max_docs_per_source
            )
            
            # Combine content by source
            context_by_source = {}
            total_chars = 0
            source_diversity_scores = {}
            
            for source, documents in source_documents.items():
                source_contents = []
                source_diversity = []
                
                for i, doc in enumerate(documents):
                    content = doc.page_content
                    
                    if total_chars + len(content) <= max_chars:
                        source_contents.append(content)
                        total_chars += len(content)
                        
                        # Calculate diversity score for this source
                        diversity_score = 1.0 - (i / len(documents))
                        source_diversity.append(diversity_score)
                    else:
                        # Add partial content if it fits
                        remaining_chars = max_chars - total_chars
                        if remaining_chars > 100:
                            source_contents.append(content[:remaining_chars])
                            source_diversity.append(0.5)
                        break
                
                if source_contents:
                    context_by_source[source] = "\n\n".join(source_contents)
                    source_diversity_scores[source] = sum(source_diversity) / len(source_diversity)
            
            # Combine all context
            all_context = "\n\n".join(context_by_source.values())
            overall_diversity = sum(source_diversity_scores.values()) / len(source_diversity_scores) if source_diversity_scores else 0.0
            
            result = {
                "context": all_context,
                "context_by_source": context_by_source,
                "sources_used": list(context_by_source.keys()),
                "total_chars": len(all_context),
                "num_sources": len(context_by_source),
                "overall_diversity_score": round(overall_diversity, 3),
                "source_diversity_scores": source_diversity_scores,
                "session_id": self.session_id
            }
            
            self.log.info("Multi-source diverse context extracted", 
                         query=query, 
                         context_length=len(all_context),
                         num_sources=len(context_by_source),
                         overall_diversity=overall_diversity)
            
            return result
            
        except Exception as e:
            self.log.error(f"Error getting multi-source diverse context: {e}")
            raise DocumentPortalException("Error getting multi-source diverse context", e) from e
    
    def optimize_mmr_parameters(self, query: str, 
                              sample_documents: List[Document],
                              test_lambda_values: List[float] = None) -> Dict[str, Any]:
        """
        Optimize MMR parameters based on query characteristics.
        
        Args:
            query: User query
            sample_documents: Sample documents to test with
            test_lambda_values: Lambda values to test (default: [0.3, 0.5, 0.7])
        """
        try:
            if test_lambda_values is None:
                test_lambda_values = [0.3, 0.5, 0.7]
            
            # Create temporary vector store for testing
            temp_store = FAISS.from_documents(sample_documents, self.embeddings)
            
            optimization_results = {}
            
            for lambda_val in test_lambda_values:
                # Create MMR retriever with current lambda
                mmr_retriever = self.create_mmr_retriever(
                    temp_store, 
                    fetch_k=10, 
                    lambda_mult=lambda_val, 
                    k=5
                )
                
                # Test retrieval
                documents = self.retrieve_diverse_documents(mmr_retriever, query)
                
                # Calculate diversity metrics
                diversity_score = self._calculate_diversity_score(documents)
                relevance_score = self._calculate_relevance_score(documents, query)
                
                optimization_results[lambda_val] = {
                    "diversity_score": diversity_score,
                    "relevance_score": relevance_score,
                    "combined_score": (diversity_score + relevance_score) / 2,
                    "num_documents": len(documents)
                }
            
            # Find optimal lambda
            best_lambda = max(optimization_results.keys(), 
                            key=lambda x: optimization_results[x]["combined_score"])
            
            result = {
                "optimal_lambda": best_lambda,
                "optimization_results": optimization_results,
                "recommended_fetch_k": 20,
                "recommended_k": 5,
                "session_id": self.session_id
            }
            
            self.log.info("MMR parameters optimized", 
                         optimal_lambda=best_lambda,
                         test_lambda_values=test_lambda_values)
            
            return result
            
        except Exception as e:
            self.log.error(f"Error optimizing MMR parameters: {e}")
            raise DocumentPortalException("Error optimizing MMR parameters", e) from e
    
    def _apply_mmr_algorithm(self, candidates: List[Tuple[Document, float]], 
                           query_embedding: List[float], 
                           lambda_mult: float, 
                           k: int, 
                           embeddings) -> List[Document]:
        """
        Apply Maximum Marginal Relevance algorithm to select diverse documents.
        
        Args:
            candidates: List of (document, score) tuples from initial retrieval
            query_embedding: Query embedding vector
            lambda_mult: Diversity parameter (0.0 = max diversity, 1.0 = max relevance)
            k: Number of documents to select
            embeddings: Embeddings model for computing similarities
        """
        try:
            if not candidates:
                return []
            
            # Sort candidates by relevance score (descending)
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            selected_docs = []
            selected_embeddings = []
            
            # Select the most relevant document first
            first_doc, first_score = candidates[0]
            selected_docs.append(first_doc)
            
            # Get embedding for first document
            first_embedding = embeddings.embed_documents([first_doc.page_content])[0]
            selected_embeddings.append(first_embedding)
            
            # Remove first document from candidates
            remaining_candidates = candidates[1:]
            
            # Select remaining documents using MMR
            for _ in range(min(k - 1, len(remaining_candidates))):
                max_mmr_score = -1
                best_candidate = None
                best_embedding = None
                
                for doc, relevance_score in remaining_candidates:
                    # Get document embedding
                    doc_embedding = embeddings.embed_documents([doc.page_content])[0]
                    
                    # Calculate similarity to query
                    query_similarity = self._cosine_similarity(query_embedding, doc_embedding)
                    
                    # Calculate maximum similarity to already selected documents
                    max_similarity = 0
                    if selected_embeddings:
                        similarities = [self._cosine_similarity(doc_embedding, sel_emb) 
                                     for sel_emb in selected_embeddings]
                        max_similarity = max(similarities)
                    
                    # Calculate MMR score
                    mmr_score = lambda_mult * query_similarity - (1 - lambda_mult) * max_similarity
                    
                    if mmr_score > max_mmr_score:
                        max_mmr_score = mmr_score
                        best_candidate = doc
                        best_embedding = doc_embedding
                
                if best_candidate is not None:
                    selected_docs.append(best_candidate)
                    selected_embeddings.append(best_embedding)
                    
                    # Remove selected candidate from remaining candidates
                    remaining_candidates = [(doc, score) for doc, score in remaining_candidates 
                                         if doc != best_candidate]
                else:
                    break
            
            return selected_docs
            
        except Exception as e:
            self.log.error(f"Error applying MMR algorithm: {e}")
            # Fallback to simple relevance-based selection
            return [doc for doc, _ in candidates[:k]]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import numpy as np
            
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            self.log.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def _calculate_diversity_score(self, documents: List[Document]) -> float:
        """Calculate diversity score based on document similarity"""
        try:
            if len(documents) <= 1:
                return 1.0  # Perfect diversity for single document
            
            # Simple diversity calculation based on content length variation
            content_lengths = [len(doc.page_content) for doc in documents]
            mean_length = sum(content_lengths) / len(content_lengths)
            
            # Calculate coefficient of variation
            variance = sum((length - mean_length) ** 2 for length in content_lengths) / len(content_lengths)
            std_dev = variance ** 0.5
            
            if mean_length == 0:
                return 0.0
            
            cv = std_dev / mean_length
            diversity_score = min(1.0, cv)  # Normalize to [0, 1]
            
            return diversity_score
            
        except Exception as e:
            self.log.error(f"Error calculating diversity score: {e}")
            return 0.5  # Default score on error
    
    def _calculate_relevance_score(self, documents: List[Document], query: str) -> float:
        """Calculate relevance score based on query-document similarity"""
        try:
            if not documents:
                return 0.0
            
            # Simple relevance calculation based on query term presence
            query_terms = query.lower().split()
            relevance_scores = []
            
            for doc in documents:
                doc_content = doc.page_content.lower()
                matching_terms = sum(1 for term in query_terms if term in doc_content)
                relevance_score = matching_terms / len(query_terms) if query_terms else 0.0
                relevance_scores.append(relevance_score)
            
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            return avg_relevance
            
        except Exception as e:
            self.log.error(f"Error calculating relevance score: {e}")
            return 0.5  # Default score on error
    
    def get_mmr_statistics(self, documents: List[Document], 
                          query: str) -> Dict[str, Any]:
        """Get comprehensive statistics about MMR retrieval"""
        try:
            if not documents:
                return {"message": "No documents retrieved"}
            
            # Calculate various metrics
            diversity_score = self._calculate_diversity_score(documents)
            relevance_score = self._calculate_relevance_score(documents, query)
            
            # Calculate content statistics
            content_lengths = [len(doc.page_content) for doc in documents]
            avg_content_length = sum(content_lengths) / len(content_lengths)
            min_content_length = min(content_lengths)
            max_content_length = max(content_lengths)
            
            # Source distribution
            source_counts = {}
            for doc in documents:
                source = doc.metadata.get("source_file", "unknown")
                source_counts[source] = source_counts.get(source, 0) + 1
            
            stats = {
                "total_documents": len(documents),
                "diversity_score": round(diversity_score, 3),
                "relevance_score": round(relevance_score, 3),
                "combined_score": round((diversity_score + relevance_score) / 2, 3),
                "average_content_length": round(avg_content_length, 2),
                "min_content_length": min_content_length,
                "max_content_length": max_content_length,
                "source_distribution": source_counts,
                "num_sources": len(source_counts),
                "session_id": self.session_id,
                "retrieval_timestamp": datetime.utcnow().isoformat()
            }
            
            self.log.info("MMR statistics generated", 
                         total_docs=len(documents),
                         diversity_score=diversity_score,
                         relevance_score=relevance_score,
                         num_sources=len(source_counts))
            
            return stats
            
        except Exception as e:
            self.log.error(f"Error generating MMR statistics: {e}")
            raise DocumentPortalException("Error generating MMR statistics", e) from e
