import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import LLMChainFilter
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from datetime import datetime

class ContextualCompression:
    """
    Implements contextual compression for multi-document chat to focus on relevant information.
    """
    def __init__(self, session_id: str):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.session_id = session_id
            self.model_loader = ModelLoader()
            self.embeddings = self.model_loader.load_embeddings()
            self.llm = self.model_loader.load_llm()
            
            self.log.info("ContextualCompression initialized", session_id=session_id)
            
        except Exception as e:
            self.log.error(f"Error initializing ContextualCompression: {e}")
            raise DocumentPortalException("Error initializing ContextualCompression", e) from e
    
    def create_llm_compressor(self, threshold: float = 0.5) -> LLMChainExtractor:
        """
        Create LLM-based document compressor.
        
        Args:
            threshold: Relevance threshold for document filtering
        """
        try:
            compressor = LLMChainExtractor.from_llm(
                llm=self.llm,
                threshold=threshold
            )
            
            self.log.info("LLM compressor created", threshold=threshold)
            return compressor
            
        except Exception as e:
            self.log.error(f"Error creating LLM compressor: {e}")
            raise DocumentPortalException("Error creating LLM compressor", e) from e
    
    def create_embeddings_filter(self, threshold: float = 0.8) -> EmbeddingsFilter:
        """
        Create embeddings-based document filter.
        
        Args:
            threshold: Similarity threshold for filtering
        """
        try:
            filter_compressor = EmbeddingsFilter(
                embeddings=self.embeddings,
                similarity_threshold=threshold
            )
            
            self.log.info("Embeddings filter created", threshold=threshold)
            return filter_compressor
            
        except Exception as e:
            self.log.error(f"Error creating embeddings filter: {e}")
            raise DocumentPortalException("Error creating embeddings filter", e) from e
    
    def create_llm_filter(self, threshold: float = 0.5) -> LLMChainFilter:
        """
        Create LLM-based document filter.
        
        Args:
            threshold: Relevance threshold for filtering
        """
        try:
            filter_compressor = LLMChainFilter.from_llm(
                llm=self.llm,
                threshold=threshold
            )
            
            self.log.info("LLM filter created", threshold=threshold)
            return filter_compressor
            
        except Exception as e:
            self.log.error(f"Error creating LLM filter: {e}")
            raise DocumentPortalException("Error creating LLM filter", e) from e
    
    def create_compression_pipeline(self, 
                                  use_embeddings_filter: bool = True,
                                  use_llm_filter: bool = True,
                                  use_llm_extractor: bool = True,
                                  embeddings_threshold: float = 0.8,
                                  llm_threshold: float = 0.5) -> DocumentCompressorPipeline:
        """
        Create a document compression pipeline with multiple stages.
        
        Args:
            use_embeddings_filter: Whether to use embeddings-based filtering
            use_llm_filter: Whether to use LLM-based filtering
            use_llm_extractor: Whether to use LLM-based extraction
            embeddings_threshold: Threshold for embeddings filter
            llm_threshold: Threshold for LLM-based components
        """
        try:
            compressors = []
            
            # Add embeddings filter if enabled
            if use_embeddings_filter:
                embeddings_filter = self.create_embeddings_filter(embeddings_threshold)
                compressors.append(embeddings_filter)
            
            # Add LLM filter if enabled
            if use_llm_filter:
                llm_filter = self.create_llm_filter(llm_threshold)
                compressors.append(llm_filter)
            
            # Add LLM extractor if enabled
            if use_llm_extractor:
                llm_extractor = self.create_llm_compressor(llm_threshold)
                compressors.append(llm_extractor)
            
            if not compressors:
                raise DocumentPortalException("At least one compressor must be enabled")
            
            pipeline = DocumentCompressorPipeline(compressors=compressors)
            
            self.log.info("Compression pipeline created", 
                         num_compressors=len(compressors),
                         use_embeddings_filter=use_embeddings_filter,
                         use_llm_filter=use_llm_filter,
                         use_llm_extractor=use_llm_extractor)
            
            return pipeline
            
        except Exception as e:
            self.log.error(f"Error creating compression pipeline: {e}")
            raise DocumentPortalException("Error creating compression pipeline", e) from e
    
    def create_contextual_retriever(self, base_retriever, 
                                  compression_pipeline: DocumentCompressorPipeline) -> ContextualCompressionRetriever:
        """
        Create contextual compression retriever.
        
        Args:
            base_retriever: Base retriever to compress
            compression_pipeline: Document compression pipeline
        """
        try:
            contextual_retriever = ContextualCompressionRetriever(
                base_compressor=compression_pipeline,
                base_retriever=base_retriever
            )
            
            self.log.info("Contextual compression retriever created")
            return contextual_retriever
            
        except Exception as e:
            self.log.error(f"Error creating contextual retriever: {e}")
            raise DocumentPortalException("Error creating contextual retriever", e) from e
    
    def compress_documents(self, documents: List[Document], 
                          query: str,
                          compression_pipeline: DocumentCompressorPipeline) -> List[Document]:
        """
        Compress documents using the compression pipeline.
        
        Args:
            documents: Documents to compress
            query: User query for context
            compression_pipeline: Document compression pipeline
        """
        try:
            # Create a simple retriever for compression
            from langchain_core.retrievers import BaseRetriever
            
            class SimpleRetriever(BaseRetriever):
                def __init__(self, documents):
                    self.documents = documents
                
                def get_relevant_documents(self, query):
                    return self.documents
                
                async def aget_relevant_documents(self, query):
                    return self.documents
            
            simple_retriever = SimpleRetriever(documents)
            
            # Create contextual retriever
            contextual_retriever = self.create_contextual_retriever(
                simple_retriever, compression_pipeline
            )
            
            # Get compressed documents
            compressed_docs = contextual_retriever.get_relevant_documents(query)
            
            self.log.info("Documents compressed successfully", 
                         original_count=len(documents),
                         compressed_count=len(compressed_docs),
                         compression_ratio=len(compressed_docs) / len(documents) if documents else 0)
            
            return compressed_docs
            
        except Exception as e:
            self.log.error(f"Error compressing documents: {e}")
            raise DocumentPortalException("Error compressing documents", e) from e
    
    def get_compressed_context(self, documents: List[Document],
                             query: str,
                             compression_pipeline: DocumentCompressorPipeline,
                             max_chars: int = 4000) -> Dict[str, Any]:
        """
        Get compressed context from documents.
        
        Args:
            documents: Documents to compress
            query: User query
            compression_pipeline: Document compression pipeline
            max_chars: Maximum characters in context
        """
        try:
            # Compress documents
            compressed_docs = self.compress_documents(documents, query, compression_pipeline)
            
            # Combine compressed content
            context_parts = []
            total_chars = 0
            compression_stats = {
                "original_docs": len(documents),
                "compressed_docs": len(compressed_docs),
                "compression_ratio": len(compressed_docs) / len(documents) if documents else 0
            }
            
            for doc in compressed_docs:
                content = doc.page_content
                
                if total_chars + len(content) <= max_chars:
                    context_parts.append(content)
                    total_chars += len(content)
                else:
                    # Add partial content if it fits
                    remaining_chars = max_chars - total_chars
                    if remaining_chars > 100:
                        context_parts.append(content[:remaining_chars])
                    break
            
            context = "\n\n".join(context_parts)
            
            result = {
                "context": context,
                "total_chars": len(context),
                "num_compressed_docs": len(context_parts),
                "compression_stats": compression_stats,
                "session_id": self.session_id
            }
            
            self.log.info("Compressed context extracted", 
                         query=query,
                         context_length=len(context),
                         compression_ratio=compression_stats["compression_ratio"])
            
            return result
            
        except Exception as e:
            self.log.error(f"Error getting compressed context: {e}")
            raise DocumentPortalException("Error getting compressed context", e) from e
    
    def analyze_compression_effectiveness(self, original_docs: List[Document],
                                        compressed_docs: List[Document],
                                        query: str) -> Dict[str, Any]:
        """
        Analyze the effectiveness of document compression.
        
        Args:
            original_docs: Original documents
            compressed_docs: Compressed documents
            query: User query
        """
        try:
            # Calculate compression metrics
            compression_ratio = len(compressed_docs) / len(original_docs) if original_docs else 0
            
            # Calculate content reduction
            original_content_length = sum(len(doc.page_content) for doc in original_docs)
            compressed_content_length = sum(len(doc.page_content) for doc in compressed_docs)
            content_reduction_ratio = (original_content_length - compressed_content_length) / original_content_length if original_content_length > 0 else 0
            
            # Calculate relevance preservation (simplified)
            query_terms = query.lower().split()
            original_relevance = self._calculate_document_relevance(original_docs, query_terms)
            compressed_relevance = self._calculate_document_relevance(compressed_docs, query_terms)
            
            effectiveness = {
                "compression_ratio": round(compression_ratio, 3),
                "content_reduction_ratio": round(content_reduction_ratio, 3),
                "original_relevance": round(original_relevance, 3),
                "compressed_relevance": round(compressed_relevance, 3),
                "relevance_preservation": round(compressed_relevance / original_relevance, 3) if original_relevance > 0 else 0,
                "original_docs_count": len(original_docs),
                "compressed_docs_count": len(compressed_docs),
                "original_content_length": original_content_length,
                "compressed_content_length": compressed_content_length,
                "session_id": self.session_id,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
            self.log.info("Compression effectiveness analyzed", 
                         compression_ratio=compression_ratio,
                         content_reduction_ratio=content_reduction_ratio,
                         relevance_preservation=effectiveness["relevance_preservation"])
            
            return effectiveness
            
        except Exception as e:
            self.log.error(f"Error analyzing compression effectiveness: {e}")
            raise DocumentPortalException("Error analyzing compression effectiveness", e) from e
    
    def _calculate_document_relevance(self, documents: List[Document], 
                                    query_terms: List[str]) -> float:
        """Calculate average relevance of documents to query terms"""
        try:
            if not documents or not query_terms:
                return 0.0
            
            relevance_scores = []
            
            for doc in documents:
                doc_content = doc.page_content.lower()
                matching_terms = sum(1 for term in query_terms if term in doc_content)
                relevance_score = matching_terms / len(query_terms)
                relevance_scores.append(relevance_score)
            
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            return avg_relevance
            
        except Exception as e:
            self.log.error(f"Error calculating document relevance: {e}")
            return 0.0
    
    def optimize_compression_parameters(self, documents: List[Document],
                                     query: str,
                                     test_thresholds: List[float] = None) -> Dict[str, Any]:
        """
        Optimize compression parameters based on document characteristics.
        
        Args:
            documents: Documents to test with
            query: User query
            test_thresholds: Thresholds to test (default: [0.3, 0.5, 0.7, 0.9])
        """
        try:
            if test_thresholds is None:
                test_thresholds = [0.3, 0.5, 0.7, 0.9]
            
            optimization_results = {}
            
            for threshold in test_thresholds:
                # Create compression pipeline with current threshold
                pipeline = self.create_compression_pipeline(
                    use_embeddings_filter=True,
                    use_llm_filter=True,
                    use_llm_extractor=True,
                    embeddings_threshold=threshold,
                    llm_threshold=threshold
                )
                
                # Test compression
                compressed_docs = self.compress_documents(documents, query, pipeline)
                
                # Analyze effectiveness
                effectiveness = self.analyze_compression_effectiveness(
                    documents, compressed_docs, query
                )
                
                optimization_results[threshold] = effectiveness
            
            # Find optimal threshold (balance between compression and relevance preservation)
            best_threshold = max(optimization_results.keys(), 
                               key=lambda x: optimization_results[x]["relevance_preservation"] * 
                                            (1 - optimization_results[x]["compression_ratio"]))
            
            result = {
                "optimal_threshold": best_threshold,
                "optimization_results": optimization_results,
                "recommended_embeddings_threshold": best_threshold,
                "recommended_llm_threshold": best_threshold,
                "session_id": self.session_id
            }
            
            self.log.info("Compression parameters optimized", 
                         optimal_threshold=best_threshold,
                         test_thresholds=test_thresholds)
            
            return result
            
        except Exception as e:
            self.log.error(f"Error optimizing compression parameters: {e}")
            raise DocumentPortalException("Error optimizing compression parameters", e) from e
    
    def get_compression_statistics(self, documents: List[Document],
                                 compressed_docs: List[Document],
                                 query: str) -> Dict[str, Any]:
        """Get comprehensive statistics about document compression"""
        try:
            if not documents:
                return {"message": "No documents to analyze"}
            
            # Calculate basic statistics
            compression_ratio = len(compressed_docs) / len(documents)
            
            # Calculate content statistics
            original_lengths = [len(doc.page_content) for doc in documents]
            compressed_lengths = [len(doc.page_content) for doc in compressed_docs]
            
            avg_original_length = sum(original_lengths) / len(original_lengths)
            avg_compressed_length = sum(compressed_lengths) / len(compressed_lengths) if compressed_lengths else 0
            
            # Calculate source distribution
            original_sources = {}
            compressed_sources = {}
            
            for doc in documents:
                source = doc.metadata.get("source_file", "unknown")
                original_sources[source] = original_sources.get(source, 0) + 1
            
            for doc in compressed_docs:
                source = doc.metadata.get("source_file", "unknown")
                compressed_sources[source] = compressed_sources.get(source, 0) + 1
            
            stats = {
                "total_original_documents": len(documents),
                "total_compressed_documents": len(compressed_docs),
                "compression_ratio": round(compression_ratio, 3),
                "average_original_length": round(avg_original_length, 2),
                "average_compressed_length": round(avg_compressed_length, 2),
                "length_reduction_ratio": round((avg_original_length - avg_compressed_length) / avg_original_length, 3) if avg_original_length > 0 else 0,
                "original_source_distribution": original_sources,
                "compressed_source_distribution": compressed_sources,
                "num_original_sources": len(original_sources),
                "num_compressed_sources": len(compressed_sources),
                "session_id": self.session_id,
                "compression_timestamp": datetime.utcnow().isoformat()
            }
            
            self.log.info("Compression statistics generated", 
                         original_docs=len(documents),
                         compressed_docs=len(compressed_docs),
                         compression_ratio=compression_ratio)
            
            return stats
            
        except Exception as e:
            self.log.error(f"Error generating compression statistics: {e}")
            raise DocumentPortalException("Error generating compression statistics", e) from e
