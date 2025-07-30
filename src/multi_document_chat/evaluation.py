import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain.schema import Document
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

class MultiDocumentEvaluation:
    """
    Evaluates chat quality and provides metrics for multi-document chat.
    """
    def __init__(self, session_id: str):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.session_id = session_id
            self.model_loader = ModelLoader()
            self.llm = self.model_loader.load_llm()
            
            # Initialize evaluation metrics
            self.chat_history = []
            self.evaluation_metrics = {
                "total_queries": 0,
                "successful_queries": 0,
                "average_response_time": 0.0,
                "context_relevance_scores": [],
                "response_quality_scores": [],
                "source_diversity_scores": [],
                "cross_document_coherence_scores": []
            }
            
            self.log.info("MultiDocumentEvaluation initialized", session_id=session_id)
            
        except Exception as e:
            self.log.error(f"Error initializing MultiDocumentEvaluation: {e}")
            raise DocumentPortalException("Error initializing MultiDocumentEvaluation", e) from e
    
    def add_chat_interaction(self, query: str, response: str, context: str, 
                           response_time: float, retrieved_docs: List[Document],
                           sources_used: List[str] = None):
        """Add a chat interaction to evaluation history"""
        try:
            interaction = {
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "response": response,
                "context": context,
                "response_time": response_time,
                "num_retrieved_docs": len(retrieved_docs),
                "sources_used": sources_used or [],
                "session_id": self.session_id
            }
            
            self.chat_history.append(interaction)
            self.evaluation_metrics["total_queries"] += 1
            
            # Update average response time
            total_time = sum(interaction["response_time"] for interaction in self.chat_history)
            self.evaluation_metrics["average_response_time"] = total_time / len(self.chat_history)
            
            self.log.info("Multi-document chat interaction added to evaluation", 
                         query_length=len(query),
                         response_length=len(response),
                         response_time=response_time,
                         num_sources=len(sources_used) if sources_used else 0)
            
        except Exception as e:
            self.log.error(f"Error adding multi-document chat interaction: {e}")
            raise DocumentPortalException("Error adding multi-document chat interaction", e) from e
    
    def evaluate_context_relevance(self, query: str, context: str) -> float:
        """Evaluate how relevant the retrieved context is to the query"""
        try:
            evaluation_prompt = f"""
            Evaluate how relevant the following context is to the user's query.
            Rate from 0.0 (completely irrelevant) to 1.0 (highly relevant).
            
            Query: {query}
            Context: {context[:1000]}  # Limit context length for evaluation
            
            Provide only a number between 0.0 and 1.0 as your response.
            """
            
            response = self.llm.invoke(evaluation_prompt)
            
            # Extract numeric score from response
            try:
                score = float(response.content.strip())
                score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
            except ValueError:
                score = 0.5  # Default score if parsing fails
            
            self.evaluation_metrics["context_relevance_scores"].append(score)
            
            self.log.info("Multi-document context relevance evaluated", 
                         query=query, 
                         relevance_score=score)
            
            return score
            
        except Exception as e:
            self.log.error(f"Error evaluating multi-document context relevance: {e}")
            return 0.5  # Default score on error
    
    def evaluate_response_quality(self, query: str, response: str, context: str) -> float:
        """Evaluate the quality of the AI response"""
        try:
            evaluation_prompt = f"""
            Evaluate the quality of the AI response to the user's query.
            Consider accuracy, completeness, and helpfulness.
            Rate from 0.0 (poor quality) to 1.0 (excellent quality).
            
            Query: {query}
            Context: {context[:1000]}
            Response: {response}
            
            Provide only a number between 0.0 and 1.0 as your response.
            """
            
            response_eval = self.llm.invoke(evaluation_prompt)
            
            # Extract numeric score from response
            try:
                score = float(response_eval.content.strip())
                score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
            except ValueError:
                score = 0.5  # Default score if parsing fails
            
            self.evaluation_metrics["response_quality_scores"].append(score)
            
            self.log.info("Multi-document response quality evaluated", 
                         query=query, 
                         quality_score=score)
            
            return score
            
        except Exception as e:
            self.log.error(f"Error evaluating multi-document response quality: {e}")
            return 0.5  # Default score on error
    
    def evaluate_source_diversity(self, sources_used: List[str], 
                                total_sources: int) -> float:
        """Evaluate the diversity of sources used in the response"""
        try:
            if not sources_used or total_sources == 0:
                return 0.0
            
            # Calculate diversity as the ratio of unique sources used
            unique_sources = len(set(sources_used))
            diversity_score = unique_sources / total_sources
            
            self.evaluation_metrics["source_diversity_scores"].append(diversity_score)
            
            self.log.info("Source diversity evaluated", 
                         sources_used=sources_used,
                         total_sources=total_sources,
                         diversity_score=diversity_score)
            
            return diversity_score
            
        except Exception as e:
            self.log.error(f"Error evaluating source diversity: {e}")
            return 0.5  # Default score on error
    
    def evaluate_cross_document_coherence(self, response: str, 
                                        context_by_source: Dict[str, str]) -> float:
        """Evaluate how well the response synthesizes information across documents"""
        try:
            if not context_by_source:
                return 0.0
            
            evaluation_prompt = f"""
            Evaluate how well the response synthesizes information across multiple documents.
            Consider how well it connects ideas from different sources.
            Rate from 0.0 (poor synthesis) to 1.0 (excellent synthesis).
            
            Response: {response}
            Number of sources: {len(context_by_source)}
            
            Provide only a number between 0.0 and 1.0 as your response.
            """
            
            response_eval = self.llm.invoke(evaluation_prompt)
            
            # Extract numeric score from response
            try:
                score = float(response_eval.content.strip())
                score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
            except ValueError:
                score = 0.5  # Default score if parsing fails
            
            self.evaluation_metrics["cross_document_coherence_scores"].append(score)
            
            self.log.info("Cross-document coherence evaluated", 
                         num_sources=len(context_by_source),
                         coherence_score=score)
            
            return score
            
        except Exception as e:
            self.log.error(f"Error evaluating cross-document coherence: {e}")
            return 0.5  # Default score on error
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get comprehensive evaluation summary for multi-document chat"""
        try:
            if not self.chat_history:
                return {"message": "No multi-document chat interactions to evaluate"}
            
            # Calculate metrics
            total_queries = self.evaluation_metrics["total_queries"]
            avg_response_time = self.evaluation_metrics["average_response_time"]
            
            # Calculate average scores
            avg_relevance = (sum(self.evaluation_metrics["context_relevance_scores"]) / 
                           len(self.evaluation_metrics["context_relevance_scores"]) 
                           if self.evaluation_metrics["context_relevance_scores"] else 0.0)
            
            avg_quality = (sum(self.evaluation_metrics["response_quality_scores"]) / 
                         len(self.evaluation_metrics["response_quality_scores"]) 
                         if self.evaluation_metrics["response_quality_scores"] else 0.0)
            
            avg_diversity = (sum(self.evaluation_metrics["source_diversity_scores"]) / 
                           len(self.evaluation_metrics["source_diversity_scores"]) 
                           if self.evaluation_metrics["source_diversity_scores"] else 0.0)
            
            avg_coherence = (sum(self.evaluation_metrics["cross_document_coherence_scores"]) / 
                           len(self.evaluation_metrics["cross_document_coherence_scores"]) 
                           if self.evaluation_metrics["cross_document_coherence_scores"] else 0.0)
            
            # Calculate success rate
            successful_queries = sum(1 for score in self.evaluation_metrics["context_relevance_scores"] 
                                   if score > 0.6)
            success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
            
            # Calculate average sources per query
            total_sources_used = sum(len(interaction.get("sources_used", [])) 
                                   for interaction in self.chat_history)
            avg_sources_per_query = total_sources_used / total_queries if total_queries > 0 else 0
            
            summary = {
                "session_id": self.session_id,
                "total_queries": total_queries,
                "success_rate_percentage": round(success_rate, 2),
                "average_response_time_seconds": round(avg_response_time, 2),
                "average_context_relevance": round(avg_relevance, 3),
                "average_response_quality": round(avg_quality, 3),
                "average_source_diversity": round(avg_diversity, 3),
                "average_cross_document_coherence": round(avg_coherence, 3),
                "average_sources_per_query": round(avg_sources_per_query, 2),
                "evaluation_timestamp": datetime.utcnow().isoformat()
            }
            
            self.log.info("Multi-document evaluation summary generated", 
                         total_queries=total_queries,
                         success_rate=success_rate,
                         avg_relevance=avg_relevance,
                         avg_quality=avg_quality,
                         avg_diversity=avg_diversity,
                         avg_coherence=avg_coherence)
            
            return summary
            
        except Exception as e:
            self.log.error(f"Error generating multi-document evaluation summary: {e}")
            raise DocumentPortalException("Error generating multi-document evaluation summary", e) from e
    
    def save_evaluation_data(self, file_path: Optional[str] = None):
        """Save evaluation data to file"""
        try:
            if not file_path:
                file_path = os.path.join("data", "evaluations", 
                                       f"multi_doc_evaluation_{self.session_id}.json")
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            evaluation_data = {
                "session_id": self.session_id,
                "chat_history": self.chat_history,
                "evaluation_metrics": self.evaluation_metrics,
                "summary": self.get_evaluation_summary()
            }
            
            import json
            with open(file_path, 'w') as f:
                json.dump(evaluation_data, f, indent=2)
            
            self.log.info("Multi-document evaluation data saved", file_path=file_path)
            
        except Exception as e:
            self.log.error(f"Error saving multi-document evaluation data: {e}")
            raise DocumentPortalException("Error saving multi-document evaluation data", e) from e
