import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain.schema import Document
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

class SingleDocumentEvaluation:
    """
    Evaluates chat quality and provides metrics for single document chat.
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
                "response_quality_scores": []
            }
            
            self.log.info("SingleDocumentEvaluation initialized", session_id=session_id)
            
        except Exception as e:
            self.log.error(f"Error initializing SingleDocumentEvaluation: {e}")
            raise DocumentPortalException("Error initializing SingleDocumentEvaluation", e) from e
    
    def add_chat_interaction(self, query: str, response: str, context: str, 
                           response_time: float, retrieved_docs: List[Document]):
        """Add a chat interaction to evaluation history"""
        try:
            interaction = {
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "response": response,
                "context": context,
                "response_time": response_time,
                "num_retrieved_docs": len(retrieved_docs),
                "session_id": self.session_id
            }
            
            self.chat_history.append(interaction)
            self.evaluation_metrics["total_queries"] += 1
            
            # Update average response time
            total_time = sum(interaction["response_time"] for interaction in self.chat_history)
            self.evaluation_metrics["average_response_time"] = total_time / len(self.chat_history)
            
            self.log.info("Chat interaction added to evaluation", 
                         query_length=len(query),
                         response_length=len(response),
                         response_time=response_time)
            
        except Exception as e:
            self.log.error(f"Error adding chat interaction: {e}")
            raise DocumentPortalException("Error adding chat interaction", e) from e
    
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
            
            self.log.info("Context relevance evaluated", 
                         query=query, 
                         relevance_score=score)
            
            return score
            
        except Exception as e:
            self.log.error(f"Error evaluating context relevance: {e}")
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
            
            self.log.info("Response quality evaluated", 
                         query=query, 
                         quality_score=score)
            
            return score
            
        except Exception as e:
            self.log.error(f"Error evaluating response quality: {e}")
            return 0.5  # Default score on error
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get comprehensive evaluation summary"""
        try:
            if not self.chat_history:
                return {"message": "No chat interactions to evaluate"}
            
            # Calculate metrics
            total_queries = self.evaluation_metrics["total_queries"]
            avg_response_time = self.evaluation_metrics["average_response_time"]
            
            # Calculate average relevance and quality scores
            avg_relevance = (sum(self.evaluation_metrics["context_relevance_scores"]) / 
                           len(self.evaluation_metrics["context_relevance_scores"]) 
                           if self.evaluation_metrics["context_relevance_scores"] else 0.0)
            
            avg_quality = (sum(self.evaluation_metrics["response_quality_scores"]) / 
                         len(self.evaluation_metrics["response_quality_scores"]) 
                         if self.evaluation_metrics["response_quality_scores"] else 0.0)
            
            # Calculate success rate (queries with good context relevance)
            successful_queries = sum(1 for score in self.evaluation_metrics["context_relevance_scores"] 
                                   if score > 0.6)
            success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
            
            summary = {
                "session_id": self.session_id,
                "total_queries": total_queries,
                "success_rate_percentage": round(success_rate, 2),
                "average_response_time_seconds": round(avg_response_time, 2),
                "average_context_relevance": round(avg_relevance, 3),
                "average_response_quality": round(avg_quality, 3),
                "evaluation_timestamp": datetime.utcnow().isoformat()
            }
            
            self.log.info("Evaluation summary generated", 
                         total_queries=total_queries,
                         success_rate=success_rate,
                         avg_relevance=avg_relevance,
                         avg_quality=avg_quality)
            
            return summary
            
        except Exception as e:
            self.log.error(f"Error generating evaluation summary: {e}")
            raise DocumentPortalException("Error generating evaluation summary", e) from e
    
    def save_evaluation_data(self, file_path: Optional[str] = None):
        """Save evaluation data to file"""
        try:
            if not file_path:
                file_path = os.path.join("data", "evaluations", 
                                       f"evaluation_{self.session_id}.json")
            
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
            
            self.log.info("Evaluation data saved", file_path=file_path)
            
        except Exception as e:
            self.log.error(f"Error saving evaluation data: {e}")
            raise DocumentPortalException("Error saving evaluation data", e) from e
