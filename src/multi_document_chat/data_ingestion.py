import os
import fitz
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

class MultiDocumentIngestion:
    """
    Handles PDF ingestion and text chunking for multi-document chat functionality.
    """
    def __init__(self, data_dir=None, session_id=None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.data_dir = data_dir or os.getenv(
                "DATA_STORAGE_PATH",
                os.path.join(os.getcwd(), "data", "multi_document_chat")
            )
            self.session_id = session_id or f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Create session directory
            self.session_path = os.path.join(self.data_dir, self.session_id)
            os.makedirs(self.session_path, exist_ok=True)
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            self.log.info("MultiDocumentIngestion initialized", session_id=self.session_id)
            
        except Exception as e:
            self.log.error(f"Error initializing MultiDocumentIngestion: {e}")
            raise DocumentPortalException("Error initializing MultiDocumentIngestion", e) from e
    
    def save_multiple_pdfs(self, uploaded_files: List) -> List[str]:
        """Save multiple uploaded PDFs to session directory"""
        try:
            saved_paths = []
            
            for uploaded_file in uploaded_files:
                filename = os.path.basename(uploaded_file.name)
                
                if not filename.lower().endswith(".pdf"):
                    raise DocumentPortalException(f"Invalid file type for {filename}. Only PDFs are allowed.")
                
                save_path = os.path.join(self.session_path, filename)
                
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                saved_paths.append(save_path)
                self.log.info("PDF saved successfully", file=filename, save_path=save_path)
            
            return saved_paths
            
        except Exception as e:
            self.log.error(f"Error saving multiple PDFs: {e}")
            raise DocumentPortalException("Error saving multiple PDFs", e) from e
    
    def read_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        try:
            text_chunks = []
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text_chunks.append(f"\n--- Page {page_num} ---\n{page.get_text()}")
            text = "\n".join(text_chunks)
            
            self.log.info("PDF read successfully", pdf_path=pdf_path, pages=len(text_chunks))
            return text
            
        except Exception as e:
            self.log.error(f"Error reading PDF: {e}")
            raise DocumentPortalException("Error reading PDF", e) from e
    
    def create_documents_with_source(self, text: str, source_file: str, 
                                   metadata: Dict[str, Any] = None) -> List[Document]:
        """Split text into chunks and create Document objects with source tracking"""
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create Document objects with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    "chunk_id": i,
                    "session_id": self.session_id,
                    "total_chunks": len(chunks),
                    "source_file": source_file,
                    "source_type": "pdf"
                }
                if metadata:
                    doc_metadata.update(metadata)
                
                document = Document(
                    page_content=chunk,
                    metadata=doc_metadata
                )
                documents.append(document)
            
            self.log.info("Documents created successfully", 
                         num_chunks=len(documents), 
                         source_file=source_file,
                         session_id=self.session_id)
            return documents
            
        except Exception as e:
            self.log.error(f"Error creating documents: {e}")
            raise DocumentPortalException("Error creating documents", e) from e
    
    def process_multiple_pdfs(self, uploaded_files: List) -> Tuple[List[Document], List[str]]:
        """Complete multi-PDF processing pipeline"""
        try:
            all_documents = []
            pdf_paths = []
            
            # Save all PDFs
            pdf_paths = self.save_multiple_pdfs(uploaded_files)
            
            # Process each PDF
            for i, pdf_path in enumerate(pdf_paths):
                # Read PDF
                text = self.read_pdf(pdf_path)
                
                # Create documents with source tracking
                source_file = os.path.basename(pdf_path)
                documents = self.create_documents_with_source(
                    text, 
                    source_file, 
                    {"file_index": i, "total_files": len(pdf_paths)}
                )
                
                all_documents.extend(documents)
            
            self.log.info("Multiple PDFs processed successfully", 
                         num_files=len(pdf_paths),
                         total_documents=len(all_documents),
                         session_id=self.session_id)
            
            return all_documents, pdf_paths
            
        except Exception as e:
            self.log.error(f"Error in multi-PDF processing pipeline: {e}")
            raise DocumentPortalException("Error in multi-PDF processing pipeline", e) from e
    
    def get_document_summary(self, documents: List[Document]) -> Dict[str, Any]:
        """Generate summary of processed documents"""
        try:
            # Group documents by source
            source_groups = {}
            for doc in documents:
                source = doc.metadata.get("source_file", "unknown")
                if source not in source_groups:
                    source_groups[source] = []
                source_groups[source].append(doc)
            
            # Calculate statistics
            total_chunks = len(documents)
            total_sources = len(source_groups)
            
            # Calculate average chunks per source
            avg_chunks_per_source = total_chunks / total_sources if total_sources > 0 else 0
            
            summary = {
                "total_documents": total_chunks,
                "total_sources": total_sources,
                "average_chunks_per_source": round(avg_chunks_per_source, 2),
                "sources": list(source_groups.keys()),
                "session_id": self.session_id,
                "processing_timestamp": datetime.utcnow().isoformat()
            }
            
            self.log.info("Document summary generated", 
                         total_documents=total_chunks,
                         total_sources=total_sources,
                         session_id=self.session_id)
            
            return summary
            
        except Exception as e:
            self.log.error(f"Error generating document summary: {e}")
            raise DocumentPortalException("Error generating document summary", e) from e
