import os
import fitz
import uuid
from datetime import datetime
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

class SingleDocumentIngestion:
    """
    Handles PDF ingestion and text chunking for single document chat functionality.
    """
    def __init__(self, data_dir=None, session_id=None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.data_dir = data_dir or os.getenv(
                "DATA_STORAGE_PATH",
                os.path.join(os.getcwd(), "data", "single_document_chat")
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
            
            self.log.info("SingleDocumentIngestion initialized", session_id=self.session_id)
            
        except Exception as e:
            self.log.error(f"Error initializing SingleDocumentIngestion: {e}")
            raise DocumentPortalException("Error initializing SingleDocumentIngestion", e) from e
    
    def save_pdf(self, uploaded_file) -> str:
        """Save uploaded PDF to session directory"""
        try:
            filename = os.path.basename(uploaded_file.name)
            
            if not filename.lower().endswith(".pdf"):
                raise DocumentPortalException("Invalid file type. Only PDFs are allowed.")
            
            save_path = os.path.join(self.session_path, filename)
            
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            self.log.info("PDF saved successfully", file=filename, save_path=save_path)
            return save_path
            
        except Exception as e:
            self.log.error(f"Error saving PDF: {e}")
            raise DocumentPortalException("Error saving PDF", e) from e
    
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
    
    def create_documents(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """Split text into chunks and create Document objects"""
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create Document objects with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    "chunk_id": i,
                    "session_id": self.session_id,
                    "total_chunks": len(chunks)
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
                         session_id=self.session_id)
            return documents
            
        except Exception as e:
            self.log.error(f"Error creating documents: {e}")
            raise DocumentPortalException("Error creating documents", e) from e
    
    def process_pdf(self, uploaded_file) -> tuple[List[Document], str]:
        """Complete PDF processing pipeline"""
        try:
            # Save PDF
            pdf_path = self.save_pdf(uploaded_file)
            
            # Read PDF
            text = self.read_pdf(pdf_path)
            
            # Create documents
            documents = self.create_documents(text, {"source_file": uploaded_file.name})
            
            return documents, pdf_path
            
        except Exception as e:
            self.log.error(f"Error in PDF processing pipeline: {e}")
            raise DocumentPortalException("Error in PDF processing pipeline", e) from e
