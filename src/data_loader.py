"""
Production-grade PDF loader with smart chunking, table extraction, and metadata.
Handles: text extraction, table-to-text conversion, metadata preservation.
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib

import pdfplumber
import pandas as pd
import numpy as np
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import Settings

from src.logging_utils import setup_logger

from src.exceptions import RAGError
logger = setup_logger(__name__)


@dataclass
class Document:
    """Represents a chunked document with metadata"""
    doc_id: str  # Unique ID
    text: str  # Chunk content
    source_file: str  # Original PDF filename
    page_num: int  # Page number
    chunk_idx: int  # Chunk index within page
    doc_type: str  # "text" or "table"
    word_count: int
    char_count: int
    timestamp: str  # ISO datetime
    source_hash: str  # Hash of source text
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SmartChunker:
    """
    Intelligent text chunking that respects boundaries:
    1. Sentence boundaries (don't split mid-sentence)
    2. Paragraph boundaries (preserve context)
    3. Section boundaries (if detected)
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",           # Paragraph
                "\n",             # Line
                ". ",             # Sentence
                "! ",
                "? ",
                " ",              # Word
                ""                # Character
            ],
            length_function=len,
            is_separator_regex=False
        )
    
    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Document]:
        """
        Smart chunking with quality validation
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            return []
        
        chunks = self.splitter.split_text(text)
        
        documents = []
        for idx, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue
            
            doc = Document(
                doc_id=f"{metadata.get('source_hash', 'unknown')}_chunk_{idx}",
                text=chunk_text,
                source_file=metadata.get('source_file', 'unknown'),
                page_num=metadata.get('page_num', 0),
                chunk_idx=idx,
                doc_type=metadata.get('doc_type', 'text'),
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text),
                timestamp=datetime.now(timezone.utc).isoformat(),
                source_hash=metadata.get('source_hash', ''),
                metadata=metadata
            )
            documents.append(doc)
        
        logger.info(f"Chunked {len(text)} chars into {len(documents)} chunks")
        return documents


class TableExtractor:
    """
    Converts PDF tables to RAG-friendly text representations.
    Preserves structure and creates natural language descriptions.
    """
    
    def __init__(self):
        self.max_rows_to_process = 1000
    
    def extract_and_convert(self, table: List[List[str]]) -> str:
        """
        Convert table to natural language with structured formatting.
        
        Example input:
            [['Q', 'Revenue', 'Growth'],
             ['Q1', '$2.1B', '5%'],
             ['Q2', '$2.3B', '9%']]
        
        Output:
            "Financial table with columns: Q, Revenue, Growth
             Q1: Revenue=$2.1B, Growth=5%
             Q2: Revenue=$2.3B, Growth=9%"
        """
        if not table or len(table) < 2:
            return ""
        
        try:
            df = pd.DataFrame(table[1:], columns=table[0])
            
            # Header
            lines = [f"Table with {len(df)} rows and {len(df.columns)} columns"]
            lines.append(f"Columns: {', '.join(df.columns)}")
            lines.append("")
            
            # Row-by-row representation
            for idx, row in df.iterrows():
                row_str = " | ".join([
                    f"{col}={val}" for col, val in row.items()
                ])
                lines.append(f"Row {idx}: {row_str}")
            
            # Add summary statistics if numeric columns exist
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                lines.append("\nSummary statistics:")
                for col in numeric_cols:
                    try:
                        col_data = pd.to_numeric(df[col], errors='coerce')
                        if not col_data.isna().all():
                            lines.append(
                                f"  {col}: mean={col_data.mean():.2f}, "
                                f"min={col_data.min():.2f}, max={col_data.max():.2f}"
                            )
                    except Exception:
                        pass
            
            return "\n".join(lines)
        
        except Exception as e:
            logger.error(f"Error converting table: {e}")
            return str(table)  # Fallback


class PDFLoader:
    """
    Production-grade PDF loader with error handling, caching, and metadata.
    """
    
    def __init__(self, config: Settings):
        self.config = config
        self.chunker = SmartChunker(
            chunk_size=512,
            chunk_overlap=100
        )
        self.table_extractor = TableExtractor()
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
    
    def load_pdf(self, pdf_path: str) -> Dict[str, Document]:
        """
        Load single PDF with caching.
        
        Returns: {doc_id: Document, ...}
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise RAGError(f"PDF not found: {pdf_path}")
        
        # Check cache
        cache_key = self._get_cache_key(str(pdf_path))
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info(f"Loaded {pdf_path.name} from cache")
            return cached
        
        logger.info(f"Processing PDF: {pdf_path.name}")
        documents = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                file_hash = self._hash_file(str(pdf_path))
                
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text() or ""
                    
                    if text.strip():
                        text_docs = self.chunker.chunk(
                            text,
                            metadata={
                                'source_file': pdf_path.name,
                                'page_num': page_num,
                                'doc_type': 'text',
                                'source_hash': file_hash
                            }
                        )
                        for doc in text_docs:
                            documents[doc.doc_id] = doc
                    
                    # Extract tables
                    tables = page.extract_tables() or []
                    for table_idx, table in enumerate(tables):
                        table_text = self.table_extractor.extract_and_convert(table)
                        
                        table_docs = self.chunker.chunk(
                            table_text,
                            metadata={
                                'source_file': pdf_path.name,
                                'page_num': page_num,
                                'doc_type': 'table',
                                'table_idx': table_idx,
                                'source_hash': file_hash,
                                'original_table': table  # For debugging
                            }
                        )
                        for doc in table_docs:
                            documents[doc.doc_id] = doc
        
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            raise RAGError(f"Failed to load PDF: {e}")
        
        # Cache result
        self._save_cache(cache_key, documents)
        logger.info(f"Loaded {len(documents)} chunks from {pdf_path.name}")
        
        return documents
    
    def load_directory(self, directory: str) -> Dict[str, Document]:
        """Load all PDFs from directory"""
        dir_path = Path(directory)
        
        if not dir_path.is_dir():
            raise RAGError(f"Directory not found: {directory}")
        
        all_documents = {}
        pdf_files = list(dir_path.glob("**/*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_path in pdf_files:
            try:
                docs = self.load_pdf(str(pdf_path))
                all_documents.update(docs)
            except Exception as e:
                logger.warning(f"Skipped {pdf_path.name}: {e}")
        
        logger.info(f"Total loaded: {len(all_documents)} chunks")
        return all_documents
    
    def _hash_file(self, filepath: str) -> str:
        """Create hash of file for cache invalidation"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _get_cache_key(self, filepath: str) -> str:
        """Generate cache filename"""
        file_hash = hashlib.md5(filepath.encode()).hexdigest()
        return f"pdf_{file_hash}.json"
    
    def _load_cache(self, cache_key: str) -> Optional[Dict]:
        """Load from cache"""
        cache_path = self.cache_dir / cache_key
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    return {k: Document(**v) for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        return None
    
    def _save_cache(self, cache_key: str, documents: Dict[str, Document]):
        """Save to cache"""
        cache_path = self.cache_dir / cache_key
        try:
            with open(cache_path, 'w') as f:
                json.dump({k: v.to_dict() for k, v in documents.items()}, f)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")


class DataPipeline:
    """
    Production pipeline: load PDFs → chunk → verify → index → ready for RAG
    """
    
    def __init__(self, config: Settings):
        self.config = config
        self.loader = PDFLoader(config)
        self.documents = {}
    
    def ingest(self, pdf_source: str) -> int:
        """
        Load PDFs (single file or directory)
        
        Returns: count of loaded documents
        """
        pdf_path = Path(pdf_source)
        
        if pdf_path.is_file():
            self.documents.update(self.loader.load_pdf(str(pdf_path)))
        elif pdf_path.is_dir():
            self.documents.update(self.loader.load_directory(str(pdf_path)))
        else:
            raise RAGError(f"Invalid path: {pdf_source}")
        
        return len(self.documents)
    
    def get_documents_for_indexing(self) -> Dict[str, str]:
        """Convert documents to format expected by retriever"""
        return {doc.doc_id: doc.text for doc in self.documents.values()}
    
    def get_metadata(self) -> Dict:
        """Statistics about loaded documents"""
        return {
            "total_documents": len(self.documents),
            "total_words": sum(d.word_count for d in self.documents.values()),
            "total_chars": sum(d.char_count for d in self.documents.values()),
            "by_doc_type": {
                "text": len([d for d in self.documents.values() if d.doc_type == "text"]),
                "table": len([d for d in self.documents.values() if d.doc_type == "table"])
            },
            "source_files": list(set(d.source_file for d in self.documents.values()))
        }
