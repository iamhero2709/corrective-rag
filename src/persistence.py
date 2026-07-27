"""
Production persistence layer: SQLite/PostgreSQL-ready database for state management.
Handles: document indexing, query caching, performance metrics logging.
"""

import sqlite3
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import pickle
from src.logging_utils import setup_logger

from src.exceptions import RAGError

logger = setup_logger(__name__)


class DocumentStore:
    """
    Persistent storage for documents and their embeddings.
    Supports SQLite locally, PostgreSQL in production.
    """
    
    def __init__(self, db_path: str = "data/rag.db"):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(exist_ok=True, parents=True)
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        cursor = self.conn.cursor()
        
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source_file TEXT,
                page_num INTEGER,
                chunk_idx INTEGER,
                doc_type TEXT,  -- 'text' or 'table'
                word_count INTEGER,
                char_count INTEGER,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Embeddings table (FAISS index references)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id TEXT PRIMARY KEY,
                embedding BLOB,  -- pickled numpy array
                embedding_model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        """)
        
        # Query cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                query_hash TEXT PRIMARY KEY,
                query TEXT,
                answer TEXT,
                confidence TEXT,
                latency_ms REAL,
                trace JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)
        
        # Performance metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                value REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tags JSON
            )
        """)
        
        # Knowledge graph triples
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS triples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT,
                relation TEXT,
                object TEXT,
                confidence REAL,
                source_doc TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(subject, relation, object)
            )
        """)
        
        # Create indices for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_source ON documents(source_file)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_triple_subject ON triples(subject)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metric_name ON metrics(metric_name)")
        
        self.conn.commit()
        logger.info(f"Database initialized: {self.db_path}")
    
    # ==================== Documents ====================
    
    def insert_document(self, doc: Dict[str, Any]) -> bool:
        """Insert a document"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO documents 
                (doc_id, text, source_file, page_num, chunk_idx, doc_type, 
                 word_count, char_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc['doc_id'],
                doc['text'],
                doc.get('source_file'),
                doc.get('page_num'),
                doc.get('chunk_idx'),
                doc.get('doc_type', 'text'),
                doc.get('word_count', 0),
                doc.get('char_count', 0),
                json.dumps(doc.get('metadata', {}))
            ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")
            return False
    
    def insert_documents_batch(self, docs: List[Dict]) -> int:
        """Batch insert documents"""
        count = 0
        for doc in docs:
            if self.insert_document(doc):
                count += 1
        return count
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Retrieve a document by ID"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_documents_by_source(self, source_file: str) -> List[Dict]:
        """Get all documents from a source file"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM documents WHERE source_file = ? ORDER BY page_num, chunk_idx",
            (source_file,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM embeddings WHERE doc_id = ?", (doc_id,))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    def document_count(self) -> int:
        """Total document count"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        return cursor.fetchone()[0]
    
    # ==================== Embeddings ====================
    
    def store_embedding(self, doc_id: str, embedding: Any, model: str) -> bool:
        """Store embedding for document"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings (doc_id, embedding, embedding_model)
                VALUES (?, ?, ?)
            """, (doc_id, pickle.dumps(embedding), model))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
            return False
    
    def get_embedding(self, doc_id: str) -> Optional[Any]:
        """Retrieve embedding"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT embedding FROM embeddings WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        if row:
            return pickle.loads(row[0])
        return None
    
    # ==================== Query Cache ====================
    
    def cache_query_result(
        self,
        query: str,
        answer: str,
        confidence: str,
        latency_ms: float,
        trace: Dict = None,
        ttl_hours: int = 24
    ) -> bool:
        """Cache a query result"""
        try:
            query_hash = hash(query.lower()) % (2**63)
            expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO query_cache 
                (query_hash, query, answer, confidence, latency_ms, trace, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                query_hash,
                query,
                answer,
                confidence,
                latency_ms,
                json.dumps(trace or {}),
                expires_at
            ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to cache query: {e}")
            return False
    
    def get_cached_query(self, query: str) -> Optional[Dict]:
        """Retrieve cached query result if not expired"""
        try:
            query_hash = hash(query.lower()) % (2**63)
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM query_cache 
                WHERE query_hash = ? AND expires_at > datetime('now')
            """, (query_hash,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Failed to retrieve cached query: {e}")
            return None
    
    def clear_expired_cache(self) -> int:
        """Delete expired cache entries"""
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM query_cache WHERE expires_at <= datetime('now')"
        )
        self.conn.commit()
        return cursor.rowcount
    
    # ==================== Knowledge Graph ====================
    
    def store_triple(
        self,
        subject: str,
        relation: str,
        obj: str,
        confidence: float = 1.0,
        source_doc: str = ""
    ) -> bool:
        """Store a knowledge graph triple"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO triples (subject, relation, object, confidence, source_doc)
                VALUES (?, ?, ?, ?, ?)
            """, (subject, relation, obj, confidence, source_doc))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store triple: {e}")
            return False
    
    def store_triples_batch(self, triples: List[Dict]) -> int:
        """Batch store triples"""
        count = 0
        for triple in triples:
            if self.store_triple(
                triple['subject'],
                triple['relation'],
                triple['obj'],
                triple.get('confidence', 1.0),
                triple.get('source_doc', '')
            ):
                count += 1
        return count
    
    def get_triples_by_subject(self, subject: str) -> List[Dict]:
        """Get all triples with subject"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM triples WHERE subject = ? ORDER BY confidence DESC",
            (subject,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_triples_by_relation(self, relation: str) -> List[Dict]:
        """Get all triples with relation"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM triples WHERE relation = ? ORDER BY confidence DESC",
            (relation,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def triple_count(self) -> int:
        """Total triple count"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM triples")
        return cursor.fetchone()[0]
    
    # ==================== Metrics ====================
    
    def log_metric(self, name: str, value: float, tags: Dict = None):
        """Log a performance metric"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO metrics (metric_name, value, tags)
                VALUES (?, ?, ?)
            """, (name, value, json.dumps(tags or {})))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to log metric: {e}")
    
    def get_metrics(self, name: str, hours: int = 24) -> List[Dict]:
        """Get metrics from last N hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM metrics 
            WHERE metric_name = ? AND timestamp > ?
            ORDER BY timestamp DESC
        """, (name, cutoff))
        return [dict(row) for row in cursor.fetchall()]
    
    # ==================== Health ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            "documents": self.document_count(),
            "triples": self.triple_count(),
            "db_path": self.db_path,
            "db_size_mb": Path(self.db_path).stat().st_size / (1024 * 1024)
        }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
