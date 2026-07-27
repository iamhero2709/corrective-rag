"""
LLM-based triple extraction for knowledge graph construction.
Replaces noisy spacy SVO parsing with structured LLM extraction.
"""

import json
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import re

from transformers import pipeline
import torch
from src.config import Settings

from src.logging_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class Triple:
    """Represents a (subject, relation, object) triple"""
    subject: str
    relation: str
    obj: str
    confidence: float = 1.0
    source_text: str = ""
    
    def to_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.relation, self.obj)


class LLMTripleExtractor:
    """
    Uses a small LLM to extract structured (S, R, O) triples from text.
    More reliable than SVO parsing for complex sentences.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
            max_length=512
        )
        
        self.extraction_prompt = """Extract all factual relationships from the text below.
Format each relationship as JSON: {"subject": "...", "relation": "...", "object": "..."}

Text: {text}

Return only valid JSON objects, one per line. If no relationships, return empty list.
"""
    
    def extract(self, text: str, max_triples: int = 10) -> List[Triple]:
        """
        Extract triples from text using LLM.
        
        Returns: List of (subject, relation, object) triples
        """
        if not text or len(text.strip()) < 10:
            return []
        
        # Truncate to reasonable length for processing
        text = text[:1000]
        
        prompt = self.extraction_prompt.format(text=text)
        
        try:
            response = self.pipe(prompt, num_return_sequences=1, do_sample=False)
            response_text = response[0]["generated_text"]
            
            # Parse JSON from response
            triples = self._parse_json_response(response_text, text)
            
            # Deduplicate and rank by confidence
            triples = self._deduplicate_triples(triples)
            
            return triples[:max_triples]
        
        except Exception as e:
            logger.error(f"Triple extraction failed: {e}")
            return []
    
    def extract_batch(self, texts: List[str], max_triples_per_text: int = 5) -> List[Triple]:
        """Extract triples from multiple texts"""
        all_triples = []
        for text in texts:
            triples = self.extract(text, max_triples_per_text)
            all_triples.extend(triples)
        return all_triples
    
    def _parse_json_response(self, response: str, source_text: str) -> List[Triple]:
        """Extract JSON objects from LLM response"""
        triples = []
        
        # Find JSON objects in response
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, response)
        
        for match in matches:
            try:
                data = json.loads(match)
                
                # Validate required fields
                if all(k in data for k in ['subject', 'relation', 'object']):
                    triple = Triple(
                        subject=str(data['subject']).strip(),
                        relation=str(data['relation']).strip(),
                        obj=str(data['object']).strip(),
                        confidence=float(data.get('confidence', 0.8)),
                        source_text=source_text
                    )
                    
                    # Basic validation
                    if self._is_valid_triple(triple):
                        triples.append(triple)
            
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
        
        return triples
    
    def _is_valid_triple(self, triple: Triple) -> bool:
        """Validate triple quality"""
        # Minimum length
        if len(triple.subject) < 2 or len(triple.relation) < 2 or len(triple.obj) < 2:
            return False
        
        # No identical subject and object (no self-loops usually)
        if triple.subject.lower() == triple.obj.lower():
            return False
        
        # Confidence threshold
        if triple.confidence < 0.5:
            return False
        
        return True
    
    def _deduplicate_triples(self, triples: List[Triple]) -> List[Triple]:
        """Remove duplicate triples, keep highest confidence"""
        seen = {}
        for triple in sorted(triples, key=lambda t: t.confidence, reverse=True):
            key = (triple.subject.lower(), triple.relation.lower(), triple.obj.lower())
            if key not in seen:
                seen[key] = triple
        return list(seen.values())


class RulesBasedTripleExtractor:
    """
    Fallback: simple rule-based extraction when LLM extraction fails.
    Uses regex patterns for common relationships.
    """
    
    def __init__(self):
        self.patterns = [
            # "X is a Y" → (X, is_a, Y)
            (r'(\w+)\s+is\s+a\s+(\w+(?:\s+\w+)?)', 'is_a'),
            
            # "X works at Y" → (X, works_at, Y)
            (r'(\w+)\s+(?:works|worked)\s+at\s+(.+?)(?:\.|,|$)', 'works_at'),
            
            # "X located in Y" → (X, located_in, Y)
            (r'(\w+)\s+(?:is\s+)?located\s+in\s+(.+?)(?:\.|,|$)', 'located_in'),
            
            # "X founded Y" → (X, founded, Y)
            (r'(\w+)\s+(?:founded|created)\s+(.+?)(?:\.|,|$)', 'founded'),
            
            # "X has Y" → (X, has, Y)
            (r'(\w+)\s+has\s+(.+?)(?:\.|,|$)', 'has'),
            
            # "X directed Y" → (X, directed, Y)
            (r'(\w+(?:\s+\w+)?)\s+directed\s+(.+?)(?:\.|,|$)', 'directed'),
            
            # "X wrote/authored Y" → (X, wrote, Y)
            (r'(\w+(?:\s+\w+)?)\s+(?:wrote|authored)\s+(.+?)(?:\.|,|$)', 'wrote'),
            
            # "X stars in Y" → (X, stars_in, Y)
            (r'(\w+(?:\s+\w+)?)\s+stars?\s+(?:in|as)\s+(.+?)(?:\.|,|$)', 'stars_in'),
            
            # "X won Y" → (X, won, Y)
            (r'(\w+(?:\s+\w+)?)\s+won\s+(.+?)(?:\.|,|$)', 'won'),
            
            # "X released in Y" → (X, released_in, Y)
            (r'(\w+(?:\s+\w+)?)\s+released\s+(?:in|on)\s+(.+?)(?:\.|,|$)', 'released_in'),
        ]
    
    def extract(self, text: str) -> List[Triple]:
        """Extract triples using regex patterns"""
        triples = []
        
        for pattern, relation in self.patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    triple = Triple(
                        subject=match.group(1).strip(),
                        relation=relation,
                        obj=match.group(2).strip(),
                        confidence=0.6,  # Lower confidence for rule-based
                        source_text=text
                    )
                    if self._is_valid_triple(triple):
                        triples.append(triple)
        
        return triples
    
    def _is_valid_triple(self, triple: Triple) -> bool:
        """Validate triple quality"""
        if len(triple.subject) < 2 or len(triple.obj) < 2:
            return False
        if triple.subject.lower() == triple.obj.lower():
            return False
        return True


class HybridTripleExtractor:
    """
    Combines LLM extraction with fallback rule-based extraction.
    Strategy: Try LLM first, fallback to rules if it fails or returns too few.
    """
    
    def __init__(self, config: Settings = None):
        try:
            self.llm_extractor = LLMTripleExtractor()
            self.use_llm = True
        except Exception as e:
            logger.warning(f"LLM extractor initialization failed: {e}, using rules only")
            self.use_llm = False
        
        self.rules_extractor = RulesBasedTripleExtractor()
    
    def extract(self, text: str, min_triples: int = 1) -> List[Triple]:
        """Extract triples with hybrid strategy"""
        triples = []
        
        # Try LLM first
        if self.use_llm:
            try:
                triples = self.llm_extractor.extract(text)
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}, falling back to rules")
        
        # Fallback: rules-based if insufficient
        if len(triples) < min_triples:
            rules_triples = self.rules_extractor.extract(text)
            triples.extend(rules_triples)
        
        # Deduplicate
        triples = self._deduplicate(triples)
        
        return triples
    
    def extract_batch(self, texts: List[str]) -> List[Triple]:
        """Extract triples from multiple texts"""
        all_triples = []
        for text in texts:
            triples = self.extract(text)
            all_triples.extend(triples)
        return all_triples
    
    def _deduplicate(self, triples: List[Triple]) -> List[Triple]:
        """Remove duplicates, prefer higher confidence"""
        seen = {}
        for triple in sorted(triples, key=lambda t: t.confidence, reverse=True):
            key = (triple.subject.lower(), triple.relation.lower(), triple.obj.lower())
            if key not in seen:
                seen[key] = triple
        return list(seen.values())


# Simple fallback: just SVO parsing (for emergency)
class SPACYSVOExtractor:
    """Fallback to spacy SVO parsing if LLM unavailable"""
    
    def __init__(self):
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spacy model not loaded, SVOExtractor will be limited")
            self.nlp = None
    
    def extract(self, text: str) -> List[Triple]:
        """Extract using spacy"""
        if not self.nlp:
            return []
        
        triples = []
        doc = self.nlp(text)
        
        for sent in doc.sents:
            # Find subject, verb, object
            subject = None
            verb = None
            obj = None
            
            for token in sent:
                if token.pos_ == "NOUN" and not subject:
                    subject = token.text
                elif token.pos_ == "VERB" and not verb:
                    verb = token.lemma_
                elif token.pos_ == "NOUN" and verb and not obj:
                    obj = token.text
            
            if subject and verb and obj:
                triple = Triple(
                    subject=subject,
                    relation=verb,
                    obj=obj,
                    confidence=0.5,
                    source_text=text
                )
                triples.append(triple)
        
        return triples
