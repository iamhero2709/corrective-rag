"""
Production evaluation harness for HotpotQA, ASQA, PopQA benchmarks.
Supports ablation configs A0-A4 for research paper.
"""

import logging
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
import re
import string

from datasets import load_dataset
import numpy as np

from src.pipeline import CorrectiveRAG
from src.config import Settings
from src.logging_utils import setup_logger
from src.persistence import DocumentStore

logger = setup_logger(__name__)


@dataclass
class EvaluationResult:
    """Single evaluation result"""
    question: str
    predicted_answer: str
    ground_truth: str
    exact_match: bool
    f1_score: float
    confidence: str
    latency_ms: float
    config: str


@dataclass
class BenchmarkResults:
    """Aggregate benchmark results"""
    config: str
    dataset: str
    num_samples: int
    exact_match: float  # Percentage
    f1_score: float
    confidence_distribution: Dict[str, int]
    latency_mean: float
    latency_p95: float
    per_question_results: List[EvaluationResult] = None


class TextNormalizer:
    """Normalize text for EM/F1 comparison"""
    
    @staticmethod
    def extract_short_answer(prediction: str) -> str:
        """Extract short answer from verbose prediction (no ground truth access)"""
        pred = prediction.strip()
        
        # If prediction is already short, return as-is
        if len(pred.split()) <= 5:
            return pred
        
        # Strip common prefixes
        for prefix in ["The answer is ", "The answer is: ", "Answer: ", "Answer is "]:
            if pred.lower().startswith(prefix.lower()):
                pred = pred[len(prefix):]
                break
        
        # Strip trailing punctuation
        pred = pred.rstrip(".")
        
        # Take first sentence (likely the direct answer)
        first_sentence = re.split(r'[.\n]', pred)[0].strip()
        if len(first_sentence.split()) <= 8:
            return first_sentence
        
        # Take first clause (up to comma)
        first_clause = pred.split(",")[0].strip()
        if len(first_clause.split()) <= 6:
            return first_clause
        
        # Return first 5 words as fallback
        return " ".join(pred.split()[:5])
    
    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for fair evaluation"""
        # Lowercase
        text = text.lower()
        
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        
        # Remove punctuation
        text = ''.join(ch for ch in text if ch not in string.punctuation)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def f1_score(prediction: str, ground_truth: str) -> float:
        """Calculate F1 score (token-level)"""
        pred_tokens = set(TextNormalizer.normalize(prediction).split())
        truth_tokens = set(TextNormalizer.normalize(ground_truth).split())
        
        if not pred_tokens and not truth_tokens:
            return 1.0
        if not pred_tokens or not truth_tokens:
            return 0.0
        
        common = pred_tokens & truth_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> bool:
        """Check exact match (after normalization)"""
        return TextNormalizer.normalize(prediction) == TextNormalizer.normalize(ground_truth)


class HotpotQAEvaluator:
    """Evaluator for HotpotQA dataset"""
    
    def __init__(self, rag: CorrectiveRAG, db: DocumentStore, subset: str = "distractor"):
        """
        subset: 'distractor' (harder, requires reasoning), 'fullwiki' (even harder)
        """
        self.rag = rag
        self.db = db
        self.subset = subset
        self.dataset = None
        self.noise_paragraphs = {}  # title -> text pool for padding
        self._load_dataset()
        self._load_noise_pool()
    
    def _load_dataset(self):
        """Load HotpotQA dataset"""
        try:
            self.dataset = load_dataset("hotpotqa/hotpot_qa", self.subset, split="validation")
            logger.info("Loaded HotpotQA (%s): %d samples", self.subset, len(self.dataset))
        except Exception as e:
            logger.error("Failed to load HotpotQA: %s", e)

    def _load_noise_pool(self, pool_size: int = 200):
        """Load noise paragraphs from training set for realistic retrieval"""
        try:
            train = load_dataset("hotpotqa/hotpot_qa", self.subset, split="train")
            # Sample random questions and collect their context paragraphs as noise
            import random
            indices = random.sample(range(len(train)), min(pool_size, len(train)))
            for idx in indices:
                sample = train[idx]
                for title, sents in zip(sample['context']['title'], sample['context']['sentences']):
                    if title not in self.noise_paragraphs:
                        self.noise_paragraphs[title] = " ".join(sents)
            logger.info("Loaded %d noise paragraphs for retrieval padding", len(self.noise_paragraphs))
        except Exception as e:
            logger.warning("Could not load noise pool: %s", e)
    
    def evaluate(self, num_samples: Optional[int] = None) -> BenchmarkResults:
        """Evaluate RAG on HotpotQA"""
        if not self.dataset:
            raise RuntimeError("Dataset not loaded")
        
        samples = self.dataset.select(range(min(num_samples or 100, len(self.dataset))))
        results = []
        
        # Save original retriever state so we can restore after evaluation
        retriever = self.rag.retriever
        orig_chunks = list(retriever.chunks)
        orig_index = retriever.index
        
        # Create a fresh FAISS index for benchmarking (we'll rebuild per-question)
        import faiss
        dim = orig_index.d if orig_index is not None else 384
        bench_index = faiss.IndexFlatIP(dim)
        retriever.index = bench_index
        retriever.chunks = []
        
        try:
            for i, sample in enumerate(samples):
                if i % 10 == 0:
                    logger.info("Progress: %d/%d", i, len(samples))
                
                question = sample['question']
                ground_truth = sample['answer']
                
                # Index this question's context paragraphs + noise padding
                titles = sample['context']['title']
                sentences = sample['context']['sentences']
                docs = {}
                for title, sents in zip(titles, sentences):
                    docs[title] = " ".join(sents)
                # Add noise paragraphs to simulate realistic retrieval
                noise_titles = [t for t in self.noise_paragraphs if t not in docs]
                import random
                noise_sample = random.sample(noise_titles, min(30, len(noise_titles)))
                for title in noise_sample:
                    docs[title] = self.noise_paragraphs[title]
                retriever.add_documents(docs)
                
                # Query RAG
                try:
                    rag_result = self.rag.run(question)
                    predicted_answer = rag_result.answer
                    confidence = rag_result.confidence
                    latency = rag_result.latency_s * 1000
                except Exception as e:
                    logger.warning("RAG failed for question %d: %s", i, e)
                    predicted_answer = ""
                    confidence = "FAILED"
                    latency = 0
                
                # Reset retriever for next question
                retriever.index = faiss.IndexFlatIP(dim)
                retriever.chunks = []
                
                # Extract short answer for better EM/F1 matching
                short_answer = TextNormalizer.extract_short_answer(predicted_answer)
                em = TextNormalizer.exact_match(short_answer, ground_truth)
                f1 = TextNormalizer.f1_score(short_answer, ground_truth)
                
                result = EvaluationResult(
                    question=question,
                    predicted_answer=predicted_answer,
                    ground_truth=ground_truth,
                    exact_match=em,
                    f1_score=f1,
                    confidence=confidence,
                    latency_ms=latency,
                    config=getattr(self.rag, 'config_name', 'unknown')
                )
                results.append(result)
                
                self.db.log_metric(
                    "hotpot_qa_f1",
                    f1,
                    {"config": result.config, "confidence": confidence}
                )
        finally:
            # Restore original retriever state
            retriever.index = orig_index
            retriever.chunks = orig_chunks
        
        # Aggregate results
        exact_matches = sum(1 for r in results if r.exact_match)
        f1_scores = [r.f1_score for r in results]
        confidence_dist = defaultdict(int)
        for r in results:
            confidence_dist[r.confidence] += 1
        latencies = [r.latency_ms for r in results if r.latency_ms > 0]
        
        return BenchmarkResults(
            config=getattr(self.rag, 'config_name', 'unknown'),
            dataset="HotpotQA",
            num_samples=len(results),
            exact_match=100 * exact_matches / len(results) if results else 0,
            f1_score=np.mean(f1_scores) if f1_scores else 0,
            confidence_distribution=dict(confidence_dist),
            latency_mean=np.mean(latencies) if latencies else 0,
            latency_p95=np.percentile(latencies, 95) if latencies else 0,
            per_question_results=results
        )


class ASQAEvaluator:
    """Evaluator for ASQA — queries without context (answer from model's parametric knowledge)"""
    
    def __init__(self, rag: CorrectiveRAG, db: DocumentStore):
        self.rag = rag
        self.db = db
        self.dataset = None
        self._load_dataset()
    
    def _load_dataset(self):
        """Load ASQA dataset"""
        try:
            self.dataset = load_dataset("din0s/asqa", split="dev")
            logger.info("Loaded ASQA: %d samples", len(self.dataset))
        except Exception as e:
            logger.error("Failed to load ASQA: %s", e)
    
    def evaluate(self, num_samples: Optional[int] = None) -> BenchmarkResults:
        """Evaluate RAG on ASQA"""
        if not self.dataset:
            raise RuntimeError("Dataset not loaded")
        
        samples = self.dataset.select(range(min(num_samples or 50, len(self.dataset))))
        results = []
        
        for i, sample in enumerate(samples):
            if i % 10 == 0:
                logger.info("Progress: %d/%d", i, len(samples))
            
            question = sample['ambiguous_question']
            qa_pairs = sample.get('qa_pairs', [])
            ground_truth = qa_pairs[0]['answer'] if qa_pairs else ''
            
            try:
                rag_result = self.rag.run(question)
                predicted_answer = rag_result.answer
                confidence = rag_result.confidence
                latency = rag_result.latency_s * 1000
            except Exception as e:
                logger.warning("RAG failed for ASQA question %d: %s", i, e)
                predicted_answer = ""
                confidence = "FAILED"
                latency = 0
            
            em = TextNormalizer.exact_match(predicted_answer, ground_truth)
            f1 = TextNormalizer.f1_score(predicted_answer, ground_truth)
            
            result = EvaluationResult(
                question=question,
                predicted_answer=predicted_answer,
                ground_truth=ground_truth,
                exact_match=em,
                f1_score=f1,
                confidence=confidence,
                latency_ms=latency,
                config=getattr(self.rag, 'config_name', 'unknown')
            )
            results.append(result)
        
        # Aggregate
        exact_matches = sum(1 for r in results if r.exact_match)
        f1_scores = [r.f1_score for r in results]
        confidence_dist = defaultdict(int)
        for r in results:
            confidence_dist[r.confidence] += 1
        latencies = [r.latency_ms for r in results if r.latency_ms > 0]
        
        return BenchmarkResults(
            config=getattr(self.rag, 'config_name', 'unknown'),
            dataset="ASQA",
            num_samples=len(results),
            exact_match=100 * exact_matches / len(results) if results else 0,
            f1_score=np.mean(f1_scores) if f1_scores else 0,
            confidence_distribution=dict(confidence_dist),
            latency_mean=np.mean(latencies) if latencies else 0,
            latency_p95=np.percentile(latencies, 95) if latencies else 0,
            per_question_results=results
        )


class PopQAEvaluator:
    """Evaluator for PopQA (Popular knowledge QA)"""
    
    def __init__(self, rag: CorrectiveRAG, db: DocumentStore):
        self.rag = rag
        self.db = db
        self.dataset = None
        self._load_dataset()
    
    def _load_dataset(self):
        """Load PopQA dataset"""
        try:
            self.dataset = load_dataset("akariasai/PopQA", split="test")
            logger.info(f"Loaded PopQA: {len(self.dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to load PopQA: {e}")
    
    def evaluate(self, num_samples: Optional[int] = None) -> BenchmarkResults:
        """Evaluate RAG on PopQA"""
        if not self.dataset:
            raise RuntimeError("Dataset not loaded")
        
        samples = self.dataset.select(range(min(num_samples or 100, len(self.dataset))))
        results = []
        
        for i, sample in enumerate(samples):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(samples)}")
            
            question = sample['question']
            possible_answers = sample.get('possible_answers', '[]')
            try:
                answers_list = json.loads(possible_answers) if isinstance(possible_answers, str) else possible_answers
                ground_truth = answers_list[0] if answers_list else sample.get('obj', '')
            except (json.JSONDecodeError, IndexError):
                ground_truth = sample.get('obj', '')
            
            try:
                rag_result = self.rag.run(question)
                predicted_answer = rag_result.answer
                confidence = rag_result.confidence
                latency = rag_result.latency_s * 1000
            except Exception as e:
                logger.warning(f"RAG failed for PopQA question {i}: {e}")
                predicted_answer = ""
                confidence = "FAILED"
                latency = 0
            
            em = TextNormalizer.exact_match(predicted_answer, ground_truth)
            f1 = TextNormalizer.f1_score(predicted_answer, ground_truth)
            
            result = EvaluationResult(
                question=question,
                predicted_answer=predicted_answer,
                ground_truth=ground_truth,
                exact_match=em,
                f1_score=f1,
                confidence=confidence,
                latency_ms=latency,
                config=getattr(self.rag, 'config_name', 'unknown')
            )
            results.append(result)
        
        # Aggregate
        exact_matches = sum(1 for r in results if r.exact_match)
        f1_scores = [r.f1_score for r in results]
        confidence_dist = defaultdict(int)
        for r in results:
            confidence_dist[r.confidence] += 1
        latencies = [r.latency_ms for r in results if r.latency_ms > 0]
        
        return BenchmarkResults(
            config=getattr(self.rag, 'config_name', 'unknown'),
            dataset="PopQA",
            num_samples=len(results),
            exact_match=100 * exact_matches / len(results) if results else 0,
            f1_score=np.mean(f1_scores) if f1_scores else 0,
            confidence_distribution=dict(confidence_dist),
            latency_mean=np.mean(latencies) if latencies else 0,
            latency_p95=np.percentile(latencies, 95) if latencies else 0,
            per_question_results=results
        )


class AblationStudy:
    """Run ablation study A0-A4"""
    
    # Ablation configs
    CONFIGS = {
        "A0": {"embed": False, "nli": False, "hrr": False},  # Vanilla
        "A1": {"embed": True, "nli": False, "hrr": False},   # Embed only
        "A2": {"embed": True, "nli": True, "hrr": False},    # Embed + NLI (~ CRAG)
        "A3": {"embed": True, "nli": True, "hrr": True},     # Full system
        "A4": {"embed": True, "nli": False, "hrr": True},    # Embed + HRR (isolates HRR)
    }
    
    def __init__(self, dataset: str = "hotpot_qa", num_samples: int = 100):
        self.dataset = dataset
        self.num_samples = num_samples
        self.results = {}
    
    def run(self, rag: CorrectiveRAG, db: DocumentStore):
        """Run ablation study across all configs"""
        for config_name, config_settings in self.CONFIGS.items():
            logger.info(f"Running {config_name}: {config_settings}")
            
            # Update RAG config
            rag.verifier.use_embedding = config_settings["embed"]
            rag.verifier.use_entailment = config_settings["nli"]
            rag.verifier.use_structural = config_settings["hrr"]
            rag.config_name = config_name
            
            # Evaluate
            if self.dataset == "hotpot_qa":
                evaluator = HotpotQAEvaluator(rag, db)
            elif self.dataset == "asqa":
                evaluator = ASQAEvaluator(rag, db)
            else:  # popqa
                evaluator = PopQAEvaluator(rag, db)
            
            result = evaluator.evaluate(self.num_samples)
            self.results[config_name] = result
            
            logger.info(
                f"{config_name}: EM={result.exact_match:.1f}%, "
                f"F1={result.f1_score:.3f}, Latency={result.latency_mean:.0f}ms"
            )
        
        return self.results
    
    def print_table(self):
        """Print results as table"""
        print(f"\n{'Config':<8} {'EM %':<8} {'F1':<8} {'Latency ms':<12}")
        print("-" * 40)
        for config, result in self.results.items():
            print(
                f"{config:<8} {result.exact_match:>6.1f}% "
                f"{result.f1_score:>6.3f} {result.latency_mean:>10.0f}"
            )
    
    def save_results(self, filepath: str):
        """Save results to JSON"""
        data = {}
        for config, result in self.results.items():
            data[config] = {
                "exact_match": result.exact_match,
                "f1_score": result.f1_score,
                "latency_mean": result.latency_mean,
                "latency_p95": result.latency_p95,
                "num_samples": result.num_samples,
                "confidence_dist": result.confidence_distribution
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
