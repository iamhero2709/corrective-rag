"""
Production monitoring: performance metrics, health checks, alerts.
"""

import time
import logging
from typing import Dict, Optional, Any, Callable
from datetime import datetime, timedelta
from functools import wraps
import json
from collections import deque

import numpy as np

from src.logging_utils import setup_logger
from src.persistence import DocumentStore

logger = setup_logger(__name__)


class MetricsCollector:
    """
    Collects performance metrics with rolling window statistics.
    Supports percentile tracking, alerts, and anomaly detection.
    """
    
    def __init__(self, db: Optional[DocumentStore] = None, window_size: int = 1000):
        self.db = db
        self.window_size = window_size
        self.metrics = {}  # metric_name -> deque of values
    
    def record(self, metric_name: str, value: float, tags: Optional[Dict] = None):
        """Record a metric value"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = deque(maxlen=self.window_size)
        
        self.metrics[metric_name].append(value)
        
        # Log to database
        if self.db:
            self.db.log_metric(metric_name, value, tags)
    
    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric (mean, p50, p95, p99, etc.)"""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return {}
        
        values = list(self.metrics[metric_name])
        
        return {
            "count": len(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "p50": np.percentile(values, 50),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99),
        }
    
    def all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all metrics"""
        return {
            name: self.get_stats(name)
            for name in self.metrics.keys()
        }
    
    def detect_anomaly(self, metric_name: str, threshold_std: float = 2.0) -> bool:
        """
        Detect if latest value is anomalous (> threshold_std standard deviations from mean)
        """
        if metric_name not in self.metrics or len(self.metrics[metric_name]) < 10:
            return False
        
        values = list(self.metrics[metric_name])[:-1]  # Exclude latest
        latest = list(self.metrics[metric_name])[-1]
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return abs(latest - mean) > 1e-9
        
        z_score = abs(latest - mean) / std
        return z_score > threshold_std


class LatencyTracker:
    """Track and alert on latency issues"""
    
    def __init__(self, db: Optional[DocumentStore] = None, threshold_ms: float = 5000):
        self.db = db
        self.threshold_ms = threshold_ms
        self.collector = MetricsCollector(db)
    
    def start_timer(self) -> Callable:
        """Context manager for timing code blocks"""
        class Timer:
            def __init__(self, tracker):
                self.tracker = tracker
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, *args):
                elapsed_ms = (time.time() - self.start_time) * 1000
                self.tracker.record_latency(elapsed_ms)
        
        return Timer(self)
    
    def record_latency(self, latency_ms: float):
        """Record latency and check for anomalies"""
        self.collector.record("latency_ms", latency_ms)
        
        if latency_ms > self.threshold_ms:
            logger.warning(
                f"High latency detected: {latency_ms:.0f}ms (threshold: {self.threshold_ms:.0f}ms)"
            )
        
        # Anomaly detection
        if self.collector.detect_anomaly("latency_ms"):
            logger.warning("Latency anomaly detected")
    
    def get_stats(self) -> Dict:
        """Get latency statistics"""
        return self.collector.get_stats("latency_ms")


class HealthCheck:
    """
    Health check system with component status tracking.
    Used for readiness probes in production.
    """
    
    def __init__(self):
        self.components = {}  # component_name -> status
    
    def register_check(self, component: str, check_func: Callable[[], bool]):
        """Register a health check function"""
        self.components[component] = {
            "check_func": check_func,
            "last_check": None,
            "status": "unknown"
        }
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
            "overall_healthy": True
        }
        
        for component, info in self.components.items():
            try:
                status = info["check_func"]()
                results["components"][component] = {
                    "healthy": status,
                    "timestamp": datetime.utcnow().isoformat()
                }
                if not status:
                    results["overall_healthy"] = False
                    logger.warning(f"Health check failed for {component}")
            except Exception as e:
                logger.error(f"Error in health check for {component}: {e}")
                results["components"][component] = {
                    "healthy": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                results["overall_healthy"] = False
        
        return results


class RequestMonitor:
    """Monitor individual requests"""
    
    def __init__(self, db: Optional[DocumentStore] = None):
        self.db = db
        self.latency_tracker = LatencyTracker(db)
        self.request_count = 0
        self.error_count = 0
    
    def monitor_request(self, func: Callable) -> Callable:
        """Decorator to monitor request execution"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.request_count += 1
            
            with self.latency_tracker.start_timer() as timer:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    self.error_count += 1
                    logger.error(f"Request failed: {e}")
                    raise
        
        return wrapper
    
    def get_stats(self) -> Dict:
        """Get request statistics"""
        return {
            "total_requests": self.request_count,
            "errors": self.error_count,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0,
            "latency": self.latency_tracker.get_stats()
        }


class CachePerformanceMonitor:
    """Monitor cache hit rates and efficiency"""
    
    def __init__(self, db: Optional[DocumentStore] = None):
        self.db = db
        self.hits = 0
        self.misses = 0
    
    def record_hit(self):
        """Record cache hit"""
        self.hits += 1
        if self.db:
            self.db.log_metric("cache_hit", 1)
    
    def record_miss(self):
        """Record cache miss"""
        self.misses += 1
        if self.db:
            self.db.log_metric("cache_miss", 1)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        if total == 0:
            return {"hits": 0, "misses": 0, "hit_rate": 0}
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": 100 * self.hits / total,
            "total": total
        }


class RetrievalQualityMonitor:
    """Monitor retrieval quality metrics"""
    
    def __init__(self, db: Optional[DocumentStore] = None):
        self.db = db
        self.retrieved_chunks = []
        self.verification_results = []
    
    def record_retrieval(self, num_chunks: int, top_scores: list):
        """Record retrieval results"""
        self.retrieved_chunks.append({
            "num_chunks": num_chunks,
            "top_scores": top_scores,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        if self.db:
            self.db.log_metric("retrieved_chunks", float(num_chunks))
            if top_scores:
                self.db.log_metric("top_chunk_score", float(top_scores[0]))
    
    def record_verification(self, verdict: str, score: float):
        """Record verification results"""
        self.verification_results.append({
            "verdict": verdict,
            "score": score,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        if self.db:
            self.db.log_metric(f"verification_{verdict}", 1)
            self.db.log_metric("verification_score", score)
    
    def get_stats(self) -> Dict:
        """Get retrieval quality statistics"""
        if not self.verification_results:
            return {}
        
        verdicts = [r["verdict"] for r in self.verification_results]
        scores = [r["score"] for r in self.verification_results]
        
        return {
            "total_verifications": len(verdicts),
            "correct": verdicts.count("CORRECT"),
            "ambiguous": verdicts.count("AMBIGUOUS"),
            "incorrect": verdicts.count("INCORRECT"),
            "avg_score": np.mean(scores),
            "verification_rate": {
                "correct": 100 * verdicts.count("CORRECT") / len(verdicts),
                "ambiguous": 100 * verdicts.count("AMBIGUOUS") / len(verdicts),
                "incorrect": 100 * verdicts.count("INCORRECT") / len(verdicts),
            }
        }


class PerformanceDashboard:
    """
    Unified dashboard aggregating all monitoring data.
    """
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.health_check = HealthCheck()
        self.request_monitor = RequestMonitor()
        self.cache_monitor = CachePerformanceMonitor()
        self.retrieval_monitor = RetrievalQualityMonitor()
    
    def get_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "requests": self.request_monitor.get_stats(),
            "cache": self.cache_monitor.get_stats(),
            "retrieval": self.retrieval_monitor.get_stats(),
            "metrics": self.metrics_collector.all_stats(),
            "health": self.health_check.run_checks()
        }
    
    def print_summary(self):
        """Print human-readable summary"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("PERFORMANCE DASHBOARD")
        print("="*60)
        
        print("\nRequests:")
        req = summary["requests"]
        print(f"  Total: {req['total_requests']}")
        print(f"  Errors: {req['errors']} ({req['error_rate']*100:.1f}%)")
        
        print("\nCache:")
        cache = summary["cache"]
        print(f"  Hit Rate: {cache['hit_rate']:.1f}%")
        print(f"  Hits: {cache['hits']}, Misses: {cache['misses']}")
        
        print("\nRetrieval Quality:")
        ret = summary["retrieval"]
        if ret:
            print(f"  Correct: {ret.get('correct', 0)} ({ret.get('verification_rate', {}).get('correct', 0):.1f}%)")
            print(f"  Ambiguous: {ret.get('ambiguous', 0)} ({ret.get('verification_rate', {}).get('ambiguous', 0):.1f}%)")
            print(f"  Incorrect: {ret.get('incorrect', 0)} ({ret.get('verification_rate', {}).get('incorrect', 0):.1f}%)")
        
        print("\nLatency:")
        if "latency_ms" in summary["metrics"]:
            lat = summary["metrics"]["latency_ms"]
            print(f"  Mean: {lat['mean']:.0f}ms")
            print(f"  P95: {lat['p95']:.0f}ms")
            print(f"  P99: {lat['p99']:.0f}ms")
        
        print("\n" + "="*60)
    
    def export_json(self, filepath: str):
        """Export summary to JSON"""
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Dashboard exported to {filepath}")
