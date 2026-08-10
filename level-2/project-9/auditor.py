import numpy as np

def compute_centroid(vectors: np.ndarray) -> np.ndarray:
    """
    Computes the geometric center (mean vector) across a matrix of vectors.
    vectors shape: (N, dim)
    Returns shape: (dim,)
    """
    return np.mean(vectors, axis=0)

def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    dot_prod = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Simple guard against division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 1.0
        
    cosine_sim = dot_prod / (norm_v1 * norm_v2)
    return float(1.0 - cosine_sim)

class EmbeddingAuditor:
    def __init__(self):
        self.centroid = None
        self.p95_threshold = None
        self.p99_threshold = None
        self.baseline_distances = None

    def fit_baseline(self, baseline_vectors: np.ndarray):
        self.centroid = compute_centroid(baseline_vectors)
        self.baseline_distances = np.array([
            cosine_distance(vec, self.centroid) for vec in baseline_vectors
        ])
        self.p95_threshold = float(np.percentile(self.baseline_distances, 95))
        self.p99_threshold = float(np.percentile(self.baseline_distances, 99))
        
        print("=== Baseline Registration Complete ===")
        print(f"Baseline Count: {len(baseline_vectors)} vectors")
        print(f"Mean Baseline Distance: {np.mean(self.baseline_distances):.4f}")
        print(f"P95 Perimeter Threshold: {self.p95_threshold:.4f}")
        print(f"P99 Perimeter Threshold: {self.p99_threshold:.4f}\n")

    def audit_query(self, query_vector: np.ndarray) -> dict:
        """
        Audits a single incoming query vector against the baseline domain perimeter.
        """
        dist = cosine_distance(query_vector, self.centroid)
        
        if dist <= self.p95_threshold:
            status = "IN_DISTRIBUTION"
        elif dist <= self.p99_threshold:
            status = "MILD_DRIFT"
        else:
            status = "OUT_OF_DISTRIBUTION"
            
        return {
            "distance_to_centroid": round(dist, 5),
            "status": status
        }
        
    def generate_health_report(self, query_batch: np.ndarray) -> dict:
        """
        Audits a batch of production queries and produces an aggregated vector health report.
        """
        total = len(query_batch)
        if total == 0:
            return {"error": "Empty query batch"}

        statuses = [self.audit_query(q)["status"] for q in query_batch]
        distances = [cosine_distance(q, self.centroid) for q in query_batch]

        in_dist_count = statuses.count("IN_DISTRIBUTION")
        mild_drift_count = statuses.count("MILD_DRIFT")
        ood_count = statuses.count("OUT_OF_DISTRIBUTION")

        ood_rate = (ood_count / total) * 100

        if ood_rate > 25.0:
            alert_level = "CRITICAL"
        elif ood_rate > 10.0:
            alert_level = "WARNING"
        else:
            alert_level = "HEALTHY"

        report = {
            "total_queries": total,
            "avg_distance": round(float(np.mean(distances)), 5),
            "in_distribution_pct": round((in_dist_count / total) * 100, 2),
            "mild_drift_pct": round((mild_drift_count / total) * 100, 2),
            "ood_pct": round(ood_rate, 2),
            "system_health": alert_level
        }
        return report


# --- Quick Verification ---
if __name__ == "__main__":
    np.random.seed(42)
    domain_base = np.array([0.8, 0.5, 0.2, 0.1])
    noise = np.random.normal(0, 0.05, (100, 4))
    baseline_vectors = domain_base + noise

    auditor = EmbeddingAuditor()
    auditor.fit_baseline(baseline_vectors)

    # 2. Simulate production query stream with heavy domain drift (e.g., user shift)
    id_queries = domain_base + np.random.normal(0, 0.04, (60, 4))
    drift_queries = np.random.normal(0.5, 0.2, (40, 4)) # 40% drift queries

    # Simulated Incoming Production Queries
    production_batch = np.vstack([id_queries, drift_queries])

    # 3. Generate Health Report
    report = auditor.generate_health_report(production_batch)
    
    print("========================================")
    print("      VECTOR HEALTH AUDIT REPORT        ")
    print("========================================")
    for key, val in report.items():
        print(f"  {key.upper().replace('_', ' ')}: {val}")
    print("========================================")