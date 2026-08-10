from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np

ROLE_HIERARCHY = {
    "viewer": 1,
    "analyst": 2,
    "admin": 3
}

@dataclass
class UserContext:
    user_id: str
    tenant_id: str
    role: str

@dataclass
class VectorDocument:
    doc_id: str
    tenant_id: str
    vector: np.ndarray
    payload: Dict[str, Any]
    required_role: str = "viewer"

class MultiTenantVectorStore:
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self._store: Dict[str, VectorDocument] = {}

    def insert(self, doc: VectorDocument) -> None:
        if len(doc.vector) != self.dimension:
            raise ValueError(f"Vector Dimension Mismatch. Expected {self.dimension}")
        self._store[doc.doc_id] = doc

    def count(self) -> int:
        return len(self._store)

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def search(self, query_vector: np.ndarray, user: UserContext, top_k: int = 3) -> List[Dict[str, Any]]:
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector dimension mismatch. Expected {self.dimension}")

        user_role_level = ROLE_HIERARCHY.get(user.role, 0)
        results = []

        for doc in self._store.values():
            # Guard 1: Tenant Isolation
            if doc.tenant_id != user.tenant_id:
                continue

            # Guard 2: RBAC Clearance
            doc_role_level = ROLE_HIERARCHY.get(doc.required_role, 0)
            if user_role_level < doc_role_level:
                continue

            # Calculate similarity score if both guards pass
            score = self._cosine_similarity(query_vector, doc.vector)
            results.append({
                "doc_id": doc.doc_id,
                "tenant_id": doc.tenant_id,
                "score": round(score, 4),
                "payload": doc.payload,
                "required_role": doc.required_role
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


if __name__ == "__main__":
    store = MultiTenantVectorStore(dimension=128)

    # Seed Document 1: Tenant Alpha - Public Viewer doc
    store.insert(VectorDocument(
        doc_id="alpha_public", tenant_id="tenant_alpha",
        vector=np.ones(128), payload={"title": "Alpha General Announcement"}, required_role="viewer"
    ))

    # Seed Document 2: Tenant Alpha - Confidential Admin doc
    store.insert(VectorDocument(
        doc_id="alpha_secret", tenant_id="tenant_alpha",
        vector=np.ones(128), payload={"title": "Alpha Q3 Financial Forecast"}, required_role="admin"
    ))

    # Seed Document 3: Tenant Beta - Secret doc
    store.insert(VectorDocument(
        doc_id="beta_secret", tenant_id="tenant_beta",
        vector=np.ones(128), payload={"title": "Beta Unreleased Product Roadmap"}, required_role="viewer"
    ))

    query_vec = np.ones(128)

    # Test Case 1: Tenant Beta Viewer tries to search
    beta_user = UserContext(user_id="u_beta", tenant_id="tenant_beta", role="viewer")
    beta_results = store.search(query_vec, user=beta_user)
    print(f"Beta User Results ({len(beta_results)} found):")
    for r in beta_results:
        print("  ->", r["doc_id"], r["payload"])

    # Test Case 2: Tenant Alpha Viewer tries to access Alpha Admin secret
    alpha_viewer = UserContext(user_id="u_alpha_v", tenant_id="tenant_alpha", role="viewer")
    alpha_v_results = store.search(query_vec, user=alpha_viewer)
    print(f"\nAlpha Viewer Results ({len(alpha_v_results)} found):")
    for r in alpha_v_results:
        print("  ->", r["doc_id"], r["payload"])