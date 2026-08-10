import pytest
import numpy as np
from multi_tenant_vector_store import MultiTenantVectorStore, VectorDocument, UserContext

@pytest.fixture
def seeded_store():
    """Fixture providing a vector store populated with multi-tenant data."""
    store = MultiTenantVectorStore(dimension=4)
    
    # Tenant 1 Docs
    store.insert(VectorDocument("t1_public", "tenant_1", np.array([1.0, 0.0, 0.0, 0.0]), {"name": "T1 Public"}, "viewer"))
    store.insert(VectorDocument("t1_admin", "tenant_1", np.array([1.0, 0.0, 0.0, 0.0]), {"name": "T1 Admin Secret"}, "admin"))
    
    # Tenant 2 Docs (Identical vectors to attempt maximum cross-tenant similarity match)
    store.insert(VectorDocument("t2_public", "tenant_2", np.array([1.0, 0.0, 0.0, 0.0]), {"name": "T2 Public"}, "viewer"))
    
    return store

def test_cross_tenant_isolation_attack(seeded_store):
    """Attack Simulation: Tenant 2 user attempts to fetch Tenant 1's data."""
    attacker = UserContext(user_id="attacker_99", tenant_id="tenant_2", role="admin")
    query_vector = np.array([1.0, 0.0, 0.0, 0.0])  # Perfect match for T1 docs
    
    results = seeded_store.search(query_vector, user=attacker)
    
    # 1. Assert all returned results have tenant_id == "tenant_2"
    for r in results:
        assert r["tenant_id"] == "tenant_2", f"Leak detected! Found doc from tenant {r['tenant_id']}"
    # 2. Assert no document with doc_id starting with "t1_" is in results
    returned_doc_ids = [r["doc_id"] for r in results]
    assert "t1_public" not in returned_doc_ids
    assert "t1_admin" not in returned_doc_ids

def test_rbac_unauthorized_access_prevented(seeded_store):
    """Test that viewer cannot see admin documents within same tenant."""
    viewer = UserContext(user_id="user_v", tenant_id="tenant_1", role="viewer")
    query_vector = np.array([1.0, 0.0, 0.0, 0.0])
    
    results = seeded_store.search(query_vector, user=viewer)
    
    # 1. Assert len(results) == 1
    assert len(results) == 1, f"Expected 1 document, but got {len(results)}"
    # 2. Assert results[0]["doc_id"] == "t1_public"
    assert results[0]["doc_id"] == "t1_public"

def test_rbac_admin_elevated_access(seeded_store):
    """Test that admin can see all docs within their tenant."""
    admin = UserContext(user_id="user_a", tenant_id="tenant_1", role="admin")
    query_vector = np.array([1.0, 0.0, 0.0, 0.0])
    
    results = seeded_store.search(query_vector, user=admin)
    
    # 1. Assert len(results) == 2
    assert len(results) == 2, f"Expected 2 documents, but got {len(results)}"
    