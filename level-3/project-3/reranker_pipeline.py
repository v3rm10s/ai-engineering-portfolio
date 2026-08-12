from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch

# 1. Sample Corpus (Documents)
documents = [
    "To reset your password, click on 'Forgot Password' on the login screen and follow the instructions sent to your email.",
    "Password security guidelines require at least 12 characters, including uppercase, lowercase, numbers, and symbols.",
    "If you cannot log in, check if your account is locked due to multiple failed password attempts.",
    "Our refund policy allows returns within 30 days of purchase with a valid receipt.",
    "To change your billing email, navigate to Account Settings > Billing > Contact Information.",
    "You can update your account password under Account Settings > Security > Change Password.",
    "Two-factor authentication (2FA) adds an extra layer of security to your account login process."
]

# Query with subtle distinction (Looking for HOW to change an existing password vs resetting a lost one)
query = "How do I change my account password?"

print(f"Query: '{query}'\n")

print("--- STAGE 1: Bi-Encoder Retrieval (all-MiniLM-L6-v2) ---")
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')

doc_embeddings  =  bi_encoder.encode(documents, convert_to_tensor =True)
query_embedding = bi_encoder.encode(query, convert_to_tensor = True)

hits = util.semantic_search(query_embedding, doc_embeddings, top_k=5)[0]

candidate_docs = []
print("Top 5 Candidates from Stage 1:")
for rank, hit in enumerate(hits):
    doc_id = hit['corpus_id']
    score = hit['score']
    candidate_docs.append(documents[doc_id])
    print(f"Rank {rank + 1} | Score: {score:.4f} | Doc #{doc_id}: \"{documents[doc_id]}\"")

print("\n--- STAGE 2: Cross-Encoder Re-Ranking (ms-marco-MiniLM-L-6-v2) ---")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

pairs = [[query, doc] for doc in candidate_docs]

cross_scores = cross_encoder.predict(pairs)

reranked = sorted(zip(cross_scores,candidate_docs), key=lambda x: x[0], reverse =True)

print("Top 3 Final Results After Re-Ranking:")
for rank, (score, doc) in enumerate(reranked[:3]):
    print(f"Rank {rank + 1} | Score: {score:.4f} | Doc: \"{doc}\"")