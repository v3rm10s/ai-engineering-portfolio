# benchmark.py
import sys
from tabulate import tabulate
from evaluators import RAGEvaluator
from dataset import GOLDEN_TEST_SUITE

def run_benchmark():
    print("=" * 70)
    print("🚀 STARTING ENTERPRISE RAG EVALUATION BENCHMARK")
    print("=" * 70)

    evaluator = RAGEvaluator(model_name="gemini-3.1-flash-lite")
    results = []
    
    # SLA Thresholds for Production Readiness
    SLA_THRESHOLD = 0.70

    for idx, test_case in enumerate(GOLDEN_TEST_SUITE, start=1):
        print(f"\n[{idx}/{len(GOLDEN_TEST_SUITE)}] Evaluating {test_case.test_id}...")
        print(f"Query: \"{test_case.query}\"")
        
        # 1. Faithfulness Check
        faith = evaluator.evaluate_faithfulness(test_case)
        print(f"  ├─ Faithfulness Score: {faith.score:.2f}")
        if faith.unsupported_claims:
            print(f"  │  └─ ⚠️ Unsupported Claims Detected:")
            for claim in faith.unsupported_claims:
                print(f"  │     • {claim}")
        
        # 2. Answer Relevance Check
        relevance = evaluator.evaluate_answer_relevance(test_case)
        print(f"  ├─ Relevance Score:    {relevance.score:.2f}")
        if relevance.redundant_or_tangential_info:
            print(f"  │  └─ ⚠️ Tangential Info:")
            for info in relevance.redundant_or_tangential_info:
                print(f"  │     • {info}")

        # 3. Context Precision Check
        precision = evaluator.evaluate_context_precision(test_case)
        print(f"  ├─ Precision Score:    {precision.score:.2f}")
        print(f"  │  └─ Chunk Relevance: {precision.chunk_relevance}")

        overall = round((faith.score + relevance.score + precision.score) / 3.0, 3)
        print(f"  └─ Overall Case Score: {overall:.2f}")

        results.append({
            "Test ID": test_case.test_id,
            "Faithfulness": f"{faith.score:.2f}",
            "Relevance": f"{relevance.score:.2f}",
            "Precision": f"{precision.score:.2f}",
            "Overall": f"{overall:.2f}",
            "Status": "PASS" if min(faith.score, relevance.score, precision.score) >= SLA_THRESHOLD else "FAIL"
        })

    # Summary Benchmark Report
    print("\n" + "=" * 70)
    print("📊 BENCHMARK SUMMARY REPORT")
    print("=" * 70)
    
    table_headers = ["Test ID", "Faithfulness", "Relevance", "Precision", "Overall", "Status"]
    table_data = [
        [r["Test ID"], r["Faithfulness"], r["Relevance"], r["Precision"], r["Overall"], r["Status"]]
        for r in results
    ]
    print(tabulate(table_data, headers=table_headers, tablefmt="fancy_grid"))

    # Aggregate Metrics
    avg_faith = sum(float(r["Faithfulness"]) for r in results) / len(results)
    avg_rel = sum(float(r["Relevance"]) for r in results) / len(results)
    avg_prec = sum(float(r["Precision"]) for r in results) / len(results)
    avg_total = sum(float(r["Overall"]) for r in results) / len(results)

    print("\n📈 AGGREGATE SYSTEM METRICS:")
    print(f"• Mean Faithfulness:     {avg_faith:.2f}")
    print(f"• Mean Answer Relevance: {avg_rel:.2f}")
    print(f"• Mean Context Precision:{avg_prec:.2f}")
    print(f"• System Benchmark Score: {avg_total:.2f}")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()