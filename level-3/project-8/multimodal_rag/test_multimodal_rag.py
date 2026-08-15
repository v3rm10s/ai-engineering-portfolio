from src.ingestion import DocumentElement, ElementType
from src.summarizer import ElementSummarizer
from src.indexer import MultimodalIndexer
from src.pipeline import MultimodalRAGPipeline

def main():
    print("=== Step 1: Ingesting Mixed Modality Corpus ===")
    raw_elements = [
        DocumentElement(
            id="text_macro_01",
            element_type=ElementType.TEXT,
            content="Federal Reserve interest rate policy remains hawkish through Q1 2026, causing market volatility across risk assets."
        ),
        DocumentElement(
            id="table_q1_stats",
            element_type=ElementType.TABLE,
            content=(
                "| Asset | Q1 Inflow ($B) | Max Drawdown (%) |\n"
                "|---|---|---|\n"
                "| Tech Equities | $14.2B | -8.4% |\n"
                "| Crypto Assets | $3.1B | -18.2% |\n"
                "| Treasury Bonds | $28.0B | -1.1% |"
            )
        ),
        DocumentElement(
            id="chart_technical_01",
            element_type=ElementType.IMAGE,
            content="data/sample_docs/chart.png"
        )
    ]

    print("\n=== Step 2: Generating Vector Summaries ===")
    summarizer = ElementSummarizer()
    enriched_elements = [summarizer.process_element(e) for e in raw_elements]

    print("\n=== Step 3: Indexing Elements into ChromaDB ===")
    indexer = MultimodalIndexer()
    indexer.index_elements(enriched_elements)

    print("\n=== Step 4: Initializing Multimodal RAG Pipeline ===")
    pipeline = MultimodalRAGPipeline(indexer=indexer)

    # Test Query 1: Visual reasoning on the candlestick chart
    query_1 = "What is the current last close price shown on the chart, and how is it behaving relative to the moving average line?"
    print(f"\n--- QUERY 1: {query_1} ---")
    answer_1 = pipeline.run(query_1, top_k=2)
    print("\n[RAG ANSWER 1]:")
    print(answer_1)

    print("\n" + "="*60 + "\n")

    # Test Query 2: Tabular reasoning
    query_2 = "Which asset class had the highest drawdown in Q1 and how much was it?"
    print(f"--- QUERY 2: {query_2} ---")
    answer_2 = pipeline.run(query_2, top_k=1)
    print("\n[RAG ANSWER 2]:")
    print(answer_2)

if __name__ == "__main__":
    main()