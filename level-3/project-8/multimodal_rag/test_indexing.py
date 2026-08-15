from src.ingestion import DocumentElement, ElementType
from src.summarizer import ElementSummarizer
from src.indexer import MultimodalIndexer

def main():
    # 1. Prepare mixed modality data
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

    # 2. Summarize visual & table elements
    summarizer = ElementSummarizer(model_name="gemini-3.6-flash") # Use your working model name
    enriched_elements = [summarizer.process_element(e) for e in raw_elements]

    # 3. Store into Vector DB
    indexer = MultimodalIndexer()
    indexer.index_elements(enriched_elements)

    # 4. Test Semantic Retrieval for the chart
    test_query = "What is the price action and moving average support level?"
    print(f"\n--- QUERY: '{test_query}' ---")
    
    results = indexer.search(test_query, top_k=1)
    for res in results:
        print(f"\n[Retrieved Element ID]: {res.id}")
        print(f"[Type]: {res.element_type.value}")
        print(f"[Content / Path]: {res.content}")
        print(f"[Summary Preview]: {res.summary[:200]}...")

if __name__ == "__main__":
    main()