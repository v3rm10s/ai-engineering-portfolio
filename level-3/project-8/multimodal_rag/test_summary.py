import os
from src.ingestion import DocumentElement, ElementType
from src.summarizer import ElementSummarizer

# Make sure GEMINI_API_KEY is exported in your environment
# export GEMINI_API_KEY="your-api-key"

def main():
    # Adjust path to where your screenshot is saved (e.g., data/sample_docs/chart.png)
    chart_path = "data/sample_docs/chart.png"
    
    element = DocumentElement(
        id="chart_001",
        element_type=ElementType.IMAGE,
        content=chart_path,
        metadata={"source": "market_snapshot.png"}
    )
    
    summarizer = ElementSummarizer()
    enriched_element = summarizer.process_element(element)
    
    print("\n--- GENERATED VISUAL SUMMARY ---")
    print(enriched_element.summary)

if __name__ == "__main__":
    main()