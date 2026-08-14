import os
from typing import List, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ddgs import DDGS
from langchain_core.documents import Document


load_dotenv()
api_key=os.getenv("GEMINI-API-KEY")

# 1. Initialize Mock Internal Documents
docs = [
    "Project Apollo Architecture: Apollo uses a microservices model with FastAPI, gRPC, and PostgreSQL.",
    "Company Remote Work Policy: Employees may work remotely up to 3 days per week with manager approval.",
    "Internal Deployment SOP: Production deployments require sign-off from QA and a passing staging smoke test."
]

class GradeDocument(BaseModel):
    binary_score: Literal["yes","no"] = Field(
        description="Whether the document is relevant to the question ('yes' or 'no')"
    )

class RetrievalEvaluator:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        llm = ChatGoogleGenerativeAI(model=model_name,temperature=0)
        self.structured_llm = llm.with_structured_output(GradeDocument)
    
        system_prompt = (
                "You are a strict retrieval evaluator assessing whether a retrieved document "
                "contains information relevant to answering the user question. "
                "Give a binary score 'yes' or 'no'."
            )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "User question:\n{question}\n\nRetrieved context:\n{context}")
        ])
        self.grader_chain = self.prompt |  self.structured_llm
    
    def grade_documents(self, question: str, documents: list) -> dict:
        filtered_docs = []
        scores = []
        
        for doc in documents:
            result = self.grader_chain.invoke({
                "question": question,
                "context": doc.page_content
            })
            score = result.binary_score.lower().strip()
            scores.append(score)
            if score == "yes":
                filtered_docs.append(doc)
        total = len(documents)
        yes_count = scores.count("yes")
        
        if total == 0 or yes_count == 0:
            confidence = "INCORRECT"
        elif yes_count == total:
            confidence = "CORRECT"
        else:
            confidence = "AMBIGUOUS"

        return {
            "confidence": confidence,
            "relevant_docs": filtered_docs,
            "raw_scores": scores
        }

class WebSearchTool:
    def __init__(self, max_results: int = 3):
        self.max_results = max_results

    def search(self, query: str) -> List[Document]:
        """Runs DuckDuckGo search and returns list of Documents."""
        results = []
        try:
            ddgs = DDGS()
            search_results = list(ddgs.text(query, max_results=self.max_results))
            for res in search_results:
                title = res.get("title", "")
                body = res.get("body", "")
                snippet = f"{title}: {body}"
                results.append(Document(page_content=snippet, metadata={"source": "web", "url": res.get("href", "")}))
        except Exception as e:
            print(f"[Warning] Web search failed: {e}")
        return results


class CorrectiveRAG:
    def __init__(self, retriever, evaluator: RetrievalEvaluator, model_name: str = "gemini-2.5-flash"):
        self.retriever = retriever
        self.evaluator = evaluator
        self.web_search = WebSearchTool(max_results=2)
        
        # Generator LLM
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
        
        gen_prompt = (
            "You are an assistant answering questions using the provided context.\n"
            "If the context comes from the web or internal docs, answer clearly and mention "
            "whether internal documents or web search informed your answer.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        self.answer_prompt = ChatPromptTemplate.from_template(gen_prompt)
        self.generator_chain = self.answer_prompt | self.llm

    def run(self, question: str) -> dict:
        print(f"\n==================================================")
        print(f"👉 Query: '{question}'")
        
        # Step 1: Initial Retrieval
        initial_docs = self.retriever.invoke(question)
        print(f"1. Retrieved {len(initial_docs)} internal chunk(s).")
        
        # Step 2: Evaluation
        evaluation = self.evaluator.grade_documents(question, initial_docs)
        confidence = evaluation["confidence"]
        relevant_internal = evaluation["relevant_docs"]
        print(f"2. Evaluator Decision: [{confidence}] (Scores: {evaluation['raw_scores']})")

        final_context = []

        # Step 3: Dynamic Routing
        if confidence == "CORRECT":
            print("3. Action: High internal confidence. Using filtered internal docs.")
            final_context = relevant_internal

        elif confidence == "INCORRECT":
            print("3. Action: Low internal confidence. Triggering Web Search Fallback...")
            web_docs = self.web_search.search(question)
            print(f"   Found {len(web_docs)} web search snippet(s).")
            final_context = web_docs

        elif confidence == "AMBIGUOUS":
            print("3. Action: Ambiguous/Partial context. Merging internal docs + Web Search...")
            web_docs = self.web_search.search(question)
            print(f"   Found {len(web_docs)} web search snippet(s).")
            final_context = relevant_internal + web_docs

        # Format context for generator
        context_str = "\n\n".join([f"[{getattr(doc, 'metadata', {}).get('source', 'internal')}] {doc.page_content}" for doc in final_context])

        # Step 4: Generation
        response = self.generator_chain.invoke({
            "question": question,
            "context": context_str
        })

        return {
            "question": question,
            "confidence": confidence,
            "num_context_used": len(final_context),
            "answer": response.content
        }

# 2. Build local in-memory Chroma vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma.from_texts(texts=docs, embedding=embeddings, collection_name="internal_kb")
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

print("Internal Knowledge Base initialized successfully.")

query = "How do we deploy to production?"
results = retriever.invoke(query)
print(f"\nRetrieved {len(results)} doc(s) for test query '{query}':")
for i, doc in enumerate(results, 1):
    print(f"[{i}] {doc.page_content}")
    
    evaluator = RetrievalEvaluator()

crag = CorrectiveRAG(retriever=retriever, evaluator=evaluator)

# Test Scenario A: Pure internal hit (or partial hit with fallback)
res_a = crag.run("What is the company remote work policy?")
print(f"\n💡 FINAL ANSWER:\n{res_a['answer']}")

# Test Scenario B: Missing internal knowledge (Triggers Web Search fallback)
res_b = crag.run("Who won the 2024 Super Bowl and what was the score?")
print(f"\n💡 FINAL ANSWER:\n{res_b['answer']}")