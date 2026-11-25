import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_INDEX_PATH = BASE_DIR / "data" / "faiss_index_v2"

_VECTOR_STORE: FAISS | None = None
_GRAPH: Neo4jGraph | None = None


def get_vector_store() -> FAISS:
    """Load and cache the local FAISS index."""
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        embeddings = OpenAIEmbeddings()
        _VECTOR_STORE = FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return _VECTOR_STORE


def get_graph() -> Neo4jGraph:
    """Connect to and cache the Neo4j graph instance."""
    global _GRAPH
    if _GRAPH is None:
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        if not all([uri, username, password]):
            raise RuntimeError("Missing NEO4J_URI, NEO4J_USERNAME, or NEO4J_PASSWORD in environment.")
        _GRAPH = Neo4jGraph(url=uri, username=username, password=password)
    return _GRAPH


def retrieve_vector_context(vector_store: FAISS, question: str, k: int = 4) -> list[dict]:
    """Retrieve FAISS chunks and drop portal accessibility/navigation boilerplate."""
    noise_markers = ("accessibility menu", "text to speech", "bigger text", "invert colors")
    fetch_k = max(k * 3, 15)
    docs = vector_store.similarity_search(question, k=fetch_k)
    cleaned = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
        if not any(marker in doc.page_content.lower() for marker in noise_markers)
    ]
    return cleaned[:k]


def extract_keywords(question: str) -> list[str]:
    """Extract clean, non-stopword search terms from the user's question for Cypher filtering."""
    stop_words = {
        "what", "which", "who", "whom", "where", "when", "why", "how",
        "is", "are", "was", "were", "the", "a", "an", "for", "of", "in",
        "on", "to", "and", "or", "with", "under", "tamil", "nadu",
        "scheme", "schemes", "government", "apply", "documents", "required",
    }
    words = [w.strip("?.,!\"'()[]{}:;").lower() for w in question.split()]
    return [w for w in words if len(w) > 2 and w not in stop_words]


def retrieve_graph_context(graph: Neo4jGraph, question: str, limit: int = 15) -> list[dict]:
    """Query Neo4j across Scheme and target nodes, prioritizing exact query phrase matches."""
    keywords = extract_keywords(question)
    q_lower = question.strip().lower()
    if not keywords and not q_lower:
        return []

    cypher = """
    MATCH (s:Scheme)-[r:OFFERED_BY|PROVIDES_BENEFIT|TARGETS_BENEFICIARY|REQUIRES_DOCUMENT]->(target)
    WHERE toLower(coalesce(s.id, s.text, '')) CONTAINS $q_lower
       OR any(kw IN $keywords WHERE
            toLower(coalesce(s.id, s.text, '')) CONTAINS kw
            OR toLower(coalesce(target.id, target.text, '')) CONTAINS kw
       )
    WITH s, r, target,
         CASE WHEN toLower(coalesce(s.id, s.text, '')) CONTAINS $q_lower THEN 0 ELSE 1 END AS rank_score
    RETURN
        coalesce(s.id, 'Unknown Scheme') AS scheme,
        type(r) AS relationship,
        coalesce(target.id, target.text, labels(target)[0], '') AS target_value
    ORDER BY rank_score ASC, scheme ASC
    LIMIT $limit
    """
    return graph.query(
        cypher,
        params={"keywords": keywords, "q_lower": q_lower, "limit": limit},
    )


def retrieve_parallel(
    vector_store: FAISS,
    graph: Neo4jGraph,
    question: str,
    k: int = 4,
) -> tuple[list[dict], list[dict]]:
    """Run FAISS vector retrieval and Neo4j graph retrieval concurrently in separate threads."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        vec_future = executor.submit(retrieve_vector_context, vector_store, question, k)
        kg_future = executor.submit(retrieve_graph_context, graph, question)
        return vec_future.result(), kg_future.result()


def answer_query(question: str, k: int = 4, chat_history: list[dict] | None = None) -> dict:
    """Combine parallel FAISS and Neo4j evidence to generate a grounded answer via LLM."""
    vector_store = get_vector_store()
    graph = get_graph()

    vector_hits, graph_hits = retrieve_parallel(vector_store, graph, question, k=k)

    vector_block = "\n\n".join(
        f"[Chunk {i+1} | meta={item['metadata']}]\n{item['content']}"
        for i, item in enumerate(vector_hits)
    ) or "No matching vector passages found."

    graph_block = "\n".join(
        f"- {row['scheme']} --[{row['relationship']}]--> {row['target_value']}"
        for row in graph_hits
        if row.get("relationship") and row.get("target_value")
    ) or "No matching graph triples found."

    history_block = ""
    if chat_history:
        formatted_turns = "\n".join(
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
            for msg in chat_history[-6:]
            if msg.get("content")
        )
        if formatted_turns:
            history_block = f"\n--- RECENT CONVERSATION ---\n{formatted_turns}\n"

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""You are a Tamil Nadu government schemes assistant.
Answer the user's question using ONLY the retrieved vector passages and knowledge graph relationships below.
If a detail (eligibility rule, benefit amount, department, or required document) is not in the context, say so clearly rather than guessing.
{history_block}
Question: {question}

--- VECTOR CONTEXT (FAISS) ---
{vector_block}

--- KNOWLEDGE GRAPH CONTEXT (Neo4j) ---
{graph_block}
"""
    response = llm.invoke(prompt)
    return {
        "answer": response.content,
        "vector_hits": vector_hits,
        "graph_hits": graph_hits,
    }


def ask_hybrid_rag(question: str, chat_history: list[dict] | None = None, k: int = 4) -> dict:
    """Entry point called by the Streamlit frontend for text and voice queries."""
    return answer_query(question, k=k, chat_history=chat_history)


if __name__ == "__main__":
    test_q = "Nef Scheme Of Tiic"
    result = ask_hybrid_rag(test_q)
    print(result["answer"])