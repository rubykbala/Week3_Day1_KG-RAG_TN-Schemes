import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer


def get_neo4j_graph() -> Neo4jGraph:
    """Connect to Neo4j AuraDB using neo4j+ssc:// for local SSL compatibility."""
    load_dotenv(override=True)
    uri = os.getenv("NEO4J_URI", "").strip()
    username = os.getenv("NEO4J_USERNAME", "").strip()
    password = os.getenv("NEO4J_PASSWORD", "").strip()
    database = os.getenv("NEO4J_DATABASE", "02b400a7").strip() or "02b400a7"

    if not uri:
        raise ValueError("NEO4J_URI is empty. Check your .env file in the project root.")

    connect_uri = uri.replace("neo4j+s://", "neo4j+ssc://", 1)
    graph = Neo4jGraph(
        url=connect_uri,
        username=username,
        password=password,
        database=database,
    )
    graph.refresh_schema()
    return graph


def select_batch(chunk_docs: list, start_idx: int = 17, end_idx: int | None = 47) -> list:
    """Return a controlled slice of chunk_docs without re-processing earlier indices."""
    total = len(chunk_docs)
    if start_idx >= total:
        return []
    return chunk_docs[start_idx:end_idx] if end_idx is not None else chunk_docs[start_idx:]


def load_chunk_documents() -> list[Document]:
    """Load chunk documents from data/chunks_v2.json (or data/chunks.json)."""
    project_root = Path(__file__).resolve().parent.parent
    for rel_path in ("data/chunks_v2.json", "data/chunks.json"):
        chunk_file = project_root / rel_path
        if chunk_file.exists():
            raw = json.loads(chunk_file.read_text(encoding="utf-8"))
            items = raw if isinstance(raw, list) else raw.get("chunks", [])
            docs: list[Document] = []
            for item in items:
                if isinstance(item, Document):
                    docs.append(item)
                elif isinstance(item, dict):
                    page_content = (
                        item.get("page_content")
                        or item.get("text")
                        or item.get("content")
                        or ""
                    )
                    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {
                        k: v for k, v in item.items() if k not in {"page_content", "text", "content", "metadata"}
                    }
                    if page_content.strip():
                        docs.append(Document(page_content=page_content, metadata=metadata))
                elif isinstance(item, str) and item.strip():
                    docs.append(Document(page_content=item))
            print(f"Loaded {len(docs)} chunk documents from {rel_path}.")
            return docs

    raise FileNotFoundError("Could not find data/chunks_v2.json or data/chunks.json.")


def run_kg_pipeline(
    chunk_docs: list,
    llm,
    allowed_nodes: list[str] | None = None,
    allowed_relationships: list[tuple[str, str, str]] | list[str] | None = None,
    start_idx: int = 17,
    end_idx: int | None = 47,
) -> None:
    """Convert a slice of chunks into graph documents and write them to Neo4j AuraDB."""
    batch_docs = select_batch(chunk_docs, start_idx=start_idx, end_idx=end_idx)
    if not batch_docs:
        print(f"No chunks to process for slice [{start_idx}:{end_idx}] (total chunks: {len(chunk_docs)}).")
        return

    graph = get_neo4j_graph()

    transformer_kwargs = {"llm": llm}
    if allowed_nodes is not None:
        transformer_kwargs["allowed_nodes"] = allowed_nodes
    if allowed_relationships is not None:
        transformer_kwargs["allowed_relationships"] = allowed_relationships

    llm_transformer = LLMGraphTransformer(**transformer_kwargs)
    graph_documents = llm_transformer.convert_to_graph_documents(batch_docs)

    if graph_documents:
        graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True,
        )
        graph.refresh_schema()

    print(f"Processed slice [{start_idx}:{end_idx}] ({len(batch_docs)} input chunks).")
    print(f"Added {len(graph_documents)} graph documents to Neo4j AuraDB.")
    print("Graph schema updated in Neo4j AuraDB.")


if __name__ == "__main__":
    load_dotenv(override=True)
    chunk_docs = load_chunk_documents()

    # Adjust model name if your project uses a different OpenAI chat model
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    allowed_nodes = [
        "Scheme",
        "Benefit",
        "Department",
        "Beneficiarygroup",
        "Documentrequired",
    ]
    allowed_relationships = [
        ("Scheme", "OFFERED_BY", "Department"),
        ("Scheme", "PROVIDES_BENEFIT", "Benefit"),
        ("Scheme", "TARGETS_BENEFICIARY", "Beneficiarygroup"),
        ("Scheme", "REQUIRES_DOCUMENT", "Documentrequired"),
    ]

    # Batch 1 of remaining chunks: indices 17 to 47
    #run_kg_pipeline(
     #   chunk_docs=chunk_docs,
      #  llm=llm,
       # allowed_nodes=allowed_nodes,
        #allowed_relationships=allowed_relationships,
        #start_idx=17,
        #end_idx=47,
    #)

    # Once batch 1 finishes cleanly, comment out above and run batch 2:
    run_kg_pipeline(
    chunk_docs=chunk_docs,
    llm=llm,
    allowed_nodes=allowed_nodes,
    allowed_relationships=allowed_relationships,
    start_idx=47,
    end_idx=None,
    )