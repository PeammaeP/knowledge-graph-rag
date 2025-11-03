import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterable

from utils.function.chunks import chunk_text
from utils.function.getfile import get_text_from_file
from utils.function.embedding import get_embedding, get_model_stream
from utils.database.vector_db import init_graph_database
from utils.search.hybrid_search import get_hybrid_search

@dataclass
class Config:
    # File Configuration
    remote_pdf_url: str = "https://arxiv.org/pdf/1709.00666.pdf"
    pdf_filename: str = "sample-document.pdf"

    # Chunking Hyperparameter
    chunk_size: int = 500
    overlap: int = 40

    # Cypher Query Template (parameterized)
    cypher_query: str = '''
        WITH $chunks as chunks, range(0, size($chunks)) AS index
        UNWIND index AS i
        WITH i, chunks[i] AS chunk, $embeddings[i] AS embedding
        MERGE (c:Chunk {index: i})
        SET c.text = chunk, c.embedding = embedding
    '''

    # Vector search query template
    vector_query: str = '''
        CALL db.index.vector.queryNodes('pdf', 2, $question_embedding) YIELD node AS hits, score
        RETURN hits.text AS text, score, hits.index AS index
    '''

    # System prompt for answer synthesis
    system_prompt: str = (
        "You're an Einstein expert, but can only use the provided documents"
        "to respond to the questions."
    )

class RAGPipeline:
    """
    End-to-end pipeline that:
    1) Loads and chunks a source document.
    2) Embeds and stores chunks in Neo4j with a vector index.
    3) Runs vector search to fetch top-k context.
    4) (Optionally) Runs hybrid search, seeding it with vector hits.
    5) Streams a final answer from an LLM using the retrieved context.
    """
    def __init__(self, config: Config) -> None:
        self.config = config
        self.driver = None
        self.text: Optional[str] = None
        self.chunks: Optional[List[str]] = None
        self.similar_records: Optional[List[Dict[str, Any]]] = None  # from vector search

    # ---------- Setup ----------
    def init_driver(self) -> None:
        if self.driver is None:
            self.driver = init_graph_database()

    # ---------- ETL ----------
    def load_text(self) -> str:
        self.text = get_text_from_file(self.config.remote_pdf_url, self.config.pdf_filename)
        return self.text

    def build_chunks(self) -> List[str]:
        if self.text is None:
            self.load_text()
        self.chunks = chunk_text(self.text, self.config.chunk_size, self.config.overlap)
        return self.chunks

    def upsert_chunks(self) -> None:
        self.init_driver()
        if self.chunks is None:
            self.build_chunks()
        embeddings = get_embedding(self.chunks)
        self.driver.execute_query(
            self.config.cypher_query,
            chunks=self.chunks,
            embeddings=embeddings
        )

    # ---------- Retrieval ----------
    def vector_search(self, question: str, k: int = 2) -> List[Dict[str, Any]]:
        """
        Executes a vector search and caches the results in self.similar_records.
        """
        self.init_driver()
        question_embedding = get_embedding(question)[0]

        vector_query = self.config.vector_query.replace(" 2,", f" {k},")
        records, _, _ = self.driver.execute_query(
            vector_query,
            question_embedding=question_embedding
        )
        self.similar_records = records
        return records

    def hybrid_search(self, question: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Calls YOUR get_hybrid_search(driver, question_embedding, question).
        It already uses both vector and full-text (k controlled inside).
        """
        self.init_driver()
        q_emb = get_embedding(question)[0]
        records = get_hybrid_search(self.driver, q_emb, question)
        self.hybrid_hits = records
        return records

    # ---------- Synthesis ----------
    def synthesize_answer_from_docs(self, question: str, docs: List[str]) -> str:
        """
        Streams tokens from the chat model using docs as context, and returns the full answer string.
        """
        user_message = f"""
            Use the following documents to answer the question that will follow:
            {docs}
            ---
            The question to answer using information only from the above documents: {question}
        """

        stream: Iterable[str] = get_model_stream(self.config.system_prompt, user_message)

        final_answer_parts: List[str] = []
        for token in stream:
            print(token, end="", flush=True)  # live stream to console
            final_answer_parts.append(token)
        print()  # newline after streaming
        return "".join(final_answer_parts)

    # ---------- Orchestrations ----------
    def run_vector_only(self, question: str, k: int = 2) -> str:
        """
        Full vector-only flow: upsert -> vector search -> synthesize.
        """
        self.upsert_chunks()
        records = self.vector_search(question, k=k)
        docs = [r["text"] for r in records]
        return self.synthesize_answer_from_docs(question, docs)

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None

# ---------------- CLI entrypoint ----------------
class UserPerform:
    question: str = "At what time was Einstein really interested in experimental works?"

def main():
    parser = argparse.ArgumentParser(description="RAG pipeline (vector and hybrid).")
    parser.add_argument("--k", type=int, default=2, help="Top-k for vector search")
    args = parser.parse_args()

    config = Config()
    pipe = RAGPipeline(config)

    try:
        final_answer = pipe.run_vector_only(UserPerform.question, k=args.k)
    finally:
        pipe.close()

if __name__ == "__main__":
    main()
