from neo4j.exceptions import ClientError

def _ensure_indexes(driver, embedding_dim=384,
                    vector_index_name="pdf",
                    fulltext_index_name="PdfChunkFulltext",
                    similarity="cosine"):
    driver.execute_query(f"""
        CREATE VECTOR INDEX {vector_index_name} IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {embedding_dim},
            `vector.similarity_function`: '{similarity}'
        }}
        }}
    """)
    # Full-text index on Chunk.text
    driver.execute_query(f"""
        CREATE FULLTEXT INDEX {fulltext_index_name} IF NOT EXISTS
        FOR (c:Chunk) ON EACH [c.text]
    """)

def get_hybrid_search(driver, question_embedding, question, k=4,
                      embedding_dim=384,
                      vector_index_name="pdf",
                      fulltext_index_name="ftPdfChunk",
                      similarity="cosine",
                      database=None):
    """
    Hybrid search = vector (normalized) UNION full-text, then dedup + rank.
    Ensures indexes exist; retries once if the FT index is missing.
    """
    _ensure_indexes(driver, embedding_dim, vector_index_name, fulltext_index_name, similarity)

    hybrid_query = f"""
    CALL {{
        // vector index
        CALL db.index.vector.queryNodes('{vector_index_name}', $k, $question_embedding) YIELD node, score
        WITH collect({{node:node, score:score}}) AS nodes, max(score) AS max
        UNWIND nodes AS n
        RETURN n.node AS node, (n.score / max) AS score
        UNION
        // keyword index
        CALL db.index.fulltext.queryNodes('{fulltext_index_name}', $question, {{limit: $k}})
        YIELD node, score
        WITH collect({{node:node, score:score}}) AS nodes, max(score) AS max
        UNWIND nodes AS n
        RETURN n.node AS node, (n.score / max) AS score
    }}
    WITH node, max(score) AS score
    ORDER BY score DESC
    LIMIT $k
    RETURN node, score
    """
    try:
        records, _, _ = driver.execute_query(
            hybrid_query,
            question_embedding=question_embedding,
            question=question,
            k=k,
            database=database
        )
        return records
    except ClientError as e:
        # If FT index was missing due to race or different DB, ensure & retry once.
        _ensure_indexes(driver, embedding_dim, vector_index_name, fulltext_index_name, similarity)
        records, _, _ = driver.execute_query(
            hybrid_query,
            question_embedding=question_embedding,
            question=question,
            k=k,
            database=database
        )
        return records
