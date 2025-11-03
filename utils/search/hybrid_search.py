def get_hybrid_search(driver, question_embedding, question):
    hybrid_query = '''
    CALL {
        // vector index
        CALL db.index.vector.queryNodes('pdf', $k, $question_embedding) YIELD node, score
        WITH collect({node:node, score:score}) AS nodes, max(score) AS max
        UNWIND nodes AS n
        // We use 0 as min
        RETURN n.node AS node, (n.score / max) AS score
        UNION
        // keyword index
        CALL db.index.fulltext.queryNodes('ftPdfChunk', $question, {limit: $k})
        YIELD node, score
        WITH collect({node:node, score:score}) AS nodes, max(score) AS max
        UNWIND nodes AS n
        // We use 0 as min
        RETURN n.node AS node, (n.score / max) AS score
    }
    // dedup
    WITH node, max(score) AS score ORDER BY score DESC LIMIT $k
    RETURN node, score
    '''
    similar_hybrid_records, _, _ = driver.execute_query(hybrid_query, question_embedding=question_embedding, question=question, k=4)

    return similar_hybrid_records
