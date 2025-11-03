from src.utils.chunks import chunk_text
from src.utils.getfile import get_text_from_file
from src.utils.embedding import embedding
from src.database.vectordb import init_graph_database

class Config:
    # File Configuration
    remote_pdf_url = "https://arxiv.org/pdf/1709.00666.pdf"
    pdf_filename = "ch02-downloaded.pdf"

    # Chunking Hyperparameter
    chunk_size = 500
    overlap = 40

    # Cypher Query Template
    cypher_query = '''
    WITH $chunks as chunks, range(0, size($chunks)) AS index
    UNWIND index AS i
    WITH i, chunks[i] AS chunk, $embeddings[i] AS embedding
    MERGE (c:Chunk {index: i})
    SET c.text = chunk, c.embedding = embedding
    '''

    query = '''
    CALL db.index.vector.queryNodes('pdf', 2, $question_embedding) YIELD node AS hits, score
    RETURN hits.text AS text, score, hits.index AS index
    '''

class UserPerform:
    question = "At what time was Einstein really interested in experimental works?"

if __name__ == "__main__":
    config = Config()

    text = get_text_from_file(config.remote_pdf_url, config.__init_subclass__)

    chunks = chunk_text(text, config.chunk_size, config.overlap)

    embeddings = embedding(chunks)
    # print(len(chunks))

    driver = init_graph_database()
    driver.execute_query(config.cypher_query, chunks=chunks, embeddings=embeddings)

    # Getting data from a chunk node in Neo4j
    records, _, _ = driver.execute_query("MATCH (c:Chunk) WHERE c.index = 0 RETURN c.embedding, c.text")
    # print(records[0]["c.text"][0:30])
    # print(records[0]["c.embedding"][0:3])

    # Embedding User Question
    question_embedding = embedding(UserPerform.question)[0]

    similar_records, _, _ = driver.execute_query(Config.query,
    question_embedding=question_embedding)

    for record in similar_records:
        print(record["text"])
        print(record["score"], record["index"])
        print("======")
