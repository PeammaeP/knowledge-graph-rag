from src.utils.chunks import chunk_text
from src.utils.getfile import get_text_from_file
from src.utils.embedding import embedding
from src.database.vectordb import get_graph_database

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

if __name__ == "__main__":
    config = Config()

    text = get_text_from_file(config.remote_pdf_url, config.__init_subclass__)

    chunks = chunk_text(text, config.chunk_size, config.overlap)

    embeddings = embedding(chunks)
    #print(len(chunks))

    driver = get_graph_database()
    driver.execute_query(config.cypher_query, chunks=chunks, embeddings=embeddings)
