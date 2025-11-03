from neo4j import GraphDatabase

def init_graph_database():
    driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "password"))

    driver.execute_query("""CREATE VECTOR INDEX pdf IF NOT EXISTS
    FOR (c:Chunk)
    ON c.embedding""")

    return driver
