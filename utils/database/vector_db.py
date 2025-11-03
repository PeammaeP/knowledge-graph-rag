import os

from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

def init_graph_database():
    driver = GraphDatabase.driver(os.getenv("NEO4J_WEB"), auth=("neo4j", os.getenv("NEO4J_PW")))

    driver.execute_query("""CREATE VECTOR INDEX pdf IF NOT EXISTS FOR (c:Chunk) ON c.embedding""")

    return driver
