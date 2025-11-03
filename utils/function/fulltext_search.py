def get_fulltext_search(driver):
    return driver.execute_query("CREATE FULLTEXT INDEX PdfChunkFulltext FOR (c:Chunk) ON EACH [c.text]")
