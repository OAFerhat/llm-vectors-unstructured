from dotenv import load_dotenv
import os
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jConnection:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get Neo4j credentials from environment variables
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.driver = None

    def connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            self.driver.verify_connectivity()
            logger.info("✅ Successfully connected to Neo4j database!")
            return True
        except ServiceUnavailable:
            logger.error("❌ Failed to connect to Neo4j database!")
            logger.error("Please check if:")
            logger.error("- The database is running")
            logger.error("- The credentials in .env are correct")
            logger.error(f"- The URI ({self.uri}) is correct")
            return False
        except Exception as e:
            logger.error(f"❌ An error occurred: {str(e)}")
            return False

    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Database connection closed.")

    def execute_query(self, query, parameters=None):
        """Execute a Cypher query and return the results"""
        if not self.driver:
            raise Exception("Database connection not established. Call connect() first.")
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return list(result)
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise

def load_quora_data():
    """Function to load Quora Q&A data with embeddings into the database"""
    logger.info("Starting Quora data loading process...")
    
    query = """
    LOAD CSV WITH HEADERS
    FROM 'https://data.neo4j.com/llm-vectors-unstructured/Quora-QuAD-1000-embeddings.csv' AS row

    MERGE (q:Question{text:row.question})
    WITH row,q
    CALL db.create.setNodeVectorProperty(q, 'embedding', apoc.convert.fromJsonList(row.question_embedding))

    MERGE (a:Answer{text:row.answer})
    WITH row,a,q
    CALL db.create.setNodeVectorProperty(a, 'embedding', apoc.convert.fromJsonList(row.answer_embedding))

    MERGE(q)-[:ANSWERED_BY]->(a)
    """
    
    # Initialize connection
    db = Neo4jConnection()
    
    try:
        # Connect to database
        if not db.connect():
            return
        
        # Execute data loading
        results = db.execute_query(query)
        logger.info("✅ Quora data loading completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"❌ An error occurred during data loading: {str(e)}")
        raise
    finally:
        db.close()

def main():
    parser = argparse.ArgumentParser(description='Neo4j Data Operations')
    parser.add_argument('--load-quora', action='store_true', 
                      help='Load Quora Q&A data with embeddings')
    
    args = parser.parse_args()
    
    if args.load_quora:
        load_quora_data()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
