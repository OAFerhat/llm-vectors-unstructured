from dotenv import load_dotenv
import os
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

def test_connection():
    # Load environment variables
    load_dotenv()
    
    # Get Neo4j credentials from environment variables
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    try:
        # Initialize the driver
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Verify connectivity
        driver.verify_connectivity()
        print("✅ Successfully connected to Neo4j database!")
        
        # Get database version
        with driver.session() as session:
            result = session.run("CALL dbms.components() YIELD name, versions, edition")
            record = result.single()
            print(f"Database: {record['name']}")
            print(f"Version: {record['versions'][0]}")
            print(f"Edition: {record['edition']}")
            
            # Create sample movie nodes and relationship
            print("\nCreating sample movie nodes and relationship...")
            create_query = """
            MERGE (m1:Movie {title: 'The Matrix', released: 1999})
            MERGE (m2:Movie {title: 'The Matrix Reloaded', released: 2003})
            MERGE (m1)-[r:SEQUEL_TO]->(m2)
            RETURN m1.title, m2.title
            """
            result = session.run(create_query)
            record = result.single()
            print(f"✅ Created relationship: {record[0]} is a sequel to {record[1]}")
            
    except ServiceUnavailable:
        print("❌ Failed to connect to Neo4j database!")
        print("Please check if:")
        print("- The database is running")
        print("- The credentials in .env are correct")
        print(f"- The URI ({uri}) is correct")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
    finally:
        if 'driver' in locals():
            driver.close()

if __name__ == "__main__":
    test_connection()
