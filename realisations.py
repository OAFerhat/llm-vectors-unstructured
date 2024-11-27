from dotenv import load_dotenv
import os
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import logging
import argparse
from tabulate import tabulate
import openai
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings

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

def create_vector_indexes():
    """Create vector indexes for Question and Answer nodes"""
    logger.info("Starting vector indexes creation...")
    
    # Query for Question index
    question_query = """
    CREATE VECTOR INDEX questions IF NOT EXISTS
    FOR (q:Question)
    ON q.embedding
    OPTIONS {indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
    }}
    """
    
    # Query for Answer index
    answer_query = """
    CREATE VECTOR INDEX answers IF NOT EXISTS
    FOR (a:Answer)
    ON a.embedding
    OPTIONS {indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
    }}
    """
    
    # Initialize connection
    db = Neo4jConnection()
    
    try:
        # Connect to database
        if not db.connect():
            return
        
        # Create Question index
        logger.info("Creating Question index...")
        db.execute_query(question_query)
        logger.info("✅ Question index created successfully!")
        
        # Create Answer index
        logger.info("Creating Answer index...")
        db.execute_query(answer_query)
        logger.info("✅ Answer index created successfully!")
        
    except Exception as e:
        logger.error(f"❌ An error occurred while creating indexes: {str(e)}")
        raise
    finally:
        db.close()

def get_embedding(text, api_key):
    """Get embedding from OpenAI API"""
    client = openai.OpenAI(api_key=api_key)
    
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"❌ Error getting embedding from OpenAI: {str(e)}")
        raise

def wrap_text(text, max_width):
    """Wrap text to two lines with max width"""
    if len(text) <= max_width:
        return text
    
    # Find a good breaking point near the middle
    mid_point = len(text) // 2
    space_before = text.rfind(' ', 0, mid_point)
    space_after = text.find(' ', mid_point)
    
    break_point = space_before if space_before != -1 else space_after
    if break_point == -1:
        # If no good breaking point, just truncate
        return text[:max_width-3] + "..."
    
    first_line = text[:break_point]
    second_line = text[break_point+1:]
    
    # Truncate second line if too long
    if len(second_line) > max_width:
        second_line = second_line[:max_width-3] + "..."
    
    return f"{first_line}\n{second_line}"

def format_table_data(results, max_question=60, max_answer=100):
    """Format results into table data with specified maximum widths"""
    table_data = []
    headers = ["Similar Question", "Answer", "Score"]
    
    for record in results:
        table_data.append([
            wrap_text(record["question"], max_question),
            wrap_text(record["answer"], max_answer),
            f"{record['score']:.4f}"
        ])
    
    return headers, table_data

def semantic_search():
    """Perform semantic search on questions and get corresponding answers"""
    logger.info("Starting semantic search...")
    
    # Initialize connection
    db = Neo4jConnection()
    
    try:
        # Connect to database
        if not db.connect():
            return
            
        # Get OpenAI API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("❌ OpenAI API key not found in environment variables!")
            return
        
        while True:
            # Get user input
            question = input("\nEnter your question (or 'Q' to quit): ")
            if question.upper() == 'Q':
                logger.info("Exiting search...")
                break
                
            logger.info(f"Searching for similar questions to: {question}")
            
            # Query to find similar questions and their answers
            query = """
            CALL db.index.vector.queryNodes('questions', 5, $embedding)
            YIELD node, score
            MATCH (node)-[:ANSWERED_BY]->(answer)
            RETURN node.text as question, answer.text as answer, score
            ORDER BY score DESC
            """
            
            try:
                # Get embedding for the question
                embedding = get_embedding(question, api_key)
                
                # Execute search
                results = db.execute_query(
                    query, 
                    parameters={"embedding": embedding}
                )
                
                # Format results as table
                if results:
                    headers, table_data = format_table_data(results)
                    
                    # Print results in table format
                    print("\nSearch Results:")
                    print(tabulate(
                        table_data,
                        headers=headers,
                        tablefmt="grid",
                        maxcolwidths=[60, 100, 10]
                    ))
                    logger.info("✅ Search completed successfully!")
                else:
                    logger.info("No similar questions found.")
                
                # Ask if user wants to continue
                while True:
                    choice = input("\nWould you like to:\n[S] Search again\n[Q] Quit\nYour choice: ")
                    if choice.upper() in ['S', 'Q']:
                        if choice.upper() == 'Q':
                            logger.info("Exiting search...")
                            return
                        break
                    print("Invalid choice. Please enter 'S' to search again or 'Q' to quit.")
                    
            except Exception as e:
                logger.error(f"❌ An error occurred during this search: {str(e)}")
                continue
            
    except Exception as e:
        logger.error(f"❌ An error occurred during semantic search: {str(e)}")
        raise
    finally:
        db.close()

def load_and_chunk_documents():
    """Load documents from a directory, chunk them, and store in Neo4j with embeddings"""
    logger.info("Starting document loading and chunking process...")
    
    while True:
        # Get directory path from user
        dir_path = input("\nEnter the absolute path to your documents directory (or 'Q' to quit): ")
        if dir_path.upper() == 'Q':
            logger.info("Exiting document loading...")
            break
            
        if not os.path.exists(dir_path):
            logger.error(f"❌ Directory not found: {dir_path}")
            continue
            
        try:
            # Initialize document loader
            logger.info(f"Loading documents from: {dir_path}")
            loader = DirectoryLoader(dir_path, glob="**/lesson.adoc", loader_cls=TextLoader)
            docs = loader.load()
            logger.info(f"✅ Loaded {len(docs)} documents")
            
            # Initialize text splitter
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=1500,
                chunk_overlap=200,
            )
            
            # Split documents into chunks
            logger.info("Splitting documents into chunks...")
            chunks = text_splitter.split_documents(docs)
            logger.info(f"✅ Created {len(chunks)} chunks")
            
            # Get environment variables
            api_key = os.getenv("OPENAI_API_KEY")
            neo4j_uri = os.getenv("NEO4J_URI")
            neo4j_user = os.getenv("NEO4J_USERNAME")
            neo4j_pass = os.getenv("NEO4J_PASSWORD")
            
            if not all([api_key, neo4j_uri, neo4j_user, neo4j_pass]):
                logger.error("❌ Missing required environment variables!")
                continue
            
            # Store chunks in Neo4j with embeddings
            logger.info("Storing chunks in Neo4j and generating embeddings...")
            neo4j_db = Neo4jVector.from_documents(
                chunks,
                OpenAIEmbeddings(openai_api_key=api_key),
                url=neo4j_uri,
                username=neo4j_user,
                password=neo4j_pass,
                database="neo4j",
                index_name="chunkVector",
                node_label="Chunk",
                text_node_property="text",
                embedding_node_property="embedding",
                embedding_function_type="setNodeVectorProperty"  # Updated to use new function
            )
            logger.info("✅ Successfully stored chunks with embeddings in Neo4j")
            
            # Ask if user wants to continue
            while True:
                choice = input("\nWould you like to:\n[L] Load another directory\n[Q] Quit\nYour choice: ")
                if choice.upper() in ['L', 'Q']:
                    if choice.upper() == 'Q':
                        logger.info("Exiting document loading...")
                        return
                    break
                print("Invalid choice. Please enter 'L' to load another directory or 'Q' to quit.")
                
        except Exception as e:
            logger.error(f"❌ An error occurred: {str(e)}")
            continue

def generic_search():
    """Search for similar chunks using vector similarity"""
    logger.info("Starting generic search...")
    
    # Initialize Neo4j connection
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not all([uri, user, password, api_key]):
        logger.error("❌ Missing required environment variables!")
        return
        
    db = GraphDatabase.driver(uri, auth=(user, password))
    openai.api_key = api_key
    
    try:
        while True:
            # Get search query from user
            query = input("\nEnter your search query (or 'Q' to quit): ")
            if query.upper() == 'Q':
                logger.info("Exiting search...")
                break
                
            # Generate embedding for the query
            try:
                response = openai.embeddings.create(
                    model="text-embedding-ada-002",
                    input=query
                )
                embedding = response.data[0].embedding
            except Exception as e:
                logger.error(f"❌ Error generating embedding: {str(e)}")
                continue
            
            # Search for similar chunks
            with db.session() as session:
                result = session.run("""
                    CALL db.index.vector.queryNodes('chunkVector', 5, $embedding)
                    YIELD node, score
                    RETURN node.text as text, score
                    ORDER BY score DESC
                """, embedding=embedding)
                
                results = [{"text": record["text"], "score": record["score"]} 
                          for record in result]
                
                if results:
                    # Format results into table
                    table_data = []
                    headers = ["Text", "Similarity Score"]
                    
                    for r in results:
                        table_data.append([
                            wrap_text(r["text"], 160),
                            f"{r['score']:.4f}"
                        ])
                    
                    print("\nSearch Results:")
                    print(tabulate(
                        table_data,
                        headers=headers,
                        tablefmt="grid",
                        maxcolwidths=[160, 10]
                    ))
                    logger.info("✅ Search completed successfully!")
                else:
                    logger.info("No similar chunks found.")
                
            # Ask if user wants to continue
            while True:
                choice = input("\nWould you like to:\n[S] Search again\n[Q] Quit\nYour choice: ")
                if choice.upper() in ['S', 'Q']:
                    if choice.upper() == 'Q':
                        logger.info("Exiting search...")
                        return
                    break
                print("Invalid choice. Please enter 'S' to search again or 'Q' to quit.")
                
    except Exception as e:
        logger.error(f"❌ An error occurred: {str(e)}")
        raise
    finally:
        db.close()

def main():
    parser = argparse.ArgumentParser(description='Neo4j Data Operations')
    parser.add_argument('--load-quora', action='store_true', 
                      help='Load Quora Q&A data with embeddings')
    parser.add_argument('--create-indexes', action='store_true',
                      help='Create vector indexes for Question and Answer nodes')
    parser.add_argument('--search', action='store_true',
                      help='Perform semantic search on questions')
    parser.add_argument('--load-docs', action='store_true',
                      help='Load and chunk documents from a directory')
    parser.add_argument('--generic-search', action='store_true',
                      help='Search through document chunks using vector similarity')
    
    args = parser.parse_args()
    
    if args.load_quora:
        load_quora_data()
    elif args.create_indexes:
        create_vector_indexes()
    elif args.search:
        semantic_search()
    elif args.load_docs:
        load_and_chunk_documents()
    elif args.generic_search:
        generic_search()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
