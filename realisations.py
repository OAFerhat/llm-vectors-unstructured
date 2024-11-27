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

def get_node_type_summary(session):
    """Get summary of all node types, their counts, and embedding status"""
    # First get all labels
    labels_query = "CALL db.labels() YIELD label RETURN label"
    labels = [record["label"] for record in session.run(labels_query)]
    
    nodes_info = []
    for label in labels:
        # For each label, get the summary information
        query = f"""
            MATCH (n:`{label}`)
            WITH COUNT(n) as totalCount,
                 COUNT(n.embedding) as withEmbedding,
                 CASE 
                     WHEN COUNT(n.embedding) > 0 
                     THEN apoc.agg.first(n.embedding) 
                     ELSE null 
                 END as sample
            WHERE totalCount > 0
            RETURN 
                totalCount,
                withEmbedding,
                CASE 
                    WHEN sample IS NOT NULL 
                    THEN size(sample) 
                    ELSE 0 
                END as embedding_dim
        """
        
        try:
            result = session.run(query)
            record = result.single()
            if record:
                nodes_info.append({
                    "label": label,
                    "total_count": record["totalCount"],
                    "with_embedding": record["withEmbedding"],
                    "embedding_dim": record["embedding_dim"]
                })
        except Exception as e:
            logger.error(f"Error getting summary for label {label}: {str(e)}")
            continue
    
    return nodes_info

def check_vector_indexes(session):
    """Get all vector indexes in the database"""
    query = """
        SHOW INDEXES
        YIELD name, type, labelsOrTypes, properties
        WHERE type = 'VECTOR'
        RETURN name, labelsOrTypes[0] as label
    """
    try:
        result = session.run(query)
        return {record["label"]: record["name"] for record in result}
    except Exception as e:
        logger.error(f"Error checking vector indexes: {str(e)}")
        return {}

def create_vector_index(session, label):
    """Create a vector index for a given label if it doesn't exist"""
    try:
        # Check if any nodes of this type have embeddings
        check_query = f"""
            MATCH (n:`{label}`)
            WHERE n.embedding IS NOT NULL
            RETURN COUNT(n) as count
        """
        result = session.run(check_query)
        count = result.single()["count"]
        
        if count == 0:
            logger.warning(f"No nodes of type {label} have embeddings. Skipping index creation.")
            return False
            
        # Create the index
        index_name = f"{label.lower()}_embedding_idx"
        create_query = f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:`{label}`)
            ON (n.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
            }}}}
        """
        session.run(create_query)
        logger.info(f"✅ Created vector index for {label}")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating vector index for {label}: {str(e)}")
        return False

def drop_vector_index(session, label):
    """Drop the vector index for a given label if it exists"""
    try:
        # Get the index name for this label
        query = """
            SHOW INDEXES
            YIELD name, type, labelsOrTypes
            WHERE type = 'VECTOR'
            AND labelsOrTypes[0] = $label
            RETURN name
        """
        result = session.run(query, label=label)
        record = result.single()
        
        if not record:
            logger.warning(f"No vector index found for {label}")
            return False
            
        # Drop the index
        index_name = record["name"]
        drop_query = f"DROP INDEX {index_name}"
        session.run(drop_query)
        logger.info(f"✅ Dropped vector index for {label}")
        return True
    except Exception as e:
        logger.error(f"❌ Error dropping vector index for {label}: {str(e)}")
        return False

def create_vector_indexes(uri, user, password):
    """Create vector indexes for node types with embeddings"""
    logger.info("Starting vector index creation process...")
    
    db = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        with db.session() as session:
            # Get summary of all node types
            nodes_info = get_node_type_summary(session)
            
            if not nodes_info:
                logger.info("No nodes found in the database.")
                return
                
            # Get existing vector indexes
            existing_indexes = check_vector_indexes(session)
            
            # Prepare table data
            table_data = []
            for idx, info in enumerate(nodes_info, 1):
                label = info["label"]
                has_index = label in existing_indexes
                table_data.append([
                    idx,
                    label,
                    info["total_count"],
                    f"{info['with_embedding']}/{info['total_count']}",
                    "✅" if has_index else "❌",
                    info["embedding_dim"] if info["with_embedding"] > 0 else "N/A"
                ])
            
            # Display table
            headers = ["#", "Node Type", "Total Nodes", "With Embeddings", "Indexed", "Embedding Dim"]
            print("\nNode Types Summary:")
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            
            # Get user input for which node types to index
            while True:
                choice = input("\nEnter the number of the node type to index (or 'q' to quit): ")
                if choice.lower() == 'q':
                    break
                    
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(nodes_info):
                        label = nodes_info[idx]["label"]
                        if label in existing_indexes:
                            logger.info(f"Vector index already exists for {label}")
                        else:
                            create_vector_index(session, label)
                    else:
                        logger.warning("Invalid choice. Please try again.")
                except ValueError:
                    logger.warning("Please enter a valid number or 'q' to quit.")
                    
    except Exception as e:
        logger.error(f"❌ Error during vector index creation: {str(e)}")
    finally:
        db.close()

def manage_vector_indexes(uri, user, password):
    """Manage vector indexes for node types with embeddings"""
    logger.info("Starting vector index management...")
    
    db = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        with db.session() as session:
            while True:
                # Get summary of all node types
                nodes_info = get_node_type_summary(session)
                
                if not nodes_info:
                    logger.info("No nodes found in the database.")
                    return
                    
                # Get existing vector indexes
                existing_indexes = check_vector_indexes(session)
                
                # Prepare table data
                table_data = []
                for idx, info in enumerate(nodes_info, 1):
                    label = info["label"]
                    has_index = label in existing_indexes
                    table_data.append([
                        idx,
                        label,
                        info["total_count"],
                        f"{info['with_embedding']}/{info['total_count']}",
                        "✅" if has_index else "❌",
                        info["embedding_dim"] if info["with_embedding"] > 0 else "N/A"
                    ])
                
                # Display table
                headers = ["#", "Node Type", "Total Nodes", "With Embeddings", "Indexed", "Embedding Dim"]
                print("\nNode Types Summary:")
                print(tabulate(table_data, headers=headers, tablefmt="grid"))
                
                # Show menu
                print("\nChoose an action:")
                print("[C] Create index")
                print("[D] Drop index")
                print("[Q] Quit")
                
                action = input("Your choice: ").upper()
                
                if action == 'Q':
                    break
                elif action in ['C', 'D']:
                    try:
                        idx = int(input("\nEnter the number of the node type to process: ")) - 1
                        if 0 <= idx < len(nodes_info):
                            label = nodes_info[idx]["label"]
                            has_index = label in existing_indexes
                            
                            if action == 'C':
                                if has_index:
                                    logger.info(f"Vector index already exists for {label}")
                                else:
                                    create_vector_index(session, label)
                            else:  # action == 'D'
                                if not has_index:
                                    logger.info(f"No vector index exists for {label}")
                                else:
                                    drop_vector_index(session, label)
                        else:
                            logger.warning("Invalid choice. Please try again.")
                    except ValueError:
                        logger.warning("Please enter a valid number.")
                else:
                    logger.warning("Invalid choice. Please enter 'C', 'D', or 'Q'.")
                    
    except Exception as e:
        logger.error(f"❌ Error during vector index management: {str(e)}")
    finally:
        db.close()

def get_nodes_with_embeddings(session):
    """Get all nodes that have embeddings and their counts."""
    query = """
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        WITH labels(n)[0] as label, n
        WITH label, 
             COUNT(n) as nodeCount,
             n.embedding as sample
        RETURN label, 
               nodeCount as count,
               size(sample) as embedding_dim
        ORDER BY label
    """
    
    try:
        result = session.run(query)
        records = list(result)
        if not records:
            logger.info("No nodes with embeddings found in the database.")
            return []
        
        # Format the results into a list of dictionaries
        nodes_info = [
            {
                "label": record["label"],
                "count": record["count"],
                "embedding_dim": record["embedding_dim"]
            }
            for record in records
        ]
        
        return nodes_info
    except Exception as e:
        logger.error(f"Error getting nodes with embeddings: {str(e)}")
        return []

def check_vector_index(tx, label):
    """Check if a vector index exists for a given label"""
    result = tx.run("""
        SHOW INDEXES
        YIELD name, type, labelsOrTypes, properties
        WHERE type = 'VECTOR'
          AND $label IN labelsOrTypes 
          AND 'embedding' IN properties
        RETURN count(*) > 0 as has_index
    """, label=label).single()
    return result and result["has_index"]

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

def get_course_data(chunk):
    """Extract course, module, and lesson data from chunk metadata"""
    data = {}
    path = chunk.metadata['source'].split(os.path.sep)
    
    # Extract hierarchy information from path
    data['course'] = path[-6]
    data['module'] = path[-4]
    data['lesson'] = path[-2]
    data['url'] = f"https://graphacademy.neo4j.com/courses/{data['course']}/{data['module']}/{data['lesson']}"
    data['text'] = chunk.page_content
    
    return data

def create_document_hierarchy(tx, data):
    """Create the document hierarchy in Neo4j"""
    return tx.run("""
        MERGE (c:Course {name: $course})
        MERGE (c)-[:HAS_MODULE]->(m:Module {name: $module})
        MERGE (m)-[:HAS_LESSON]->(l:Lesson {name: $lesson, url: $url})
        MERGE (l)-[:CONTAINS]->(p:Paragraph {text: $text})
        WITH p
        CALL db.create.setNodeVectorProperty(p, "embedding", $embedding)
        RETURN p
        """, 
        data
    )

def load_structured_documents():
    """Load documents and create a hierarchical structure in Neo4j"""
    logger.info("Starting structured document loading process...")
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("❌ Missing OpenAI API key!")
        return
    
    client = openai.OpenAI(api_key=api_key)
    
    # Initialize Neo4j connection
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not all([uri, user, password]):
        logger.error("❌ Missing Neo4j credentials!")
        return
        
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
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
                
                # Process chunks and create graph structure
                logger.info("Creating document hierarchy in Neo4j...")
                with driver.session(database="neo4j") as session:
                    for chunk in chunks:
                        # Get document structure data
                        data = get_course_data(chunk)
                        
                        # Generate embedding
                        try:
                            response = client.embeddings.create(
                                model="text-embedding-ada-002",
                                input=data['text']
                            )
                            data['embedding'] = response.data[0].embedding
                        except Exception as e:
                            logger.error(f"❌ Error generating embedding: {str(e)}")
                            continue
                        
                        # Create nodes and relationships
                        session.execute_write(create_document_hierarchy, data)
                        
                logger.info("✅ Successfully created document hierarchy with embeddings")
                
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
                
    except Exception as e:
        logger.error(f"❌ An error occurred: {str(e)}")
        raise
    finally:
        driver.close()

def main():
    parser = argparse.ArgumentParser(description='Neo4j Data Operations')
    parser.add_argument('--load-quora', action='store_true', 
                      help='Load Quora Q&A data with embeddings')
    parser.add_argument('--manage-indexes', action='store_true',
                      help='Manage vector indexes (create/drop) for nodes with embeddings')
    parser.add_argument('--search', action='store_true',
                      help='Perform semantic search on questions')
    parser.add_argument('--load-docs', action='store_true',
                      help='Load and chunk documents from a directory')
    parser.add_argument('--generic-search', action='store_true',
                      help='Search through document chunks using vector similarity')
    parser.add_argument('--load-structured', action='store_true',
                      help='Load documents and create hierarchical structure')
    
    args = parser.parse_args()
    
    if args.load_quora:
        load_quora_data()
    elif args.manage_indexes:
        manage_vector_indexes(os.getenv("NEO4J_URI"), os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    elif args.search:
        semantic_search()
    elif args.load_docs:
        load_and_chunk_documents()
    elif args.generic_search:
        generic_search()
    elif args.load_structured:
        load_structured_documents()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
