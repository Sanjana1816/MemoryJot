# responsible for storing and retrieving entries
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

# loading evnironmental variables  from .env file
load_dotenv()

# setting up the embedding function
openai_ef =embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

# initialise database client-This client will manage our collections. We are using a persistent client, which means the data will be saved to disk in the 'chroma_db' folder.
client= chromadb.PersistentClient(path="./chroma_db")


#  get/create the collection -A collection is like a table in a SQL database. We pass our embedding function to the collection so it knows how to handle the text we add.
collection=client.get_or_create_collection(
    name="journal_entries",
    embedding_function=openai_ef
)


# defining functions to interact with the database
def add_journal_entry(entry_text: str, metadata: dict):
    """
    Adds a single journal entry to the ChromaDB collection.

    Args:
        entry_text (str): The text content of the journal entry.
        metadata (dict): A dictionary containing metadata, e.g., {'timestamp': '2024-05-21'}
    """
    entry_id=str(hash(entry_text))
    collection.add(
        documents=[entry_text],
        metadatas=[metadata],
        ids=[entry_id]
    )
    print(f"Succesfully addedentry with id : {entry_id}")

def query_journal_entries(query_text: str, n_results:int=3):
    """
    Queries the collection to find the most similar journal entries.

    Args:
        query_text (str): The user's query (e.g., "How was I feeling about work last month?").
        n_results (int): The number of similar entries to return.

    Returns:
        list: A list of the most relevant journal entries.
    """
    results=collection.query(
        query_texts=[query_text],
        n_results=n_results        
    )
    return results['documents'][0]





