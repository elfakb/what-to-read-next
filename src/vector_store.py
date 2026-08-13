import pandas as pd 
import chromadb 
from src.embeddings import get_embeddings_batch
BATCH_SIZE = 100

# loads all books  create embeddings and sotores in vector db 
def vector_store():
    books = pd.read_csv("data/processed/books_processed.csv")

    # filling empty features 
    books["image_url"] = books["image_url"].fillna("")
    books["authors"] = books["authors"].fillna("")
    books["genres_str"] = books["genres_str"].fillna("")

    client = chromadb.PersistentClient(path ="db/chroma") # permenant database createtred

    try:
        client.delete_collection("books")
    except Exception:
        pass

    collection = client.get_or_create_collection("books")


    # Process the Books in Batches 
    for i in range(0, len(books),BATCH_SIZE):
        batch = books.iloc[i:i + BATCH_SIZE] # selects current batch 1-100 etc
        text = batch["embed_text"].tolist() # text for embeddings
        embeddings = get_embeddings_batch(text) # create embedding for text
        ids = batch["book_id"].astype(str).tolist()
        metadatas = batch[["title", "authors", "genres_str", "image_url"]].to_dict("records") # create metadata 

        # Add Everything to ChromaDB
        collection.add(
            ids = ids,
            embeddings = embeddings,
            documents = text,
            metadatas= metadatas
        )
        print(f"{i + len(batch)}/{len(books)} books uploaded")
    print("Vectore Store Ready !!!")


# searching books based on the query : similaritiy search : SEMANTİC SEARCH .query top n_resulur wihf is top 5 results are shown
def search_books(query , n_results= 5):
    client = chromadb.PersistentClient(path ="db/chroma")
    collection = client.get_or_create_collection("books")

    query_embedding = get_embeddings_batch([query])[0]

    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    return results


if __name__ == "__main__":
    vector_store()
