import pandas as pd
from ast import literal_eval

load_dotenv()

RAW_PATH = "data/raw/books_enriched.csv"
OUT_PATH = "data/processed/books_processed.csv"

def data_prep():

    # raw data should be in raw/data directory 
    books = pd.read_csv(RAW_PATH,index_col=[0], converters={"genres": literal_eval})

    # data cleaning 
    books = books[books["description"].notna()]
    books = books[books["title"].notna()]
    books = books[books["authors"].notna()]

    books["genres_str"] = books["genres"].apply(lambda g: ", ".join(g) if g else "")

    # 1 text for embedding metadata 
    books["embed_text"] = (
        "Title: " + books["title"].astype(str) +
        ". Author: " + books["authors"].astype(str) +
        ". Genres: " + books["genres_str"] +
        ". Description: " + books["description"].astype(str)
    )

    books.to_csv(OUT_PATH,index = False)
    print(f"{len(books)} books saved to -> {OUT_PATH}")
    return books

if __name__ =="__main__":
    data_prep()
