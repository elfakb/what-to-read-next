import pandas as pd
from src.vector_store import search_books
from src.collaborative import build_item_similarity, get_similar_books

DEFAULT_SEMANTIC_WEIGHT = 0.6
DEFAULT_COLLAB_WEIGHT = 0.4


# returns books title - authors - image url 
def get_book_info(book_id , books_df):

    book_id = str(book_id)
    match = books_df[books_df["book_id"].astype(str)== book_id]

    if match.empty:
        return {"title": "Unknown Book", "authors": "", "image_url": ""}    

    row = match.iloc[0]
    return {
        "title": row.get("title", "Unknown Book"),
        "authors": row.get("authors", ""),
        "image_url": row.get("image_url", "")
    }

def recommend_by_text(query, genres, n=5):

    fetch_n = n * 4 if genres else n
    results = search_books(query, n_results=fetch_n)

    recom = []

    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        book_genres = metadata.get("genres_str", "").lower()

        if genres:
            if not all(g.lower() in book_genres for g in genres):
                continue

        recom.append({
            "book_id": results["ids"][0][i],
            "title": metadata["title"],
            "authors": metadata.get("authors", ""),
            "image_url": metadata.get("image_url", ""),
            "score": 1 - results["distances"][0][i],
            "reason": f"Matches your query and genres ({', '.join(genres) if genres else '-'})."
        })

        if len(recom) >= n:
            break
    return recom

def recommend_by_book(book_id, book_title, books_df, n=5,semantic_weight=DEFAULT_SEMANTIC_WEIGHT,collab_weight=DEFAULT_COLLAB_WEIGHT):
    semantic_results = search_books(book_title, n_results=n * 2)

    similarity, book_id_map = build_item_similarity()
    collab_results = dict(get_similar_books(book_id, similarity, book_id_map, n=n * 2))

    combined = {}

    for i in range(len(semantic_results["ids"][0])):
        bid = semantic_results["ids"][0][i]
        sem_score = 1 - semantic_results["distances"][0][i]
        meta = semantic_results["metadatas"][0][i]
        combined[bid] = {
            "title": meta.get("title"),
            "authors": meta.get("authors", ""),
            "image_url": meta.get("image_url", ""),
            "semantic_score": sem_score,
            "collab_score": 0.0
        }

    for bid, score in collab_results.items():
        bid = str(bid)
        if bid in combined:
            combined[bid]["collab_score"] = score
        else:
            info = get_book_info(bid, books_df)
            combined[bid] = {
                "title": info["title"],
                "authors": info["authors"],
                "image_url": info["image_url"],
                "semantic_score": 0.0,
                "collab_score": score
            }

    for bid, data in combined.items():
        data["final_score"] = (
            semantic_weight * data["semantic_score"] +
            collab_weight * data["collab_score"]
        )
        if data["semantic_score"] > data["collab_score"]:
            data["reason"] = "Similar in content and theme."
        else:
            data["reason"] = "Readers of this book also enjoyed it."

    ranked = sorted(combined.items(), key=lambda x: x[1]["final_score"], reverse=True)

    return ranked[:n]


def recommend_by_multiple_books(book_ids, book_titles, books_df, n=5,semantic_weight=DEFAULT_SEMANTIC_WEIGHT,collab_weight=DEFAULT_COLLAB_WEIGHT):

    
    all_results = {}

    for bid, title in zip(book_ids, book_titles):
        single_results = recommend_by_book(
            bid, title, books_df, n=n * 2,
            semantic_weight=semantic_weight,
            collab_weight=collab_weight
        )

        for candidate_id, data in single_results:
            if candidate_id in [str(b) for b in book_ids]:
                continue

            if candidate_id not in all_results:
                all_results[candidate_id] = {
                    "title": data["title"],
                    "image_url": data.get("image_url", ""),
                    "authors": data.get("authors", ""),
                    "scores": [],
                    "reasons": set()
                }

            all_results[candidate_id]["scores"].append(data["final_score"])
            all_results[candidate_id]["reasons"].add(data["reason"])

    final = []
    for bid, data in all_results.items():
        avg_score = sum(data["scores"]) / len(data["scores"])
        overlap_bonus = 0.05 * (len(data["scores"]) - 1)
        final.append({
            "book_id": bid,
            "title": data["title"],
            "authors": data["authors"],
            "image_url": data["image_url"],
            "final_score": avg_score + overlap_bonus,
            "reason": " / ".join(data["reasons"])
        })

    final.sort(key=lambda x: x["final_score"], reverse=True)
    return final[:n]



