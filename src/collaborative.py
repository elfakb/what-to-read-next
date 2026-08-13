import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Item-Based Collaborative Filtering.
def build_item_similarity():

    ratings = pd.read_csv("data/raw/ratings.csv")

    # user matrix 
    user_ids = ratings["user_id"].astype("category").cat.codes
    book_ids = ratings["book_id"].astype("category").cat.codes
    book_id_map = dict(enumerate(ratings["book_id"].astype("category").cat.categories))

    # sparse matrix: rows = books, columns = users, values = ratings
    matrix = csr_matrix((ratings["rating"],(book_ids , user_ids))) # sparse matrix for ratings 


    # itam based clustered filtering 
    # cosine similarity between every pair of books, based on how users rated them
    similarity = cosine_similarity(matrix, dense_output=False) # calculating similarites of the sparse matrix based on book_ids 1 book row = all users ratings column
    return similarity , book_id_map

def get_similar_books(book_id, similarity, book_id_map, n=5):
    book_id = int(book_id)  
    books = {v: k for k, v in book_id_map.items()}

    if book_id not in books:
        return []

    idx = books[book_id]

    # get this id of books similar
    similarity_of_book= similarity[idx]

    # converting to array 
    array_of_similars= similarity_of_book.toarray()

    scores = array_of_similars.flatten()

    scores_series = pd.Series(scores)
    top = scores_series.sort_values(ascending=False)[1:n + 1]

    return [(book_id_map[i], score) for i, score in top.items()]


if __name__ == "__main__":
    similarity, book_id_map = build_item_similarity()
    print(get_similar_books(book_id=1, similarity=similarity, book_id_map=book_id_map))