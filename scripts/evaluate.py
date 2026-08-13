
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

RATINGS_PATH = "data/raw/ratings.csv"
TEST_USER_SAMPLE = 500
LIKED_RATE = 4
TOP_K = 5

def split_train_test(ratings):
    train , test = [], []

    for id , group in ratings.groupby("user_id"):
        if len(group) < 5:
            continue  

        all_data = group.sample(frac=1, random_state=42)
        split_point = int(len(group) * 0.8)

        train.append(all_data.iloc[:split_point])
        test.append(all_data.iloc[split_point:])   
    return pd.concat(train) , pd.concat(test) 


def build_similarity_matrix(train_df):

    user = train_df["user_id"].astype("category").cat.codes
    book = train_df["book_id"].astype("category")
    book_codes = book.cat.codes


    book_map = dict(enumerate(book.cat.categories))

    matrix = csr_matrix((train_df["rating"],(book_codes, user)))

    similarity = cosine_similarity(matrix , dense_output=False)

    return similarity , book_map



def precision_for_user(user_id, recommended_ids, test_df):
    actual_liked = set(
        test_df[(test_df["user_id"] == user_id) & (test_df["rating"] >= LIKED_RATE)]["book_id"]
    )

    if not recommended_ids:
        return None

    hits = len(recommended_ids & actual_liked)
    return hits / len(recommended_ids)



#Recommends top_k books for a user based on books they liked in training data
def recommend_for_user(user_id, train_df, similarity, book_id_map, top_k=5):

    book_idx = {v: k for k, v in book_id_map.items()}

    liked_books = train_df[(train_df["user_id"]== user_id)& (train_df["rating"]>= LIKED_RATE)]["book_id"]

    if liked_books.empty:
        return set()


    # collect similarity scores of all books wich this user liked or rated 

    similarity_scores = {}

    for book_id in liked_books:

        if book_id not in book_idx:
            continue
        idx = book_idx[book_id]

        scores = similarity[idx].toarray().flatten()

        top_indices = scores.argsort()[::-1][1:20]

        for i in top_indices:
            recommended_book = book_id_map[i]
            similarity_scores[recommended_book] = similarity_scores.get(recommended_book, 0) + scores[i]
        
    # don't recommend books the user already saw in training
    already_seen = set(train_df[train_df["user_id"] == user_id]["book_id"])
    candidates = {b: s for b, s in similarity_scores.items() if b not in already_seen}

    top_books = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return set(book_id for book_id, _ in top_books)
        

def evaluate_precision_at_k():
    ratings = pd.read_csv(RATINGS_PATH)
    train_df, test_df = split_train_test(ratings)
    similarity, book_id_map = build_similarity_matrix(train_df)

    test_users = test_df["user_id"].unique()
    sample_users = np.random.RandomState(42).choice(
        test_users, size=min(TEST_USER_SAMPLE, len(test_users)), replace=False
    )

    precisions = []
    for user_id in sample_users:
        recommended_ids = recommend_for_user(user_id, train_df, similarity, book_id_map, TOP_K)
        precision = precision_for_user(user_id, recommended_ids, test_df)
        if precision is not None:
            precisions.append(precision)

    avg_precision = np.mean(precisions) if precisions else 0.0

    print(f"Evaluated on {len(precisions)} users")
    print(f"Precision@{TOP_K}: {avg_precision:.4f}")

    return avg_precision


if __name__ == "__main__":
    evaluate_precision_at_k()




