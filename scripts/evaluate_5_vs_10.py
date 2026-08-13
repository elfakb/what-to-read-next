
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

RATINGS_PATH = "data/raw/ratings.csv"
TEST_USER_SAMPLE = 500
LIKED_RATE = 4
K_VALUES = [5, 10]  # artık birden fazla K değerini aynı anda test ediyoruz


def split_train_test(ratings):
    train, test = [], []

    for user_id, group in ratings.groupby("user_id"):
        if len(group) < 5:
            continue

        all_data = group.sample(frac=1, random_state=42)
        split_point = int(len(group) * 0.8)

        train.append(all_data.iloc[:split_point])
        test.append(all_data.iloc[split_point:])
    return pd.concat(train), pd.concat(test)


def build_similarity_matrix(train_df):
    user = train_df["user_id"].astype("category").cat.codes
    book = train_df["book_id"].astype("category")
    book_codes = book.cat.codes

    book_map = dict(enumerate(book.cat.categories))

    matrix = csr_matrix((train_df["rating"], (book_codes, user)))
    similarity = cosine_similarity(matrix, dense_output=False)

    return similarity, book_map


def precision_for_user(recommended_ids, actual_liked):
    if not recommended_ids:
        return None
    hits = len(recommended_ids & actual_liked)
    return hits / len(recommended_ids)


# top_k artık parametre, tek bir sabit sayı değil - böylece aynı fonksiyonu
# hem K=5 hem K=10 için tekrar tekrar çağırabiliyoruz
def recommend_for_user(user_id, train_df, similarity, book_id_map, top_k):
    book_idx = {v: k for k, v in book_id_map.items()}

    liked_books = train_df[(train_df["user_id"] == user_id) & (train_df["rating"] >= LIKED_RATE)]["book_id"]

    if liked_books.empty:
        return set()

    similarity_scores = {}

    for book_id in liked_books:
        if book_id not in book_idx:
            continue
        idx = book_idx[book_id]
        scores = similarity[idx].toarray().flatten()
        # not: aday havuzunu en büyük top_k'ya göre yeterince geniş tutuyoruz (K=10 için en az 20 aday lazım)
        top_indices = scores.argsort()[::-1][1:max(K_VALUES) * 4]

        for i in top_indices:
            recommended_book = book_id_map[i]
            similarity_scores[recommended_book] = similarity_scores.get(recommended_book, 0) + scores[i]

    already_seen = set(train_df[train_df["user_id"] == user_id]["book_id"])
    candidates = {b: s for b, s in similarity_scores.items() if b not in already_seen}

    top_books = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return set(book_id for book_id, _ in top_books)


def evaluate_precision_at_multiple_k():
    ratings = pd.read_csv(RATINGS_PATH)
    train_df, test_df = split_train_test(ratings)
    similarity, book_id_map = build_similarity_matrix(train_df)

    test_users = test_df["user_id"].unique()
    sample_users = np.random.RandomState(42).choice(
        test_users, size=min(TEST_USER_SAMPLE, len(test_users)), replace=False
    )

    # her K değeri için ayrı bir precision listesi tutuyoruz
    precisions_by_k = {k: [] for k in K_VALUES}

    for user_id in sample_users:
        actual_liked = set(
            test_df[(test_df["user_id"] == user_id) & (test_df["rating"] >= LIKED_RATE)]["book_id"]
        )

        for k in K_VALUES:
            recommended_ids = recommend_for_user(user_id, train_df, similarity, book_id_map, top_k=k)
            precision = precision_for_user(recommended_ids, actual_liked)
            if precision is not None:
                precisions_by_k[k].append(precision)

    print(f"Evaluated on up to {len(sample_users)} users\n")
    for k in K_VALUES:
        avg = np.mean(precisions_by_k[k]) if precisions_by_k[k] else 0.0
        print(f"Precision@{k}: {avg:.4f}  (n={len(precisions_by_k[k])} users)")

    return precisions_by_k


if __name__ == "__main__":
    evaluate_precision_at_multiple_k()