# 📚 What to Read Next

**A hybrid recommendation engine combining semantic search and collaborative filtering to surface personalized book recommendations.**

This project implements two complementary recommendation strategies — dense vector retrieval over book embeddings, and item-based collaborative filtering over user rating history — and combines them into a single weighted ranking system. Built as an end-to-end demonstration of the recommendation systems pipeline: data ingestion, embedding generation, vector indexing, similarity computation, hybrid ranking, and offline evaluation.

## Demo: 





https://github.com/user-attachments/assets/5704bc0d-eb29-4baf-b96a-2fcd3e0cc8d0

https://github.com/user-attachments/assets/1bf6567e-663d-4153-9f19-1e8589a04afe






## Screenshots
<p align="center">
  <img src="https://github.com/user-attachments/assets/5538d1b3-b456-4a97-a14d-2b10a413f089" width="30%">
  <img src="https://github.com/user-attachments/assets/fe2d4c68-9b4d-4613-a337-cf8d65953350" width="30%">
  <img src="https://github.com/user-attachments/assets/ccff59ab-b56a-4251-8a72-8fc89f58b32c" width="30%">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/181cd296-d809-4d46-8288-8a2d31402fa4" width="50%">
  <img src="https://github.com/user-attachments/assets/34a3370c-9300-425f-849d-2ef822d18137" width="50%">
</p>


---

## System Overview


| Feature | Project Implementation |
|---|---|
| **Embedding model** | OpenAI `text-embedding-3-small` (1536 dimensions) |
| **Vector index** | ChromaDB (HNSW-based approximate nearest neighbor search) |
| **Collaborative filtering** | Item-item cosine similarity over a sparse user-rating matrix |
| **UI** | Streamlit |
| **Dataset** | [goodbooks-10k-extended](https://github.com/malcolmosh/goodbooks-10k-extended) (~10K books, ~6M ratings) |
---

## Architecture

```
                          ┌─────────────────────────┐
                          │   Raw book metadata      │
                          │   (title, author,        │
                          │    genres, description)  │
                          └────────────┬─────────────┘
                                       │
                          Text normalization + field
                          concatenation (embed_text)
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │  OpenAI Embeddings API    │
                          │  text-embedding-3-small   │
                          └────────────┬─────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │  ChromaDB (persistent)    │
                          │  1536-dim vectors +       │
                          │  metadata payload          │
                          └────────────┬─────────────┘
                                       │
                    ┌──────────────────┴───────────────────┐
                    ▼                                        ▼
        Query-time semantic search                Item-based CF
        (cosine similarity, top-N)                 (offline-computed
                    │                                book-book similarity
                    │                                matrix from ratings.csv)
                    │                                        │
                    └──────────────────┬───────────────────┘
                                       ▼
                          ┌─────────────────────────┐
                          │      Hybrid Ranker         │
                          │  score = α·sem + (1-α)·cf  │
                          └────────────┬─────────────┘
                                       ▼
                                Streamlit UI
```

**Two retrieval modes, one ranking layer:**
- *Genre/description queries* run through semantic retrieval only — this path requires no rating history, avoiding the cold-start problem entirely.
- *"Books you love" queries* run through both retrieval paths in parallel, then merge on a common `book_id` key before scoring.

---

## Implementation 

### Embedding 
Each book is represented as a single concatenated text field before embedding:
```
Title: {title}. Author: {authors}. Genres: {genres}. Description: {description}
```

### Collaborative filtering
User ratings are turned into a sparse `(n_books × n_users)` matrix (`scipy.sparse.csr_matrix`)  . Book-book similarity is computed as pairwise cosine similarity over this matrix: books rated similarly by the same users converge on a high similarity score, independent of any text or content signal.

### Hybrid scoring
For a given book, semantic and collaborative scores are combined as:

```
final_score = semantic_weight · semantic_score + (1 - semantic_weight) · collab_score
```

`semantic_weight` defaults : `0.6` and is a adjustable parameter in the UI, so retrieval behavior can be seen

### Multi-book queries
When multiple liked books are provided, per-book hybrid scores are averaged across all input books for users that surface as a strong match across more than one input book .

---

## Evaluation

The collaborative filtering component was evaluated offline using a held-out test split, measuring **Precision@K**.

For ~500 sampled users, ratings were split 80/20 (train/test) per user. Using only the training split the model recommends the top-K most similar books to what the user rated ≥4 stars in training. 

| Metric | Score | Random Baseline |
|---|---|---|
| Precision@5  | **0.30** | ~0.05–0.1% |
| Precision@10 | **0.22** | ~0.05–0.1% |

**Why precision decreases as K increases:** Recommendations are ranked by similarity score, so the top 5 are always the model's strongest matches. Extending the list to 10 adds lower-confidence candidates (ranks 6–10), which pulls the average down — this is expected, not a sign the model is getting worse. Both scores still land 200–300x above random chance. The app defaults to K=5 to prioritize precision over coverage.

Reproduce:
```bash
python -m scripts.evaluate            # Precision@5
python -m scripts.evaluate_5_vs_10    # Precision@5 vs Precision@10 comparison
```

---

## Project Structure

```
what-to-read-next/
├── app.py                      # Streamlit UI
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                    # source goodbooks-10k-extended files
│   └── processed/              # cleaned data with constructed embed_text
├── src/
│   ├── data_prep.py            # cleaning, field normalization, embed_text construction
│   ├── embeddings.py           # OpenAI embedding client wrapper
│   ├── vector_store.py         # ChromaDB indexing + semantic search
│   ├── collaborative.py        # sparse rating matrix + item-item similarity
│   └── hybrid_ranker.py        # score fusion, multi-book aggregation
├── scripts/
│   ├── evaluate.py             # Precision@5 offline evaluation
│   └── evaluate_5_vs_10.py     # Precision@K comparison across K values
└── db/chroma/                  # persisted vector index (gitignored, generated locally)
```

---

## Setup

Requires an OpenAI API key (for embedding generation).

```bash
git clone https://github.com/elfakb/what-to-read-next.git
cd what-to-read-next
pip install -r requirements.txt
```

Add your key to a `.env` file in the project root:
```
OPENAI_API_KEY=your-key-here
```

Download `books_enriched.csv` from [goodbooks-10k-extended](https://github.com/malcolmosh/goodbooks-10k-extended) and `ratings.csv` from [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) into `data/raw/`.

Build the pipeline:
```bash
python -m src.data_prep       # clean data, construct embed_text
python -m src.vector_store    # generate embeddings, populate ChromaDB (one-time, ~10K API calls)
```

Run:
```bash
streamlit run app.py
```

---

## Tech / Credits

Built on the [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) and [goodbooks-10k-extended](https://github.com/malcolmosh/goodbooks-10k-extended) datasets.
