# app.py

import streamlit as st
import pandas as pd
from src.hybrid_ranker import recommend_by_text, recommend_by_book, recommend_by_multiple_books


st.set_page_config(page_title="What to Read Next", page_icon="📚", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    .main {
        background-color: #FFF9F2;
    }

    /* hero header */
    .hero {
        background: linear-gradient(135deg, #F5EBE0 0%, #FFF9F2 100%);
        border-radius: 18px;
        padding: 32px 36px;
        margin-bottom: 28px;
        border: 1px solid #F0E4D7;
    }
    .hero-title {
        font-size: 34px;
        font-weight: 800;
        color: #4A3F35;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 15px;
        color: #8C7B6B;
    }

    /* book cards */
    .book-card {
        background-color: #FFFFFF;
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #F0E4D7;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .book-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(176,137,104,0.18);
    }
    .book-title { font-size: 16px; font-weight: 700; color: #4A3F35; margin-bottom: 4px; }
    .book-author { font-size: 13px; color: #8C7B6B; margin-bottom: 6px; }
    .stars { font-size: 14px; color: #B08968; margin-bottom: 4px; }
    .reason-text { font-size: 12px; color: #6B5D4F; font-style: italic; }

    .hint-text { font-size: 13px; color: #8C7B6B; margin-bottom: 10px; }

    /* empty state */
    .empty-state {
        text-align: center;
        padding: 50px 20px;
        color: #B0A493;
    }
    .empty-state-icon { font-size: 44px; margin-bottom: 10px; }

    /* footer */
    .footer {
        text-align: center;
        color: #B0A493;
        font-size: 12px;
        padding: 30px 0 10px 0;
        border-top: 1px solid #F0E4D7;
        margin-top: 40px;
    }

    /* streamlit tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F5EBE0;
        border-radius: 10px;
        padding: 8px 18px;
        color: #8C7B6B;
    }
    .stTabs [aria-selected="true"] {
        background-color: #B08968 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("""
<div class="hero">
    <div class="hero-title">📚 What to Read Next</div>
    <div class="hero-subtitle">Tell us what you're in the mood for, and we'll find your next favorite book — matched to your taste, not just keywords.</div>
</div>
""", unsafe_allow_html=True)


# ---------- DATA ----------
@st.cache_data
def load_books():
    return pd.read_csv("data/processed/books_processed.csv")


@st.cache_data
def get_popular_genres(df, top_n=10):
    counts = {}
    for g in df["genres_str"].dropna():
        for genre in g.split(","):
            genre = genre.strip()
            if genre:
                counts[genre] = counts.get(genre, 0) + 1
    return [g for g, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]]


books_df = load_books()
popular_genres = get_popular_genres(books_df,top_n=20)

EXAMPLE_QUERIES = [
    "dark and psychological with a twist ending",
    "cozy, feel-good story with a happy ending",
    "fast-paced adventure with a strong female lead",
    "slow-burn romance with emotional depth",
    "mind-bending sci-fi that questions reality",
]

DEFAULT_QUERY = "highly rated, popular, well-loved book"


# ---------- HELPERS ----------x
def scores_to_stars(scores: list):
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    stars_list = []
    for s in scores:
        if max_s == min_s:
            stars = 5
        else:
            normalized = (s - min_s) / (max_s - min_s)
            stars = round(1 + normalized * 4)
        stars = max(1, min(5, stars))
        stars_list.append("⭐" * stars + "☆" * (5 - stars))
    return stars_list


def get_book_detail(title):
    row = books_df[books_df["title"] == title]
    if row.empty:
        return None
    row = row.iloc[0]

    detail = {"description": row.get("description", "No description available.")}

    for col in ["average_rating"]:
        if col in row and pd.notna(row[col]):
            detail["average_rating"] = row[col]

    for col in ["pages", "num_pages"]:
        if col in row and pd.notna(row[col]):
            detail["pages"] = row[col]
            break

    for col in ["publishDate", "original_publication_year", "publication_year"]:
        if col in row and pd.notna(row[col]):
            detail["year"] = row[col]
            break

    return detail


def show_books_grid(results, score_key="score", key_prefix=""):
    scores = [book.get(score_key, 0) for book in results]
    stars_list = scores_to_stars(scores)

    cols = st.columns(5)
    for i, book in enumerate(results):
        with cols[i % 5]:
            image_url = book.get("image_url")
            if image_url and pd.notna(image_url) and str(image_url).startswith("http"):
                st.image(image_url, width=120)
            else:
                st.markdown("<div style='font-size:40px;'>📕</div>", unsafe_allow_html=True)

            st.markdown(f"**{book['title']}**")
            st.caption(f"✍️ {book.get('authors', '')}")
            st.markdown(f"<div class='stars'>{stars_list[i]}</div>", unsafe_allow_html=True)
            st.caption(f"💡 {book.get('reason', '')}")

            with st.expander("📖 Details"):
                detail = get_book_detail(book["title"])
                if detail:
                    if "average_rating" in detail:
                        st.caption(f"⭐ Average Rating: {detail['average_rating']:.2f}")
                    if "pages" in detail:
                        st.caption(f"📄 Pages: {int(detail['pages'])}")
                    if "year" in detail:
                        st.caption(f"📅 Published: {detail['year']}")

                    desc = str(detail.get("description", ""))
                    st.write(desc[:400] + ("..." if len(desc) > 400 else ""))
                else:
                    st.write("No details available for this book.")


# ---------- TABS ----------
tab1, tab2 = st.tabs(["🔍  Search by Genre & Description", "📖  Recommend from Books You Love"])

# ---------- TAB 1 ----------
# ---------- TAB 1 ----------
# ---------- TAB 1 ----------
with tab1:
    st.markdown(
        "<p class='hint-text'>Pick a genre, describe the vibe you're after, or both — it's up to you.</p>",
        unsafe_allow_html=True
    )

    if "selected_genres" not in st.session_state:
        st.session_state.selected_genres = []

    st.markdown("<p class='hint-text' style='margin-bottom:6px;'>Genres</p>", unsafe_allow_html=True)

    all_genre_chips = get_popular_genres(books_df, top_n=60)

    cols_per_row = 6
    for row_start in range(0, len(all_genre_chips), cols_per_row):
        row_genres = all_genre_chips[row_start:row_start + cols_per_row]
        row_cols = st.columns(cols_per_row)
        for col, genre in zip(row_cols, row_genres):
            with col:
                is_selected = genre in st.session_state.selected_genres
                label = f"✓ {genre}" if is_selected else genre
                if st.button(label, key=f"chip_{genre}"):
                    if is_selected:
                        st.session_state.selected_genres.remove(genre)
                    else:
                        st.session_state.selected_genres.append(genre)
                    st.rerun()

    selected_genres = st.session_state.selected_genres

    st.write("")

    query = st.text_input(
        "What kind of story are you looking for? (optional)",
        placeholder="e.g. dark, psychological, few characters",
        key="query_input"
    )
    st.markdown(
        "<p class='hint-text'>💡 Try: " +
        " · ".join(f"<i>{ex}</i>" for ex in EXAMPLE_QUERIES[:3]) +
        "</p>",
        unsafe_allow_html=True
    )

    search_clicked = st.button("✨ Recommend", type="primary", key="text_search")

    if search_clicked:
        if not selected_genres and not query.strip():
            effective_query = DEFAULT_QUERY
            st.info("No genre or description given — showing generally well-loved books.")
        elif not query.strip():
            effective_query = ", ".join(selected_genres)
        else:
            effective_query = query.strip()

        with st.spinner("Finding the best books for you..."):
            results = recommend_by_text(effective_query, genres=selected_genres or None, n=5)

        if not results:
            st.warning("No books matched these criteria. Try fewer genres or a different description.")
        else:
            st.write("")
            show_books_grid(results, score_key="score", key_prefix="text")
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📚</div>
            Your recommendations will appear here.<br>
            Pick a genre, type a vibe, or both — then hit Recommend.
        </div>
        """, unsafe_allow_html=True)
# ---------- TAB 2 ----------
with tab2:
    st.write("Pick the books you loved, and we'll recommend similar ones:")

    selected_titles = st.multiselect(
        "Books you love (you can pick more than one)",
        options=books_df["title"].tolist(),
        placeholder="Search and add a book..."
    )

    if selected_titles:
        st.write("Your picks:")
        preview_cols = st.columns(min(len(selected_titles), 6))
        for i, title in enumerate(selected_titles[:6]):
            row = books_df[books_df["title"] == title].iloc[0]
            img = row.get("image_url", "")
            with preview_cols[i % 6]:
                if img and isinstance(img, str) and img.startswith("http"):
                    st.image(img, width=70)
                st.caption(title[:22] + ("..." if len(title) > 22 else ""))

    st.write("")

    with st.expander("⚙️ Fine-tune how recommendations are made"):
        semantic_weight = st.slider(
            "Content similarity ↔ Reader behavior",
            min_value=0.0, max_value=1.0, value=0.6, step=0.1,
            help="Left = recommend based on plot/theme similarity. Right = recommend based on what similar readers enjoyed."
        )
    collab_weight = 1.0 - semantic_weight

    recommend_clicked = st.button("✨ Recommend Similar Books", type="primary", key="book_search")

    if recommend_clicked:
        if not selected_titles:
            st.warning("Please select at least one book.")
        else:
            selected_ids = [
                int(books_df[books_df["title"] == t]["book_id"].iloc[0])
                for t in selected_titles
            ]

            with st.spinner("Building recommendations based on your picks..."):
                results = recommend_by_multiple_books(
                    selected_ids, selected_titles, books_df, n=5,
                    semantic_weight=semantic_weight,
                    collab_weight=collab_weight
                )

            if not results:
                st.warning("No recommendations found.")
            else:
                st.write("")
                show_books_grid(results, score_key="final_score", key_prefix="multi")
    elif not selected_titles:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">💭</div>
            Select a few books you love above,<br>
            and we'll find what to read next.
        </div>
        """, unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
    Built with OpenAI Embeddings · ChromaDB · Collaborative Filtering · Streamlit<br>
    <a href="https://github.com/" style="color:#B08968;">View on GitHub</a>
</div>
""", unsafe_allow_html=True)