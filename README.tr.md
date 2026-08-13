🌐 **[English](README.md)** | [Türkçe](README.tr.md)
# 📚 What to Read Next

**Kişiselleştirilmiş kitap önerileri sunmak için semantic search ve collaborative filtering'i birleştiren hibrit bir öneri motoru.**

Bu proje iki tamamlayıcı öneri stratejisini bir araya getiriyor — kitap embedding'leri üzerinde dense vector retrieval ve kullanıcı puanlama geçmişi üzerinde item-based collaborative filtering — ve bunları tek bir ağırlıklı sıralama sistemi altında birleştiriyor. Öneri sistemleri pipeline'ının uçtan uca bir gösterimi olarak geliştirildi: veri toplama, embedding üretimi, vektör indeksleme, benzerlik hesaplama, hibrit sıralama ve offline değerlendirme.

## Demo:

https://github.com/user-attachments/assets/5704bc0d-eb29-4baf-b96a-2fcd3e0cc8d0

https://github.com/user-attachments/assets/1bf6567e-663d-4153-9f19-1e8589a04afe




## Screenshots
<p align="center">
  <img src="https://github.com/user-attachments/assets/5538d1b3-b456-4a97-a14d-2b10a413f089" width="40%">
  <img src="https://github.com/user-attachments/assets/fe2d4c68-9b4d-4613-a337-cf8d65953350" width="40%">
  <img src="https://github.com/user-attachments/assets/ccff59ab-b56a-4251-8a72-8fc89f58b32c" width="50%">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/181cd296-d809-4d46-8288-8a2d31402fa4" width="50%">
  <img src="https://github.com/user-attachments/assets/34a3370c-9300-425f-849d-2ef822d18137" width="50%">
</p>


---

## Sistem Özeti

| Bileşen | Kullanılan Teknoloji |
|---|---|
| Embedding modeli | OpenAI `text-embedding-3-small` (1536 boyutlu) |
| Vektör indeksi | ChromaDB (HNSW tabanlı approximate nearest neighbor arama) |
| Collaborative sinyal | Seyrek kullanıcı-puan matrisi üzerinde item-item cosine similarity |
| Arayüz | Streamlit |
| Veri seti | [goodbooks-10k-extended](https://github.com/malcolmosh/goodbooks-10k-extended) (~10K kitap, ~6M puan) |

---

## Mimari

```
                          ┌─────────────────────────┐
                          │   Ham kitap verisi        │
                          │   (başlık, yazar,         │
                          │    tür, açıklama)         │
                          └────────────┬─────────────┘
                                       │
                          Metin normalizasyonu +
                          alan birleştirme (embed_text)
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
                          │  1536 boyutlu vektörler +  │
                          │  metadata                 │
                          └────────────┬─────────────┘
                                       │
                    ┌──────────────────┴───────────────────┐
                    ▼                                        ▼
        Sorgu anında semantic arama                Item-based CF
        (cosine similarity, top-N)                 (offline hesaplanmış
                    │                                kitap-kitap benzerlik
                    │                                matrisi, ratings.csv'den)
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

**İki retrieval modu, tek sıralama katmanı:**
- *Tür/açıklama sorguları* yalnızca semantic retrieval üzerinden çalışır — bu yol hiç puanlama geçmişi gerektirmez, cold-start problemini tamamen ortadan kaldırır.
- *"Sevdiğim kitaplar" sorguları* her iki retrieval yolunu paralel çalıştırır, ardından ortak `book_id` anahtarı üzerinden birleştirip skorlar.

---

## Uygulama Detayları

### Embedding oluşturma
Her kitap, embed edilmeden önce tek bir birleşik metin alanı olarak temsil ediliyor:
```
Title: {title}. Author: {authors}. Genres: {genres}. Description: {description}
```

### Collaborative filtering
Kullanıcı puanları seyrek bir `(n_kitap × n_kullanıcı)` matrisine (`scipy.sparse.csr_matrix`) dönüştürülüyor. Kitap-kitap benzerliği bu matris üzerinde ikili cosine similarity olarak hesaplanıyor: aynı kullanıcılar tarafından benzer şekilde puanlanan kitaplar, herhangi bir metin/içerik sinyalinden bağımsız olarak yüksek benzerlik skoruna yakınsıyor.

### Hybrid scoring
Verilen bir kitap için semantic ve collaborative skorları şöyle birleştiriliyor:

```
final_score = semantic_weight · semantic_score + (1 - semantic_weight) · collab_score
```

`semantic_weight` varsayılan olarak `0.6` ve arayüzde ayarlanabilir bir parametre, böylece retrieval davranışı canlı olarak gözlemlenebiliyor.

### Çoklu kitap sorguları
Birden fazla sevilen kitap girildiğinde, her kitap için hesaplanan hybrid skorlar tüm girdi kitapları üzerinden ortalanıyor; birden fazla girdi kitabında da güçlü eşleşme olarak öne çıkan adaylar için bu bir güç sinyali oluyor.

---

## Değerlendirme (Evaluation)

Collaborative filtering bileşeni, held-out bir test seti kullanılarak offline olarak değerlendirildi, **Precision@K** metriği ölçüldü.

~500 örneklem kullanıcı için, puanlar kullanıcı başına %80/%20 (train/test) olarak bölündü. Yalnızca train verisi kullanılarak, model kullanıcının train setinde ≥4 yıldız verdiği kitaplara en benzer top-K kitabı öneriyor.

| Metrik | Skor | Rastgele Baseline |
|---|---|---|
| Precision@5  | **0.30** | ~%0.05–0.1 |
| Precision@10 | **0.22** | ~%0.05–0.1 |

**K arttıkça precision neden düşüyor:** Öneriler benzerlik skoruna göre sıralanıyor, yani ilk 5 her zaman modelin en güçlü eşleşmeleri. Listeyi 10'a çıkarmak, daha düşük güvenilirlikte adayları (6-10. sıradakiler) da eklemek anlamına geliyor, bu da ortalamayı aşağı çekiyor — bu beklenen bir durum, modelin kötüleştiğinin göstergesi değil. Her iki skor da rastgele tahminin 200-300 katı üzerinde kalıyor. Uygulama, precision'ı kapsamdan (coverage) önceliklendirmek için varsayılan olarak K=5 kullanıyor.

Tekrar üretmek için:
```bash
python -m scripts.evaluate            # Precision@5
python -m scripts.evaluate_5_vs_10    # Precision@5 vs Precision@10 karşılaştırması
```

---

## Proje Yapısı

```
what-to-read-next/
├── app.py                      # Streamlit arayüzü
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                    # kaynak goodbooks-10k-extended dosyaları
│   └── processed/              # temizlenmiş veri + oluşturulan embed_text
├── src/
│   ├── data_prep.py            # temizlik, alan normalizasyonu, embed_text oluşturma
│   ├── embeddings.py           # OpenAI embedding client wrapper
│   ├── vector_store.py         # ChromaDB indeksleme + semantic arama
│   ├── collaborative.py        # seyrek puan matrisi + item-item benzerlik
│   └── hybrid_ranker.py        # skor birleştirme, çoklu kitap agregasyonu
├── scripts/
│   ├── evaluate.py             # Precision@5 offline değerlendirme
│   └── evaluate_5_vs_10.py     # farklı K değerleri için Precision@K karşılaştırması
└── db/chroma/                  # persist edilen vektör indeksi (gitignore'da, lokal olarak üretiliyor)
```

---

## Kurulum

OpenAI API key gerektirir (yalnızca embedding üretimi için kullanılır).

```bash
git clone https://github.com/elfakb/what-to-read-next.git
cd what-to-read-next
pip install -r requirements.txt
```

Proje kök dizininde bir `.env` dosyası oluşturup key'ini ekle:
```
OPENAI_API_KEY=your-key-here
```

`books_enriched.csv` dosyasını [goodbooks-10k-extended](https://github.com/malcolmosh/goodbooks-10k-extended)'ten ve `ratings.csv` dosyasını [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k)'ten indirip `data/raw/` klasörüne yerleştir.

Pipeline'ı kur:
```bash
python -m src.data_prep       # veriyi temizle, embed_text oluştur
python -m src.vector_store    # embedding'leri üret, ChromaDB'yi doldur (tek seferlik, ~10K API çağrısı)
```

Çalıştır:
```bash
streamlit run app.py
```

---

## Teknoloji / Kaynaklar

[goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) ve [goodbooks-10k-extended](https://github.com/malcolmosh/goodbooks-10k-extended) veri setleri kullanılarak geliştirilmiştir.
