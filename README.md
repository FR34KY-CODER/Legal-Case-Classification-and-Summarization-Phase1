Below is a **clean, modern, recruiter-friendly GitHub README** based on your architecture story.
It’s written to *look premium*, like something a top ML engineer would ship.
You can copy–paste directly into `README.md`.

---

# ⚖️ Legal Document Classifier + Summarizer + RAG Search

*A Zero-Shot Semantic Pipeline for Classifying, Summarizing, and Querying Long Legal Documents*

---

## 🚀 Overview

This project is an end-to-end **AI workflow for legal intelligence**:

* Zero-shot **semantic classification**
* Hybrid **extractive + abstractive summarization**
* **RAG-powered question answering** with vector search

Designed especially for **long judgments**, **case files**, and **statutory documents**.

---

## 🏛️ System Architecture (The 4-Stage Story)

![Image](https://media.licdn.com/dms/image/v2/D5612AQH-AcQSsJT7Wg/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1722023278165?e=2147483647\&t=W9khkDqas5ZQKLvmpUEDf4hA8PB61BDpvfN1ahfcI-M\&v=beta\&utm_source=chatgpt.com)

![Image](https://www.ibm.com/adobe/dynamicmedia/deliver/dm-aid--ba8a3265-c815-4c0d-a9ea-8381274dcc66/rag-product-mapping.png?preferwebp=true\&utm_source=chatgpt.com)

![Image](https://www.researchgate.net/publication/381969524/figure/fig1/AS%3A11431281258496885%401720100285191/The-overall-pipeline-of-data-processing-storage-in-vector-database-and-querying.ppm?utm_source=chatgpt.com)

### **Stage 1 — Semantic Vectorization & Taxonomy (The Classifier)**

Traditional classifiers need thousands of labeled samples.
**This system doesn’t.**

It uses a **Zero-Shot Semantic Classifier** powered by embeddings.

#### 🔹 Preprocessing

* Convert raw text into structured **TOON JSON format**
* **Chunk** long documents (because BERT-style models have a 512-token limit)

#### 🔹 Embeddings

Using: **BAAI/bge-small-en**

* Top performer on the **MTEB Benchmark**
* Light enough to **run locally**
* Often outperforms older OpenAI embedding models

#### 🔹 Automated Taxonomy

* 65+ legal categories are embedded into vectors
* Compute **cosine similarity** of each chunk to each category
* Apply **Max-Pooling Category Assignment**:

  > *If even one chunk strongly signals “Murder”, the whole document is classified as Murder.*

---

### **Stage 2 — Hybrid Summarization (The Reader)**

Legal docs require **accuracy + readability**.
To avoid hallucinations, the summarizer uses a **hybrid pipeline**:

#### 🔹 Extractive (LexRank)

Captures the core facts using mathematical sentence centrality.

#### 🔹 Abstractive (BART)

Transforms facts into a polished, human-like summary.

The combination ensures the output is **smooth but grounded in truth**.

---

### **Stage 3 — RAG + Semantic Search (The Brain)**

All processed chunks are stored in a **ChromaDB** vector database.

Workflow:

1. **User asks a question**
2. System performs **semantic retrieval**
3. Retrieved chunks + prompt → **LLM**
4. LLM produces a grounded, context-aware answer

This becomes the Q&A brain of the system.

---

## 🔍 Deep Dive: Why **LexRank** Instead of TextRank?

![Image](https://image1.slideserve.com/2611552/differences-between-lexrank-and-textrank-l.jpg?utm_source=chatgpt.com)

![Image](https://www.researchgate.net/profile/Chinedu-Mbonu/publication/360354355/figure/fig2/AS%3A1151893774581769%401651644285830/Performance-results-for-the-results-TextRank-and-LexRank-algorithms-compared-with-the_Q320.jpg?utm_source=chatgpt.com)

![Image](https://www.researchgate.net/publication/308765988/figure/fig3/AS%3A439448136622082%401481784008285/Comparison-between-LexRank-and-Modified-LexRank-in-terms-of-F-measure-Precision-Recall.png?utm_source=chatgpt.com)

![Image](https://www.researchgate.net/publication/342116954/figure/fig3/AS%3A961651830444066%401606287071742/Rouge-1-comparison-of-the-precision-recall-and-F-measure-value-of-TextRank-Luhns.png?utm_source=chatgpt.com)

> “TextRank relies on **word overlap**, which fails in legal documents where long sentences share common boilerplate words (‘plaintiff’, ‘court’, ‘order’).
> LexRank uses **TF-IDF + cosine similarity**, making rare legal terms more influential and finding the true **centroid** sentence that represents the document’s core meaning.”

Benefits:

* ✔ Highlights rare but meaningful legal terms
* ✔ Identifies the **central holding / verdict**
* ✔ Avoids selecting long meaningless sentences

---

## 🧠 Deep Dive: Hallucination Control via Hybrid Design

> “Legal summarization must be hallucination-free.
> Pure abstractive models (like BART/GPT) may invent dates or sections if fed noisy inputs.
> LexRank extracts the top 20 factual sentences first, and BART is constrained to rewrite **only those**.
> This makes the summary polished **but mathematically grounded**.”

Why Hybrid?

* ✔ Eliminates noise from 50+ page judgments
* ✔ Guarantees factual consistency
* ✔ Produces human-readable summaries without risk

---

## 🏗️ Tech Stack

| Component                 | Technology            |
| ------------------------- | --------------------- |
| Embeddings                | **BAAI/bge-small-en** |
| Extractive Summarization  | **LexRank**           |
| Abstractive Summarization | **BART**              |
| Vector DB                 | **ChromaDB**          |
| Similarity Metric         | **Cosine Similarity** |
| RAG Pipeline              | Custom implementation |

---

## 📁 Folder Structure

```
├── data/
├── preprocessing/
│   ├── toon_converter.py
│   ├── chunker.py
├── embeddings/
│   └── embedder.py
├── taxonomy/
│   └── classifier.py
├── summarizer/
│   ├── lexrank_extractor.py
│   └── bart_summarizer.py
├── rag/
│   ├── chroma_store.py
│   └── retrieval.py
└── app/
    └── main.py
```

---

## ▶️ How It Works (In 30 Seconds)

1. Convert raw legal PDF → TOON JSON
2. Chunk + embed using BGE-Small
3. Vector similarity → assign category
4. LexRank → extract facts
5. BART → abstract summary
6. ChromaDB → store chunks
7. Ask question → retrieve relevant chunks
8. LLM → grounded answer

---

## 🎯 Ideal For

* Legal-tech startups
* Court document analysis
* Compliance automation
* Case-law retrieval systems
* Enterprise search solutions

---
