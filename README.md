# Video Similarity Search — Embedding Architecture Benchmark

**STAI345 Generative AI — Mid-Term** · Vijaybhoomi School of Science & Technology

Query a video database with a raw video file and retrieve its nearest neighbours using
**only** the latent vector of the pixel content. Three embedding architectures are compared
head-to-head, ingested into a real vector database, and benchmarked on retrieval quality,
approximate-search recall, and query latency.

---

## The constraint, and how it is enforced

> *"You are strictly prohibited from using textual data or metadata for the similarity
> search. The search must be purely based on the latent vector representation."*

This is enforced **structurally**, not by convention:

| Boundary | Guarantee |
|---|---|
| `search(collection, model, vector, k)` | Accepts a **vector**. Returns **ids and scores**. Nothing else crosses. |
| Qdrant payload | Stores only `{"row": int}` — no label, no filename, no class. The DB cannot leak metadata it does not hold. |
| CLIP text tower | `del clip_model.text_model` — deleted outright. It cannot influence a search if it does not exist in memory. |
| Class labels | Parsed **once** from filenames to build ground truth, used **only** in offline scoring after results return. |

Labels are unavoidable: recall and mAP are undefined without knowing which results were
correct. The rule is about *where* they are allowed, and they are confined to scoring.

---

## Modality and dataset

**Video** — UCF101 human action recognition.

Filenames follow `v_<Class>_g<group>_c<clip>.avi`. The **group** field is load-bearing:
every clip within a group is cut from the *same source video*, so group-mates are near
duplicates.

**Query/gallery splitting is group-disjoint.** A random split would place group-mates on both
sides, and every retrieval score would be inflated — the model would merely be re-recognising
the same source video rather than generalising. The notebook asserts zero group leakage.

---

## The three embedding models

| Model | Dim | Training paradigm | Sees motion? |
|---|---|---|---|
| **CLIP ViT-B/32** (image tower only) | 512 | language-supervised (image–text contrastive) | No — 16 frames mean-pooled |
| **DINOv2 ViT-S/14** | 384 | self-supervised, no labels, no text | No — 16 frames mean-pooled |
| **VideoMAE base** (Kinetics-finetuned) | 768 | self-supervised video masked autoencoding | **Yes** — spatiotemporal attention |

Three genuinely different training paradigms, not three flavours of one idea.

**The experiment:** mean-pooling is order-blind — a clip and the same clip played backwards
produce *identical* vectors. VideoMAE's tubelet embedding and spatiotemporal attention encode
frame order and cannot be fooled that way. So the question this benchmark answers is:

> **Does temporal modelling actually improve action retrieval, or is a bag of frames enough?**

The 2048/512/384/768 dimensionality spread also gives a real memory-vs-quality axis for the
indexing analysis — embedding dimension is an infrastructure decision, not only an accuracy one.

---

## Two granularities

| Index | Vectors | Enables |
|---|---|---|
| **Clip-level** | one per video | Querying with a whole clip |
| **Frame-level** | 16 per video | Querying with **a single video frame**, which the brief explicitly permits |

Frame-level vectors are free: CLIP and DINOv2 compute them anyway before pooling. VideoMAE
**cannot** produce them — it is inherently clip-level. That is a genuine architectural
limitation, reported as a finding rather than hidden.

---

## Vector database

**ChromaDB**, running **embedded in the notebook process** — no server, no Docker, no
container. It persists to `chroma_db/` and uses hnswlib underneath, so the HNSW graph is real
and its build parameters are configurable. It is to vector databases what SQLite is to
relational ones: a real engine without an server.

One collection per model, since a Chroma collection holds a single embedding space.

Index variants built: `max_neighbors` (M) ∈ {8, 16, 32}, `ef_construction` ∈ {64, 100, 256}.

**Why Chroma and not Qdrant.** Qdrant would have needed a Docker container to be meaningful —
its embedded mode does brute-force search and ignores HNSW settings entirely, so benchmarking
index mechanics against it would measure nothing. Chroma gives a genuine HNSW index with no
infrastructure at all, which suits a self-contained, reproducible submission better.

**Division of labour with FAISS.** Chroma fixes `ef_search` at collection-build time —
`collection.modify()` accepts a new value but query latency does not change, so the runtime
dial is not usable for a sweep. FAISS is therefore used for the parts Chroma cannot express:

| Tool | Role |
|---|---|
| **ChromaDB** | the vector database — ingestion, persistence, HNSW retrieval, build-parameter variants |
| **FAISS** | exact oracle, `ef_search` / `nprobe` sweeps, and index-family comparison (Flat, HNSW, IVF-Flat, IVF-PQ) |

---

## Evaluation

**"Recall" means two different things, and conflating them is the most common error here:**

| | Measures | Driven by |
|---|---|---|
| **ANN recall@k** | overlap between the index's top-k and exact brute-force top-k | `ef_search` — the **index** |
| **Retrieval Recall@K / Precision@K / mAP@K** | do returned clips share the query's true class | the **embedding model** |

An index can achieve 100% ANN recall while returning useless results — it is faithfully
retrieving bad vectors. Both are reported in separate tables.

FAISS `IndexFlatIP` over L2-normalised vectors serves as the exact-kNN oracle. Because all
vectors are unit-length, cosine similarity, dot product and Euclidean ranking are equivalent.

A **scaling study** on synthetic vectors (1k → 500k) measures index mechanics at realistic
corpus sizes; retrieval *quality* is measured on real data only. The two are kept separate and
labelled as such.

---

## Layout

```
GenAI_Midterm_VideoSearch.ipynb   <- main deliverable, run top to bottom
notebooks/                        <- same content split by rubric section
src/
  download_data.py                <- dataset acquisition (resumable)
  videomae_fix.py                 <- VideoMAE weight remap (see below)
data/                             <- videos (not committed)
embeddings/                       <- .npy vectors + manifest.csv
results/                          <- metrics CSVs and figures
```

## Running it

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python src/download_data.py          # optional: full UCF101; a subset is included
jupyter lab GenAI_Midterm_VideoSearch.ipynb
```

Then run the notebook top to bottom. Nothing else needs to be started — the vector database
runs inside the notebook process.

---

## Engineering notes (things that fail silently)

**1. VideoMAE is broken under `transformers` 5 unless patched.**
The `MCG-NJU/videomae-*` checkpoints store `attention.attention.q_bias` / `v_bias`.
transformers 5 renamed these to `attention.attention.{query,key,value}.bias`.
`from_pretrained` reports the checkpoint's tensors as UNEXPECTED, the model's as MISSING, and
**zero-initialises the query and value biases**. The model still loads, still runs, and still
emits 768-d vectors — they are simply computed with trained parameters discarded. Left
unnoticed this would have produced the false finding *"temporal modelling doesn't help."*
`src/videomae_fix.py` remaps the names and asserts the restored biases are non-zero.

**2. `CLIPModel.get_image_features()` returns an unprojected output under transformers 5.**
It yields a `BaseModelOutputWithPooling` with no `image_embeds`, so naive use lands in 768-d
vision space instead of the 512-d shared space. The notebook calls `vision_model` and
`visual_projection` explicitly.

**3. pandas 3 changed `groupby(...).apply(...)`.**
The grouping column is no longer passed to the function, so results silently lose it. Use
`groupby(...).head(n)`. pandas 3 also backs string columns with PyArrow, and those arrays
reject 2-D fancy indexing (`labels[idx]` where `idx` is a matrix) — labels are converted once
to a plain object ndarray.

**4. Dataset acquisition.**
The HuggingFace CDN rate-limits unauthenticated bulk transfers and drops the connection
mid-file *without raising*, so a Python retry loop never fires; `hf_hub_download` also opens a
**new** `.incomplete` file per attempt, restarting rather than resuming. The official CRCV
mirror measured 0.08 MB/s. The working approach is `curl -C -` with `--speed-limit`/`--speed-time`
stall detection, which resumes a single file across drops.
