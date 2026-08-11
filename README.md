# Video Similarity Search — Comparing Three Embedding Architectures

**STAI345 Generative AI — Mid-Term** · Vijaybhoomi School of Science & Technology

---

## 1. The problem

**Give the system a video. It returns the most similar videos in the database — judged only by
what the video *looks and moves like*, never by its filename, tags, or any text.**

This is the problem behind reverse image search, content-based copyright matching, "more like
this" recommendations, and duplicate detection in media archives. In all of those, the item you
search with is *new* — nobody has labelled it yet — so text and metadata are unavailable by
definition. The only thing you have is the raw content.

### Why this is hard

A computer cannot compare two videos pixel by pixel. Two clips of the same action can differ in
lighting, camera angle, clothing, and speed, while being byte-wise almost entirely different.
So we need a representation that captures *meaning* rather than appearance-as-bytes.

That representation is an **embedding**: a fixed-length list of numbers produced by a neural
network, arranged so that similar content lands close together and dissimilar content lands far
apart. Once every video is a point in that space, "find similar videos" becomes "find nearby
points" — a geometry problem instead of a perception problem.

### The question this project actually answers

Different models produce *different* embedding spaces, and the space determines what "similar"
means. So the real question is:

> **Does a model that understands motion retrieve videos better than models that only understand
> still images?**

We test this by building the same search system three times, with three models trained in
fundamentally different ways, and measuring the difference.

### The rule we work under

> *"You are strictly prohibited from using textual data or metadata for the similarity search.
> The search must be purely based on the latent vector representation of the content."*

Labels are still needed to *score* results — recall is undefined without knowing what was
correct. So the discipline is about **where** labels are allowed:

| Location | Allowed? | Why |
|---|---|---|
| Inside the search | ❌ Never | This is the rule that would fail the submission |
| Scoring, after results return | ✅ Required | Otherwise no metric can be computed |

This is enforced structurally, not by good intentions:

- `search(model, vector, k)` accepts **a vector** and returns **id numbers and distances**. Nothing else crosses that boundary.
- Records in the database carry **an id and nothing else** — no label, no filename, no class. The database cannot leak what it was never given.
- CLIP's text encoder is **deleted from memory** (`del clip_model.text_model`). It cannot influence a search if it does not exist.

---

## 2. Modality and dataset

**Video**, using **UCF101** — short clips of people performing everyday actions.

| | |
|---|---|
| Source | 13,320 clips, 101 action classes |
| Used here | **1,600 clips across 40 classes** (40 per class, balanced) |
| Split | **1,194 gallery** (searchable) vs **406 queries** |
| Frames per clip | 16, sampled evenly across the whole clip |

Video was chosen deliberately over images: it is the only modality where "does temporal
modelling matter?" is a question worth asking, which turns a routine model comparison into an
actual experiment.

### The evaluation trap we had to avoid

UCF101 filenames look like `v_Archery_g03_c02.avi`. The `g03` is a **group**, and every clip in
a group was cut from the *same source video* — same person, same room, same clothes. Group-mates
are near-duplicates.

A random train/test split puts group-mates on both sides. The model then scores near-perfectly
for recognising **the same video twice**, which measures nothing. Our split holds out **whole
groups**, and the notebook asserts zero overlap before proceeding:

```python
shared = set(zip(queries.label, queries.group)) & set(zip(gallery.label, gallery.group))
assert not shared, f"group leakage found: {shared}"
```

---

## 3. The three models, and why these three

Each was trained on a **completely different learning signal**. Comparing three variants of the
same idea would prove nothing; comparing three paradigms tells you what actually matters.

| Model | Dim | How it learned | Sees motion? |
|---|---|---|---|
| **CLIP ViT-B/32** (image tower only) | 512 | Matched 400M images to their captions — *language supervision* | No — 16 frames averaged |
| **DINOv2 ViT-S/14** | 384 | Looked at 142M images with **no labels and no text** — *self-supervision* | No — 16 frames averaged |
| **VideoMAE base** (Kinetics) | 768 | Reconstructed hidden parts of **videos** — *self-supervision over time* | **Yes** — reads all 16 frames jointly |

### The hypothesis

CLIP and DINOv2 see one frame at a time, so their 16 frame-vectors are **averaged** into one.
An average has no sense of order:

> A mean-pooled embedding of a video is **identical** to the embedding of that same video played
> **backwards**.

Sitting down vs standing up. Push-ups vs pull-ups. VideoMAE uses *tubelet embedding* — 3D blocks
spanning space **and** time — so frame order is encoded and it cannot be fooled that way.

**Prediction:** VideoMAE should win, and should win *most* on actions whose appearance is
ambiguous but whose motion is not.

---

## 4. Results

### Does the model find the right videos?

*406 held-out queries against a 1,194-video gallery. Random guessing = 0.025.*

| Model | Recall@1 | Recall@10 | Precision@10 | **mAP@10** |
|---|---|---|---|---|
| **VideoMAE base** | **0.946** | **0.980** | **0.904** | **0.943** |
| DINOv2 ViT-S/14 | 0.818 | 0.968 | 0.763 | 0.840 |
| CLIP ViT-B/32 | 0.813 | 0.934 | 0.715 | 0.812 |

**The hypothesis holds.** VideoMAE leads by **13 points of Recall@1** and **10 points of mAP@10**
over the best frame-averaging model. On a 40-way problem where chance is 2.5%, it identifies the
correct action first-try **95% of the time**.

Corroborating evidence from the embedding space itself, before any database is involved:

| Model | Silhouette | Neighbour accuracy (k=10) |
|---|---|---|
| VideoMAE | **0.410** | **0.937** |
| DINOv2 | 0.254 | 0.862 |
| CLIP | 0.247 | 0.835 |

### Does the database keep up?

| Model | Index recall@10 | Typical | p95 | p99 | Queries/sec |
|---|---|---|---|---|---|
| CLIP | 1.000 | 1.77 ms | 2.08 ms | 2.26 ms | 496 |
| DINOv2 | 0.999 | 1.70 ms | 2.04 ms | 2.19 ms | 525 |
| VideoMAE | 0.997 | 1.86 ms | 2.23 ms | 2.40 ms | 529 |

Note that **CLIP has perfect index recall and the worst search results.** That is the single
most important distinction in this project — see §6.

### Why approximate indexes exist

Measured on clustered synthetic vectors at realistic scale:

| Database size | Brute force | HNSW (effort 256) | HNSW recall |
|---|---|---|---|
| 1,000 | 0.015 ms | 0.022 ms | 1.000 |
| 10,000 | 0.157 ms | 0.053 ms | 1.000 |
| 100,000 | 3.31 ms | 0.056 ms | 0.9995 |
| 500,000 | **25.06 ms** | **0.17 ms** | **0.983** |

Brute force grows **1,670×**; HNSW grows **8×** and still agrees with the exact answer 98% of the
time. At 500k vectors HNSW is **147× faster** for a 1.7% accuracy cost.

### What it cost to build

| Stage | Seconds (1,600 clips) |
|---|---|
| Reading and decoding video | 108 |
| CLIP embeddings | 152 |
| DINOv2 embeddings | 253 |
| VideoMAE embeddings | 427 |

A detail worth noticing: **DINOv2-small is slower than CLIP despite having 7× fewer parameters**
(21M vs 151M). CLIP ViT-B/**32** splits an image into 49 patches; DINOv2 ViT-S/**14** splits it
into 256. Transformer cost scales with token count, not parameter count.

---

## 5. The vector database

**ChromaDB, running embedded inside the notebook.** No server, no Docker, no container — it is to
vector databases what SQLite is to relational databases: a real engine without the infrastructure.

It builds an **HNSW index** — a graph linking each vector to its neighbours. A search hops
through the graph toward the query instead of comparing against everything.

| Parameter | Our name | Standard name | Effect |
|---|---|---|---|
| Links per vector | `neighbours` | `M` | More = better recall, more memory |
| Care while building | `build_effort` | `ef_construction` | More = better graph, slower build |
| Care while searching | `search_effort` | `ef_search` | **The runtime speed/accuracy dial** |

**Why not Qdrant or Milvus?** Both need a server process. Qdrant's embedded mode would have been
the no-server option, but it silently falls back to **brute-force search and ignores HNSW
settings entirely** — benchmarking index mechanics against it would have measured nothing.

**Why FAISS as well?** Chroma exposes HNSW only, and fixes `ef_search` at build time
(`collection.modify()` accepts a new value but latency does not change — verified). FAISS
supplies what Chroma cannot: the exact brute-force oracle, the `ef_search` / `nprobe` sweeps, and
the index-family comparison (Flat, HNSW, IVF-Flat, IVF-PQ).

---

## 6. "Recall" means two different things

Conflating these is the classic error in this kind of project.

| | What it measures | Driven by |
|---|---|---|
| **Index recall@k** | Did the fast index return the same items exact search would? | `ef_search` — the **database** |
| **Search quality** (Recall@K, mAP) | Are the returned videos actually the same action? | the **embedding model** |

Our own numbers demonstrate why this matters: **CLIP scored index recall 1.000 — a perfect
database — while having the worst search quality of the three.** The index did its job flawlessly
and faithfully retrieved poor vectors. The two numbers answer different questions and are
reported in separate tables throughout.

A related caution on the plots: in the separability table, VideoMAE keeps the **least** variance
in 2D PCA (0.076 vs CLIP's 0.178) yet separates classes **best**. That is not a contradiction —
PCA's explained variance measures how **compressible** a space is, not how **good** it is.

---

## 7. Running it

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab GenAI_Midterm_VideoSearch.ipynb
```

Then **Run All**. Nothing else needs starting — the database runs inside the notebook.

- The dataset downloads and extracts automatically (~6.5 GB) and **skips if already present**.
- Embeddings are **cached**; a second run reloads them in seconds instead of re-computing for 15 minutes. Delete `embeddings/` to force a rebuild.

### Files

| File | Purpose |
|---|---|
| `GenAI_Midterm_VideoSearch.ipynb` | **Everything** — data, models, database, benchmarks, demo |
| `LEARNING_GUIDE.md` | Concept-by-concept explanation + jargon translation table + viva Q&A |
| `README.md` | This document |
| `requirements.txt` | Dependencies |
| `results/` | Metric CSVs and 12 figures |
| `embeddings/` | Cached vectors and the clip catalogue |

---

## 8. The demo (§11 of the notebook)

Three things to show, in order:

1. **`show_results(video)`** — retrieves with all three models side by side, thumbnails bordered green for correct and red for wrong.
2. **Single-frame search** — proves CLIP and DINOv2 can search from one still image, and that **VideoMAE cannot**, because it needs all 16 frames. A genuine architectural limitation, reported rather than hidden.
3. **Upload and predict** — choose any `.avi`/`.mp4`, and the system predicts the action by a **k-nearest-neighbour vote** among the retrieved clips, with confidence bars, thumbnail grids, and vote charts.

On the prediction step: the *search* is still pixels-only. Labels are attached to the results
**afterwards**, purely to name the answer. Using them to *find* the videos would break the rule.

The upload buttons need a live kernel, so they render blank in a saved copy — the cell runs one
example automatically so the notebook always shows real output.

---

## 9. Engineering notes — four bugs that failed silently

None of these raised an error. Everything ran; the numbers were quietly wrong.

**1. VideoMAE lost its trained attention biases.**
The checkpoint stores `q_bias` / `v_bias`; transformers 5 expects `query.bias` / `key.bias` /
`value.bias`. The names do not match, so `from_pretrained` reported the checkpoint's tensors as
UNEXPECTED, the model's as MISSING, and **zero-initialised them**. The model loaded, ran, and
emitted 768-d vectors computed with trained parameters discarded. Unfixed, this would have
produced the confident and completely wrong conclusion *"temporal modelling doesn't help."*
The notebook renames them and asserts the restored biases are non-zero.

**2. `CLIPModel.get_image_features()` returns an unprojected output** under transformers 5 — a
`BaseModelOutputWithPooling` with no `image_embeds` — which would silently place us in 768-d
vision space instead of the 512-d shared space. We call the vision tower and projection directly.

**3. pandas 3 changed `groupby(...).apply(...)`** — the grouping column is no longer passed in, so
results silently lose it. Use `groupby(...).head(n)`. pandas 3 also backs text with PyArrow, whose
arrays reject 2-D indexing (`labels[matrix]`), so labels are converted once to a plain array.

**4. A misleading benchmark of our own making.** The scaling study originally used purely random
vectors. In 512 dimensions random points are all roughly equidistant, so there are no real
neighbours to find and ANN recall collapsed to 0.15 — making HNSW look broken for reasons that had
nothing to do with HNSW. Real embeddings are clustered, so the synthetic data now is too, and
recall reads 0.98 where it should.

**Dataset acquisition** was its own problem: the HuggingFace CDN rate-limits unauthenticated bulk
downloads and drops the connection **without raising**, so Python retry loops never fire, and
`hf_hub_download` opens a *new* partial file per attempt rather than resuming. The official CRCV
mirror measured 0.08 MB/s. The working approach is `curl -C -` with stall detection, which is what
the notebook uses.
