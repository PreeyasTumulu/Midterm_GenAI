# Learning Guide — everything in this project, explained

Written to be read top to bottom before the viva. Each section states the concept, why it is in
this project, and the question an examiner is likely to ask about it.

---

## Part 1 — What problem are we solving?

**Similarity search.** Given a query item, find the most similar items in a collection. Not
"find items whose *description* matches" — that would be text search. We compare the *content*
itself.

The pipeline is always the same four steps:

```
raw file  →  [embedding model]  →  vector  →  [vector index]  →  k nearest vectors
```

**Embedding / latent vector.** Take a neural network trained on images or video, cut off its
final classification layer, and read the activations underneath. That layer is a list of numbers
— 512 of them for CLIP, 384 for DINOv2, 768 for VideoMAE. The network has compressed everything
it understands about the input into that list. Two visually or semantically similar inputs land
close together in that space; dissimilar ones land far apart. "Latent" just means *hidden /
internal* — it is not a human-designed feature list, it is whatever the network learned.

**Why the brief bans metadata.** If you were allowed to use filenames or tags, the task would
collapse to a database lookup and prove nothing about representation learning. The whole point
is that the *pixels alone* must carry enough signal.

> **Likely question:** "Your files are named `v_Archery_g01_c01.avi` — how do I know you're not
> cheating?"
> **Answer:** The search function takes a vector and returns ids. The database stores ids and
> nothing else — no labels, no filenames. Labels are parsed once to build ground truth and are
> only touched by the scoring function, after results have already come back. And the CLIP text
> encoder is deleted from memory outright, so it *cannot* participate.

---

## Part 2 — The three embedding models

The brief asks to compare architectures. Comparing two similar models proves little, so these
three were chosen because each learned from a **fundamentally different training signal**.

### 2.1 CLIP ViT-B/32 — *language-supervised*

- **Trained on:** ~400M (image, caption) pairs scraped from the web.
- **How:** *Contrastive learning*. Put a batch of images and their captions through two separate
  encoders. Pull the correct image–caption pairs together in vector space and push all the wrong
  pairings apart. Neither encoder is ever told "this is a dog" — it learns from which caption
  goes with which image.
- **Architecture:** Vision Transformer. The image is cut into 32×32 pixel **patches**, each patch
  is flattened into a vector, and the transformer applies self-attention across patches so every
  patch can look at every other. A special `CLS` token accumulates a summary of the whole image.
- **Result:** the embedding is *semantic* — it encodes "what this depicts" in a
  language-influenced way, because captions shaped the space.
- **We use the image tower only.** `del clip_model.text_model`.

### 2.2 DINOv2 ViT-S/14 — *self-supervised, no labels at all*

- **Trained on:** 142M curated images, **no labels and no text whatsoever**.
- **How:** *Self-distillation*. Two copies of the network (student and teacher) see different
  crops/augmentations of the same image, and the student is trained to match the teacher's
  output. Since both views come from one image, the network must learn what stays constant under
  cropping and colour change — i.e. real visual structure.
- **Architecture:** ViT with 14×14 patches. We read the `CLS` token: `last_hidden_state[:, 0]`.
- **Result:** the embedding is *visual* rather than semantic — texture, shape, layout,
  fine-grained appearance. Often better than CLIP at telling apart things that look similar but
  would get the same caption.

### 2.3 VideoMAE base (Kinetics-finetuned) — *self-supervised video, the only one that sees motion*

- **Trained on:** video, self-supervised, then finetuned on Kinetics action recognition.
- **How:** *Masked autoencoding*. Hide ~90% of the video and train the network to reconstruct the
  missing parts. To fill in a gap you must understand how things move, not just what they look
  like.
- **Architecture:** **tubelet embedding** — instead of 2D patches, it cuts the video into small
  3D blocks spanning space *and* time (e.g. 2 frames × 16 × 16 pixels). Self-attention then runs
  across these spatiotemporal tubelets, so the model can relate what happened at frame 3 to what
  happened at frame 11.
- **Result:** the only embedding here that encodes **temporal order**.

| | CLIP | DINOv2 | VideoMAE |
|---|---|---|---|
| Supervision | language | none (self) | none (self) + Kinetics |
| Dim | 512 | 384 | 768 |
| Input | one frame | one frame | 16 frames |
| Encodes motion | ✗ | ✗ | ✓ |

---

## Part 3 — The video problem, and the actual experiment

A vector index needs **one fixed-length vector per item**. A video is a *sequence*. So every
pipeline must answer: **how do you turn T frames into 1 vector?**

- **CLIP and DINOv2** cannot see time at all. We embed 16 frames independently and take the
  **mean**.
- **VideoMAE** consumes all 16 frames at once and produces one vector natively.

**The key insight — mean-pooling is order-blind.** The mean of a set does not depend on the order
of the set. So:

> A mean-pooled embedding of a clip is **bit-identical** to the embedding of that clip played
> backwards.

Sitting down and standing up. Opening and closing. Push-ups and pull-ups. To CLIP and DINOv2
these can look nearly identical; to VideoMAE they do not. **That is the experiment this project
runs**, and it is why the modality choice (video) actually earns something.

---

## Part 4 — Vector mathematics

**L2 normalisation.** We divide every vector by its own length, making all vectors unit length.
After that:

- cosine similarity **==** dot product (the denominator is 1)
- Euclidean distance becomes a monotonic function of the dot product

So all three metrics produce the **same ranking**. Normalising once removes an entire class of
"which metric should I use?" confusion.

**Curse of dimensionality.** In high dimensions, points spread out and distances between them
become more similar to one another. Exhaustive comparison also gets expensive: 1M vectors × 512
dims is 512M multiply-adds per query. This is *why* approximate indexes exist.

---

## Part 5 — Vector databases and indexing

### The three index families

| Index | How it works | Trade-off |
|---|---|---|
| **Flat** | compare the query against every single vector | 100% exact, slowest, O(N) |
| **HNSW** | a multi-layer graph; start at a coarse layer and greedily walk downhill toward the query | fast, tunable, memory-hungry |
| **IVF** | k-means the space into `nlist` cells; only search the `nprobe` nearest cells | fast, tunable, cheap memory |
| **IVF-PQ** | IVF + product quantization | 8–32× smaller, some recall lost |

### The knobs, and which ones matter when

| Parameter | Belongs to | Set at | Effect |
|---|---|---|---|
| `M` / `max_neighbors` | HNSW | build | edges per node — higher = better recall, more RAM |
| `ef_construction` | HNSW | build | candidates while building — higher = better graph, slower build |
| `ef_search` | HNSW | **query** | **the runtime accuracy/speed dial** |
| `nlist` | IVF | build | number of k-means cells |
| `nprobe` | IVF | **query** | how many cells to actually search |

**Product Quantization (PQ)** deserves its own sentence: split a 512-d vector into 8 chunks of
64 dims, run k-means on each chunk to build a 256-entry codebook, then store just the 8 nearest
codebook indices — 8 bytes instead of 2048. That is 256× compression, at the cost of only ever
knowing each vector approximately.

### Why ChromaDB here

Chroma runs **embedded, inside the Python process** — `pip install`, no server, no Docker. It
persists to a folder and uses hnswlib underneath, so the HNSW graph is genuine and its build
parameters are real. **It is to vector databases what SQLite is to relational databases.**

> **Likely question:** "Why not Qdrant / Milvus / Pinecone?"
> **Answer:** Those need a server process (or a cloud account). Qdrant's *embedded* mode would
> have been the no-server option, but it silently degrades to **brute-force search and ignores
> HNSW settings entirely** — benchmarking index mechanics against it would measure nothing.
> Chroma gives a real HNSW index with zero infrastructure, which suits a reproducible submission.

**Where FAISS comes in.** Chroma exposes HNSW only, and fixes `ef_search` at build time — we
measured `collection.modify()` accepting new values with query latency unchanged (3.01 / 3.01 /
3.02 ms at ef 10 / 50 / 200). So FAISS handles what Chroma cannot express: the exact oracle, the
`ef_search` and `nprobe` sweeps, and the index-family comparison.

---

## Part 6 — Evaluation

### The single most important distinction in this project

**"Recall" means two different things, and conflating them is the classic error.**

| | What it measures | Driven by |
|---|---|---|
| **ANN recall@k** | overlap between the index's top-k and the *exact* brute-force top-k | `ef_search` — **the index** |
| **Retrieval Recall@K** | do the returned clips share the query's true class | **the embedding model** |

An index can score **100% ANN recall while returning complete garbage** — it is faithfully
retrieving bad vectors. The two numbers answer different questions and belong in different
tables. Saying this unprompted in the viva is worth real marks.

### The quality metrics

- **Recall@1** — is the single top result correct?
- **Recall@K** *(metric-learning convention)* — does **at least one** correct item appear in the
  top-K? Robust to how many same-class items exist in the gallery.
- **Precision@K** — what fraction of the K returned are correct?
- **mAP@K** — averages precision at every cut-off, so it rewards ranking correct items *higher*,
  not merely including them.

### Latency

Report **percentiles, not averages**. p50 is the typical experience; **p95 and p99 are what users
actually complain about**. An average hides a long tail. **QPS** = queries per second = throughput.

### The evaluation trap we avoided: group leakage

UCF101 filenames are `v_<Class>_g<group>_c<clip>.avi`. Every clip sharing a **group** was cut
from the *same source video* — same actor, same room, same lighting. They are near-duplicates.

If a random split puts group-mates in both the query set and the gallery, every score is
inflated: the model is not generalising, it is **re-recognising the same video**. Our split holds
out **whole groups**, and the notebook asserts zero overlap.

> **Likely question:** "How do you know your numbers aren't inflated?"
> **Answer:** group-disjoint splitting, enforced by an assertion, not by convention.

---

## Part 7 — Visualising the embedding spaces

| | PCA | t-SNE |
|---|---|---|
| Type | linear | non-linear |
| Deterministic | yes | no (seed-dependent) |
| Preserves | global variance | local neighbourhoods |
| Gives a number | **yes** — explained variance | no |

**Critical caveat:** in a t-SNE plot, **distances between clusters and the sizes of clusters are
not meaningful.** It is a neighbourhood-preserving cartoon. Read it only as "do same-class points
sit together." Anyone who says "these two clusters are far apart so the classes are very
different" has misread the plot.

Because of that, every visual claim here is backed by a number:

- **Silhouette score** (−1 to 1) — how tight classes are relative to how far apart they sit,
  computed in the **full** embedding space, not the 2D picture.
- **k-NN label consistency** — of each clip's 10 nearest neighbours, what fraction share its
  class. This is the number that actually predicts retrieval performance.

---

## Part 8 — What the bugs taught (all four failed *silently*)

These are worth knowing because none of them raised an error. Everything ran; the numbers were
just quietly wrong.

**1. VideoMAE lost its trained attention biases.**
The checkpoint stores `attention.attention.q_bias` / `v_bias`. transformers 5 renamed these to
`query.bias` / `key.bias` / `value.bias`. `from_pretrained` reported the checkpoint's tensors as
UNEXPECTED, the model's as MISSING, and **zero-initialised them**. The model loaded, ran, and
emitted 768-d vectors computed with trained parameters discarded.
**Why it matters:** this would have produced the confident, wrong conclusion *"temporal modelling
doesn't help."* A degraded model is far more dangerous than a crashed one.
*Lesson: read the load report; treat MISSING keys as an error, not noise.*

**2. `CLIPModel.get_image_features()` returned an unprojected output.**
Under transformers 5 it yields a `BaseModelOutputWithPooling` with no `image_embeds`, so naive
use lands in 768-d vision space instead of the 512-d shared space.
*Lesson: check the shape and type of what you get back, don't assume the old API.*

**3. `groupby(...).apply(...)` dropped the grouping column** (pandas 3 no longer passes it in).
*Lesson: major version bumps change defaults, not just APIs.*

**4. PyArrow-backed string columns rejected 2-D fancy indexing** (`labels[idx]` where `idx` is a
matrix). *Lesson: `.to_numpy()` / `.tolist()` at the boundary between pandas and numpy.*

---

## Part 9 — Rapid-fire viva prep

**Q: Why video and not images?**
Because it lets you ask a real question — does temporal modelling beat frame-averaging? — instead
of comparing two image models that mostly agree.

**Q: Why 16 frames?**
It is VideoMAE's native input length. Reusing it for CLIP and DINOv2 keeps the comparison fair:
all three models see exactly the same pixels.

**Q: Why is your Pareto curve nearly flat on real data?**
Because the corpus is small — brute force is already sub-millisecond, so HNSW has no room to win.
That is *itself* the finding, and it is why the scaling study on synthetic vectors is included:
index mechanics need realistic corpus sizes to be visible. Quality is measured on real data;
speed is measured at scale. The two are kept separate and labelled.

**Q: Could you query with a single frame?**
Yes for CLIP and DINOv2 — their per-frame vectors live in the same space as the pooled clip
vectors. **No for VideoMAE**, which is clip-level by construction. That is a genuine architectural
limitation, and it is reported rather than hidden.

**Q: What would you do with more time?**
Attention-weighted pooling instead of mean pooling; a larger backbone (DINOv2-L, VideoMAE-L);
all 101 classes; and a learned metric (triplet or ArcFace) fine-tuned on the retrieval objective.

**Q: Cosine or Euclidean?**
Irrelevant here — vectors are L2-normalised, so both give identical rankings. Cosine is stated
because it is the convention for deep embeddings.
