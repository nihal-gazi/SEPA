## SEPA: Sentence Embedding Predictive Architecture (formerly named 'GSCv3')

SEPA is an experimental, highly efficient neural architecture that fundamentally reimagines how machines generate text. Instead of grinding through standard token-by-token autoregression (which forces a model to simultaneously learn narrative logic, spelling, and grammar in the exact same layers), SEPA decouples semantic reasoning from syntactic rendering.

By compressing grammatical phrases into discrete semantic vectors (latents), SEPA can perform complex narrative reasoning with a fraction of the parameter count of traditional LLMs. It achieves real-time generation (1.3+ sentences per second) directly on edge hardware (e.g., 4GB RAM budget smartphones) entirely via CPU WebAssembly.

---

## 🧠 The Architecture & Mathematics

SEPA operates in two isolated phases: an Information Bottleneck (Autoencoder) and a Latent Reasoning Engine.

### Phase 1: The Semantic Compressor (VQ-Autoencoder)

The goal of Phase 1 is to learn a universal dictionary of concepts. We avoid arbitrary padding tokens by using 1D Temporal Interpolation—stretching input phrases (e.g., ["The", "dog", "ran"]) to perfectly fill a fixed sequence length $L_{seq}$ (e.g., ["The", "The", "dog", "dog", "ran", "ran"]).

* **The Observer $O(x)$:** A Transformer + Conv1D network that takes the interpolated sequence and compresses it down to a continuous latent bottleneck of length $L_{lat}$.

$$z = O(x), \quad z \in \mathbb{R}^{L_{lat} \times D_{model}}$$

* **Residual Vector Quantization (RVQ):** The continuous tensor $z$ is forced through a multi-stage discrete codebook (vocabulary size $K$, e.g., 2048). We use Exponential Moving Average (EMA) and Dead Code Revival to ensure 100% codebook utilization.

$$q, I = RVQ(z), \quad I \in \mathbb{Z}^{L_{lat} \times N_{stages}}$$

* **The Reconstructor $f(x)$:** A ConvTranspose1d + Transformer network that expands the quantized vectors back to the original text.

$$\hat{x} = f(q)$$

**Result:** The Autoencoder is frozen. We now have a rigid mathematical map where every possible English phrase is represented by a sequence of discrete integers $I$.

### Phase 2: The Reasoning Engine (AR Latent Generator)

With syntax handled by the Autoencoder, we train an Autoregressive (AR) Causal Transformer purely on the discrete indices $I$.

$$P(I_{t+1} | I_1, I_2, \dots, I_t)$$

Because it only predicts high-level concepts (e.g., [Action] -> [Consequence] -> [Emotion]), the reasoning engine can be drastically smaller than a standard LLM.

### Inference: Valid Codebook Masking & Squashing

During inference, standard RVQ architectures often suffer from "alien concepts"—predicting combinations of Stage 1 and Stage 2 codes that don't exist. SEPA solves this mathematically:

* We map the valid space during training: $V = \{(s_1, s_2) \in \text{Dataset}\}$.
* During inference, we mask the Stage 2 logits: $P(s_2 | s_1) = 0 \text{ if } (s_1, s_2) \notin V$.
* We decode the generated sequence in a single flash forward pass.
* We apply a 1D De-interpolation Squash (itertools.groupby) to collapse ["The", "The", "dog", "dog"] back to ["The", "dog"].

---

## 📂 Repository Structure.

```text  
├── SEPA/                          # Core SEPA Library
│   ├── config.py                   # Centralized hyperparameters
│   ├── count_param.py              # Calculates true unique parameter footprints
│   ├── dataset.py                  # Splits raw text into safe, contiguous chunks
|   ├── dataset_10.txt              # 10MB subset
│   ├── dataset_preprocess.py       # Phrase interpolation and Tokenizer training
│   ├── gsc_v3_tokenizer.json       # BPE Tokenizer
│   ├── models.py                   # PyTorch classes (Observer, RVQ, Reconstructor, Generator)
│   ├── run_all.py                  # Master automation script (Trains all phases -> Infers)
│   ├── train_phase1.py             # Trains Autoencoder
│   ├── extract_valid_pairs.py      # Maps the allowed latent space (Valid Codebook Masking)
│   ├── train_phase2.py             # Trains AR Latent Generator on frozen latents
│   ├── infer.py                    # Python inference engine
│   ├── onnx_converter.py           # Exports PyTorch models to ONNX Web format
|   ├── README.md                   # This file
│   └── (Checkpoints & Data)        # generator.pt, observer.pt, valid_pairs.json, etc.
│   
├── LLM/                            # Baseline Causal LLM (For performance comparison)
│   ├── config.py                   # Scaled to match SEPA parameter counts perfectly
│   ├── model.py                    # Standard Decoder-only architecture
│   ├── train_llm.py                # Standard next-token prediction trainer
│   └── infer_llm.py                # Baseline inference script
│   
└── web_server/                     # Edge Deployment (Browser/Mobile)
    ├── index.html                  # Web UI with integrated JS ONNX execution
    ├── server.py                   # Flask backend (Serves UI + Tokenization APIs)
    ├── valid_pairs.json            # Inference mask dictionary
    └── models/                     # Exported ONNX compute graphs
        ├── encoder.onnx            
        ├── generator.onnx          
        └── decoder.onnx            

```

---

## 🚀 Getting Started

### 1. Training the Model

You can execute the entire pipeline (Phase 1 Training $\to$ Valid Pair Extraction $\to$ Phase 2 Training $\to$ Inference) automatically using the master script.

```bash
cd SEPA
python run_all.py

```

To configure hyperparameters (Parameter budgets, Codebook sizes, D_MODEL), simply edit `GSCv3/config.py`.

### 2. Standalone Inference

If you have trained models (`.pt` files in the directory), you can run the inference engine directly from Python:

```bash
python infer.py

```

### 3. Edge Deployment (Web/Mobile)

SEPA's greatest strength is its ability to run complex narrative generation entirely on low-end edge hardware (like smartphone CPUs) using ONNX WebAssembly.

**Step A: Export to ONNX**
Convert the trained PyTorch weights into optimized graph computations:

```bash
python onnx_converter.py

```

**Step B: Start the Web Server**
Boot up the Flask server to host the UI:

```bash
cd ../web_server
python server.py

```

Open `http://localhost:5000` in your desktop browser, or connect via your smartphone on the same Wi-Fi network.

---

## 📊 Parameter Scaling

SEPA utilizes Selective Weight Tying.

* **Autoencoder:** Un-tied for maximum morphological flexibility.
* **Generator:** Tied (Embeddings = Logit Projections) to save millions of parameters.

**Example Edge Config (1.85M True Unique Parameters):**

* **Autoencoder:** ~730K parameters (Handles spelling/syntax).
* **Latent Generator:** ~1.1M parameters (Handles logic/narrative).

**Performance:** 1.3+ Sentences per second on a Samsung A06 via WebAssembly.