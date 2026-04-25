# 🧠 Lightweight Word Embeddings in C

A minimal C project that demonstrates and benchmarks **standard float32 embeddings** vs **lightweight INT4 packed embeddings** for IoT/edge vocabulary, with both a custom matrix kernel and OpenBLAS acceleration.

---

## 📁 Project Structure

```
.
├── embedding.h       # Configuration, macros, and function declarations
├── embedding.c       # Embedding creation (float32 & INT4 packed)
├── evaluation.c      # Benchmarking: custom kernel + BLAS evaluation
├── vocabulary.c      # Vocabulary definition and word lookup
└── main.c            # Entry point
```

---

## ⚙️ Configuration

Defined in `embedding.h`:

| Macro        | Value | Description                          |
|--------------|-------|--------------------------------------|
| `VOCAB_SIZE` | 6     | Number of words in vocabulary        |
| `STD_DIM`    | 16    | Dimensions for float32 embedding     |
| `LIGHT_DIM`  | 4     | Dimensions for INT4 packed embedding |

---

## 🗂️ Vocabulary

The vocabulary consists of 6 IoT/edge-domain words:

```
sensor, alert, device, edge, power, network
```

---

## 🔬 How It Works

### Standard Embedding (float32)
- A `VOCAB_SIZE × STD_DIM` matrix of `float` values
- Lookup is done via a **matrix-vector multiply** (one-hot × embedding matrix)
- Benchmarked with a custom `matmul_kernel` and with **OpenBLAS `cblas_sgemv`**

### Lightweight Embedding (INT4 packed)
- Each word has `LIGHT_DIM` values, each stored in **4 bits**
- Two INT4 values are packed into a single `uint8_t` byte
- Memory footprint is significantly reduced compared to float32

---

## 📊 Benchmark Summary

| Method                  | Iterations | Metric         |
|-------------------------|------------|----------------|
| Custom Kernel (float32) | 100,000    | Time in ms      |
| INT4 Unpack             | 500,000    | Time in ms      |
| BLAS sgemv (float32)    | 50,000     | Time in ms      |

Memory comparison is also printed at runtime:

```
Standard:            384 bytes   (6 × 16 × 4)
Lightweight (INT4):   12 bytes   (6 × 2 × 1)
```

---

## 🛠️ Build

### Prerequisites

- GCC or Clang
- [OpenBLAS](https://www.openblas.net/) installed

### Compile

```bash
gcc -O2 -o embedding main.c embedding.c evaluation.c vocabulary.c \
    -lopenblas -lm
```

> On some systems you may need `-I/usr/include/openblas` or adjust the include path for `<openblas/cblas.h>`.

---

## 🚀 Run

```bash
./embedding
```

You'll be prompted to enter a word:

```
Enter word: sensor
```

The program will output the embedding vectors and timing results for all three methods.

---

## 📌 Notes

- INT4 packing stores two 4-bit values per byte: `high nibble = val1`, `low nibble = val2`
- The project is intentionally minimal — designed as a learning reference for embedding quantization concepts on constrained hardware
- Embedding values are synthetically generated (pattern-based), not trained

---

## 📄 License

MIT
