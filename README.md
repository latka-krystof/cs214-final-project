# Load-Adaptive Routing for Speculative Decoding LLMs


This project evaluates different routing strategies for serving the Llama 3.2 3B Instruct model using speculative and regular inference servers.

The experiments compare:
- Random Router (Baseline)
- Custom Router
- Regular-only configuration

The system uses vLLM servers and sends requests generated from a synthetic dataset.

---

# Prerequisites

## 1. Get Model Access

Request permission for the model:

https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

## 2. Authenticate with Hugging Face

Run:

```bash
hf auth login
```

Enter your Hugging Face token.

---

# 1. Finding the Concurrency Threshold

Run the benchmark script:

```bash
cd benchmark
python benchmark.py
```

Notes:
- Instructions and previous results are already included in the script.
- The previous benchmark was run on a **T4 GPU**.

---

# 2. Dataset Preparation

Navigate to the dataset folder:

```bash
cd dataset
```

Generate the dataset:

```bash
python gen_data.py --duration 1 --requests 200 --output dataset_200_1.json
```

### Parameters

| Parameter | Description |
|----------|-------------|
| duration | Duration of traffic (minutes) |
| requests | Total number of requests |
| output | Output dataset file |

Example output:

```
dataset_200_1.json
```

---

# 3. Main Experiments

The following experiments were run on an **L4 GPU**.

Each experiment requires starting the appropriate **vLLM server configuration**.

---

# Experiment 1 — Baseline (Random Router)

Configuration: **1 Speculative Server + 1 Regular Server**

### Step 1 — Start vLLM

Run the **1 speculative + 1 regular** configuration from:

```
cd start_scripts
llama.sh
```

### Step 2 — Start Router

```bash
python random_router.py
```

### Step 3 — Send Requests

```bash
python request.py \
--input dataset.json \
--model meta-llama/Llama-3.2-3B-Instruct \
--endpoint http://localhost:8000/v1/chat/completions
```

---

# Experiment 2 — Our Router

Configuration: **1 Speculative Server + 1 Regular Server**

### Step 1 — Start vLLM

Run the same configuration from:

```
llama.sh
```

### Step 2 — Start Router

```bash
python new_router.py
```

### Step 3 — Send Requests

```bash
python request.py \
--input dataset.json \
--model meta-llama/Llama-3.2-3B-Instruct \
--endpoint http://localhost:8000/v1/chat/completions
```

---

# Experiment 3 — Random Router (Regular Only)

Configuration: **1 Regular Server + 1 Regular Server**

### Step 1 — Start vLLM

Run the **2 regular servers** configuration from:

```
llama.sh
```

### Step 2 — Start Router

```bash
python random_router.py
```

### Step 3 — Send Requests

```bash
python request.py \
--input dataset.json \
--model meta-llama/Llama-3.2-3B-Instruct \
--endpoint http://localhost:8000/v1/chat/completions
```

---
