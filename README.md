# Agentic RAG vs Full‑Context Comparison Framework

> **A lightweight, extensible Python framework for evaluating Retrieval‑Augmented Generation (RAG) against full‑context prompting.**
>  
> Repo: `qnixsynapse/Agentic-RAG-Full-context-test-script`

---

## 📖 Table of Contents

- [What it does](#what-it-does)
- [Why it matters](#why-it-matters)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Running the Test Suite](#running-the-test-suite)
- [Customizing the Tests](#customizing-the-tests)
- [Extending the Framework](#extending-the-framework)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contributing](#contributing)

---

## 🔍 What it does

`Agentic-RAG-Full-context-test-script` is a **self‑contained test harness** that:

1. Loads a single PDF or TXT document.
2. Builds an in‑memory vector store (chunks + embeddings).
3. Generates a set of diverse test questions automatically.
4. Runs **two inference pipelines** side‑by‑side:
   - **Agentic RAG** – uses LLM with a small set of tools (`list_attachments`, `retrieve`, `get_chunks`).
   - **Full‑Context** – feeds the entire document into the prompt.
5. Judges each response using a separate “judge” model.
6. Generates quantitative metrics (token usage, time, judge scores) and visual plots.

> **Ideal for research, demo, or benchmarking any RAG implementation.**

---

## 📚 Why it matters

- **Fair comparison** – both methods use the same test questions and judge model.
- **No external vector databases** – everything is in‑memory, making the setup lightweight.
- **Extensible tools** – easily plug in new retrieval strategies or embeddings.
- **Comprehensive reporting** – auto‑generate bar charts, token‑usage plots, and a summary table.

---

## ✨ Features

| Feature | RAG | Full Context |
|---------|-----|--------------|
| Prompt length | ≤ few hundred tokens (via retrieval) | Entire document (often > 10k tokens) |
| Token savings | ~50% on average (depends on doc size) | Full token budget |
| Tool usage | Customizable set of tools | None |
| Evaluation | Judge model (Gemini 2.5 Pro) | Same judge |
| Visualization | Tool‑usage heatmap, comparison bar chart, token bar chart | Same |
| Logging | Detailed per‑question logs | Same |

---

## ⚙️ Prerequisites

| Item | Minimum Version |
|------|-----------------|
| Python | 3.10+ |
| pip | 24+ |
| `openai` SDK | 1.15+ |
| `fitz` (PyMuPDF) | 1.24+ |
| `tiktoken` | 0.6+ |
| `matplotlib` | 3.8+ |
| `numpy` | 2.0+ |
| `requests` | 2.32+ |

> The script runs on any machine that can reach your LLM endpoints. It assumes local or proxied OpenAI endpoints for embeddings and generation.

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/<your-org>/qnixsynapse.git
cd qnixsynapse

# Optional: create a virtual env
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **`requirements.txt`** contains the exact pinned packages used in the repo.

If you don't have a `requirements.txt`, you can create one manually:

```text
openai==1.15.0
PyMuPDF==1.24.6
tiktoken==0.6.0
matplotlib==3.8.0
numpy==2.0.0
requests==2.32.0
```

---

## ⚙️ Configuration

### Environment variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENAI_API_KEY` | Key for the test model (default local server) | `<your-key>` |
| `JUDGE_API_KEY` | Key for the judge model (Gemini) | `<google-api-key>` |
| `EMBEDDINGS_ENDPOINT` | URL for embeddings provider | `http://127.0.0.1:8080/v1/embeddings` |
| `EMBEDDINGS_API_KEY` | API key for embeddings provider | `1234` |

> The script defaults to `http://127.0.0.1:8080/v1/embeddings` but you can override it.

### `doc_paths` list

At the bottom of the script, set `DOCS_FOLDER` to the directory containing your PDF or TXT files:

```python
DOCS_FOLDER = "/path/to/your/docs"
```

The script will automatically discover files in that folder and run tests for each.

---

## 🚀 Usage

1. **Prepare the document** – place your PDF/TXT files in the folder referenced by `DOCS_FOLDER`.
2. **Run the script**:

   ```bash
   python Agentic-RAG-Full-context-test-script.py
   ```

3. **Results** – after completion you’ll find:
   - `rag_comparison_results.json` (raw metrics)
   - `model_performance_comparison.png`
   - `tool_usage_chart.png`
   - `token_usage.png`

The console output will also show a quick summary table and a final "Test complete!" message.

---

## 📦 Running the Test Suite

The main entry point (`__main__`) does the following:

1. **List Documents** – pulls all PDF/TXT files from `DOCS_FOLDER`.
2. **Load & Chunk** – each file is chunked and embedded in batches.
3. **Generate Test Questions** – 4 random questions + a summary.
4. **Test RAG** – runs the agentic approach per question.
5. **Test Full Context** – runs the full‑context approach per question.
6. **Judge** – evaluates both responses.
7. **Save & Visualize** – dumps JSON and creates PNGs.

You can change the number of questions by editing `generate_test_questions(num_questions=4)`.

---

## 🛠️ Customizing the Tests

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `chunk_size` | 256 | Max tokens per chunk. |
| `overlap` | 64 | Overlap between chunks. |
| `similarity_cutoff` | 0.3 | Minimum cosine similarity to return a chunk. |
| `top_k` | 3 | Number of chunks returned per `retrieve` call. |
| `test_model` | `GPT-OSS-20B-MXFP4-F16` | Model used for answering. |
| `judge_model` | `models/gemini-2.5-pro` | Judge model used for scoring. |
| `max_iterations` | 10 | Max number of tool‑call loops. |

Adjust these values in the `SimpleVectorStore` and `RAGTester` constructors to match your use case.

---

## 🔌 Extending the Framework

1. **Add a new tool**  
   - Define a new entry in `self.rag_tools`.
   - Implement a handler in `handle_tool_call`.
   - The tool will be available for the LLM to call automatically.

2. **Switch to a different embedding provider**  
   - Update `self.vector_store.get_embeddings` and `add_document` to call your API.

3. **Use a different judge**  
   - Replace `judge_client` with your own endpoint.
   - Ensure the evaluation prompt stays the same.

4. **Persist vector store**  
   - Add serialization (`pickle` or JSON) in `SimpleVectorStore` to save and load chunks/embeddings.

---

## 🔧 Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `requests.exceptions.ConnectionError` | Endpoint unreachable | Verify URL, port, firewall. |
| `openai.APIError` | Wrong API key | Check `OPENAI_API_KEY`. |
| `IndexError` during retrieval | Chunk list mismatch | Ensure embeddings were added before retrieval. |
| `Token limit exceeded` | Document too long for model | Increase `chunk_size` or use a higher‑limit model. |
| `Invalid JSON` from judge | Prompt formatting issue | Inspect `eval_prompt`, ensure JSON object. |

---

## 📄 License

MIT © 2025 qnixsynapse

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss major changes. For small fixes, just create a PR.

---

Happy testing! 🚀
