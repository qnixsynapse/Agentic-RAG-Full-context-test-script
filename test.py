import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz
import matplotlib.pyplot as plt
import numpy as np
import openai
import requests
import tiktoken

# Instantiate separate clients for different endpoints
test_client = openai.OpenAI(base_url="http://10.200.108.57:8080/v1", api_key="<")
# Gemini 2.5 Pro as judge model
judge_client = openai.OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/v1",
    api_key="<insert google api key here>",
)


class SimpleVectorStore:
    """Simple in-memory vector store for document chunks"""

    def __init__(self, chunk_size=256, overlap=64, similarity_cutoff=0.3):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.similarity_cutoff = similarity_cutoff
        self.chunks = []
        self.embeddings = []
        self.metadata = []
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def chunk_text(self, text: str, file_id: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        tokens = self.encoding.encode(text)
        chunks = []

        start = 0
        order = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            chunks.append(
                {
                    "text": chunk_text,
                    "file_id": file_id,
                    "order": order,
                    "start_token": start,
                    "end_token": end,
                }
            )

            start += self.chunk_size - self.overlap
            order += 1

        return chunks

    def add_document(self, text: str, file_id: str, file_name: str):
        """Add a document by chunking and embedding in batches."""
        chunks_with_metadata = self.chunk_text(text, file_id)
        if not chunks_with_metadata:
            return

        texts_to_embed = [chunk["text"] for chunk in chunks_with_metadata]
        batch_size = 256
        print(f"Embedding {len(texts_to_embed)} chunks in batches of {batch_size}...")

        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i : i + batch_size]
            payload = {
                "model": "sentence-transformer-mini",
                "input": batch_texts,
                "encoding_format": "float",
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer 1234",
            }

            try:
                response = requests.post(
                    "http://127.0.0.1:8080/v1/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=600,
                )
                response.raise_for_status()
                response_json = response.json()

                batch_embeddings = [item["embedding"] for item in response_json["data"]]
                self.embeddings.extend(batch_embeddings)

                batch_metadata = [
                    {
                        "file_id": c["file_id"],
                        "file_name": file_name,
                        "order": c["order"],
                        "start_token": c["start_token"],
                        "end_token": c["end_token"],
                    }
                    for c in chunks_with_metadata[i : i + batch_size]
                ]
                self.metadata.extend(batch_metadata)
                self.chunks.extend(batch_texts)
                print(
                    f"  - Processed batch {i//batch_size + 1}/{(len(texts_to_embed) - 1)//batch_size + 1}"
                )
            except requests.exceptions.RequestException as e:
                print(f"Error during embedding: {e}")
                exit()

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a_np = np.array(a)
        b_np = np.array(b)
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

    def get_embeddings(self, query: str) -> List[float]:
        """Get embedding for a single query"""
        payload = {
            "model": "sentence-transformer-mini",
            "input": query,
            "encoding_format": "float",
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer 1234",
        }
        response = requests.post(
            "http://127.0.0.1:8080/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        response_json = response.json()
        return response_json["data"][0]["embedding"]

    def retrieve(
        self, query: str, top_k: int = 3, file_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query"""
        query_embedding = self.get_embeddings(query)

        results = []
        for i, embedding in enumerate(self.embeddings):
            if file_ids and self.metadata[i]["file_id"] not in file_ids:
                continue

            similarity = self.cosine_similarity(query_embedding, embedding)

            if similarity >= self.similarity_cutoff:
                results.append(
                    {
                        "text": self.chunks[i],
                        "similarity": similarity,
                        "metadata": self.metadata[i],
                    }
                )

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def list_files(self) -> List[Dict[str, str]]:
        """List all files in the store"""
        files = {}
        for meta in self.metadata:
            if meta["file_id"] not in files:
                files[meta["file_id"]] = {
                    "file_id": meta["file_id"],
                    "file_name": meta["file_name"],
                }
        return list(files.values())

    def get_chunks_by_range(
        self, file_id: str, start_order: int, end_order: int
    ) -> List[Dict[str, Any]]:
        """Get chunks by order range"""
        results = []
        for i, meta in enumerate(self.metadata):
            if meta["file_id"] == file_id and start_order <= meta["order"] <= end_order:
                results.append({"text": self.chunks[i], "metadata": meta})

        results.sort(key=lambda x: x["metadata"]["order"])
        return results


@dataclass
class TestMetrics:
    """Metrics for a single test case"""

    case_name: str
    question: str
    response: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    time_taken: float
    num_tool_calls: int
    tool_calls_detail: List[Dict[str, Any]]
    judge_score: Optional[float] = None
    judge_rationale: Optional[str] = None


class RAGTester:
    """Test harness for comparing RAG vs Full Context"""

    def __init__(
        self,
        test_model: str = "GPT-OSS-20B-MXFP4-F16",
        judge_model: str = "models/gemini-2.5-pro",
    ):
        self.test_model = test_model
        self.judge_model = judge_model
        self.vector_store = SimpleVectorStore()
        self.full_context = ""
        self.file_id = "doc_001"
        self.file_name = "test_document.txt"
        self.judge_history = []  # Conversation history with document context

        self.rag_tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_attachments",
                    "description": "List files attached to the current thread.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "retrieve",
                    "description": "Retrieve relevant snippets from documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Query to search for",
                            },
                            "top_k": {"type": "number", "default": 3},
                            "file_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_chunks",
                    "description": "Retrieve chunks from a file by their order range. For a single chunk, use start_order = end_order. Thread context is inferred automatically; you may optionally provide scope='thread'. Use sparingly; intended for advanced usage. Prefer using retrieve instead for relevance-based fetching.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_id": {
                                "type": "string",
                                "description": "File ID from list_attachments",
                            },
                            "start_order": {
                                "type": "number",
                                "description": "Start of chunk range (inclusive, 0-indexed)",
                            },
                            "end_order": {
                                "type": "number",
                                "description": "End of chunk range (inclusive, 0-indexed). For single chunk, use start_order = end_order.",
                            },
                            "scope": {
                                "type": "string",
                                "enum": ["thread"],
                                "description": "Retrieval scope; currently only thread is supported",
                                "default": "thread",
                            },
                        },
                        "required": ["file_id", "start_order", "end_order"],
                    },
                },
            },
        ]

    def load_document(self, document_path: str):
        """Load document (handles .txt and .pdf) and prepare for RAG."""
        self.file_name = Path(document_path).name
        print(f"Loading document: {self.file_name}")

        text_content = ""
        if document_path.lower().endswith(".pdf"):
            try:
                with fitz.open(document_path) as doc:
                    text_content = "".join(page.get_text() for page in doc)
            except Exception as e:
                print(f"Error reading PDF file '{document_path}': {e}")
                raise
        elif document_path.lower().endswith(".txt"):
            try:
                with open(document_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
            except Exception as e:
                print(f"Error reading text file '{document_path}': {e}")
                raise
        else:
            raise ValueError(
                f"Unsupported file type: '{document_path}'. Please use .txt or .pdf."
            )

        self.full_context = text_content
        self.vector_store.add_document(self.full_context, self.file_id, self.file_name)
        print(
            f"Document loaded: {len(self.full_context)} characters, {len(self.vector_store.chunks)} chunks."
        )

    def initialize_judge_with_document(self):
        """
        Check if judge has been initialized with document context.
        This is now handled by generate_test_questions(), so this is just a verification.
        """
        if not self.judge_history:
            raise RuntimeError(
                "Judge history not initialized. Call generate_test_questions() first."
            )
        print("Judge model already initialized with document context.")

    def generate_test_questions(self, num_questions: int = 4) -> List[str]:
        """Generate test questions from the document and initialize judge history."""
        prompt = f"""Based on the following document, generate {num_questions} diverse test questions that cover different aspects of the content.

**DOCUMENT:**
---
{self.full_context}
---

Return ONLY a JSON array of questions: ["question 1", "question 2", ...]"""

        # Initialize judge history with system message
        self.judge_history = [
            {
                "role": "system",
                "content": "You are an impartial judge who generates test questions and evaluates AI responses based ONLY on a provided document. Respond only with the requested format.",
            }
        ]

        # Add question generation request
        self.judge_history.append({"role": "user", "content": prompt})

        response = judge_client.chat.completions.create(
            model=self.judge_model,
            messages=self.judge_history,
        )

        questions_json = response.choices[0].message.content.strip()

        # Add assistant's response to history
        self.judge_history.append({"role": "assistant", "content": questions_json})

        if questions_json.startswith("```"):
            questions_json = questions_json.split("```")[1].lstrip("json\n")

        questions = json.loads(questions_json)
        return questions

    def judge_response(self, question: str, response: str) -> Dict[str, Any]:
        """
        Evaluate a response using the judge model.
        The judge maintains conversation history with the document context.
        """
        eval_prompt = f"""Evaluate this response:

**QUESTION:** "{question}"

**AI'S RESPONSE:** "{response}"

Provide your evaluation as JSON with:
- "relevance_score": Score from 1-5 (how relevant to the question)
- "accuracy_score": Score from 1-5 (how accurate based on the document)
- "overall_score": Average of relevance and accuracy scores
- "rationale": Brief explanation for your scores"""

        # Add evaluation request to history
        self.judge_history.append({"role": "user", "content": eval_prompt})

        try:
            judge_response_obj = judge_client.chat.completions.create(
                model=self.judge_model,
                messages=self.judge_history,
                response_format={"type": "json_object"},
            )

            judge_content = judge_response_obj.choices[0].message.content

            # Add judge's response to history
            self.judge_history.append({"role": "assistant", "content": judge_content})

            judgement = json.loads(judge_content)
            return judgement
        except Exception as e:
            print(f"Error during judging: {e}")
            return {"overall_score": 0, "rationale": "Error parsing judge response."}

    def test_with_method(self, question: str, method: str) -> TestMetrics:
        """Generic method to test either RAG or Full Context approach."""
        start_time = time.time()

        if method == "rag":
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use tools to answer questions about documents.",
                },
                {"role": "user", "content": question},
            ]

            tool_calls_detail = []
            total_input_tokens = 0
            total_output_tokens = 0

            max_iterations = 10
            for _ in range(max_iterations):
                response = test_client.chat.completions.create(
                    model=self.test_model,
                    messages=messages,
                    tools=self.rag_tools,
                    temperature=0,
                    seed=42,
                )

                total_input_tokens += response.usage.prompt_tokens
                total_output_tokens += response.usage.completion_tokens
                message = response.choices[0].message

                if message.tool_calls:
                    messages.append(message)
                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        tool_calls_detail.append(
                            {"name": function_name, "arguments": function_args}
                        )
                        result = self.handle_tool_call(function_name, function_args)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result,
                            }
                        )
                else:
                    break

            end_time = time.time()
            return TestMetrics(
                "Agentic RAG",
                question,
                message.content,
                total_input_tokens,
                total_output_tokens,
                total_input_tokens + total_output_tokens,
                end_time - start_time,
                len(tool_calls_detail),
                tool_calls_detail,
            )
        else:  # full context
            messages = [
                {
                    "role": "system",
                    "content": "Answer questions based on the provided document.",
                },
                {
                    "role": "user",
                    "content": f"Document:\n\n{self.full_context}\n\nQuestion: {question}",
                },
            ]

            response = test_client.chat.completions.create(
                model=self.test_model, messages=messages, temperature=0, seed=42
            )

            end_time = time.time()
            return TestMetrics(
                "Full Context",
                question,
                response.choices[0].message.content,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens,
                end_time - start_time,
                0,
                [],
            )

    def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Handle tool calls for RAG approach."""
        if tool_name == "list_attachments":
            return json.dumps(self.vector_store.list_files())
        elif tool_name == "retrieve":
            results = self.vector_store.retrieve(
                arguments.get("query"),
                arguments.get("top_k", 3),
                arguments.get("file_ids"),
            )
            return json.dumps(
                [
                    {
                        "text": r["text"],
                        "similarity": r["similarity"],
                        "file_id": r["metadata"]["file_id"],
                        "order": r["metadata"]["order"],
                    }
                    for r in results
                ]
            )
        elif tool_name == "get_chunks":
            file_id = arguments.get("file_id")
            start_order = arguments.get("start_order")
            end_order = arguments.get("end_order")

            if file_id is None or start_order is None or end_order is None:
                return json.dumps(
                    {
                        "error": "Missing required parameters: file_id, start_order, end_order"
                    }
                )

            results = self.vector_store.get_chunks_by_range(
                file_id, start_order, end_order
            )
            return json.dumps(
                [
                    {
                        "text": r["text"],
                        "file_id": r["metadata"]["file_id"],
                        "order": r["metadata"]["order"],
                    }
                    for r in results
                ]
            )
        return json.dumps({"error": "Unknown tool"})

    def generate_comparison_graph(
        self, results: List[Dict], output_path: str = "model_performance_comparison.png"
    ):
        """Generate a comparison graph of performance metrics."""
        question_labels = [f"Q{r['question_id']}" for r in results]
        rag_scores = [r["agentic_rag"]["judge_score"] for r in results]
        full_context_scores = [r["full_context"]["judge_score"] for r in results]

        x = np.arange(len(question_labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 7))
        rects1 = ax.bar(
            x - width / 2, rag_scores, width, label="Agentic RAG", color="skyblue"
        )
        rects2 = ax.bar(
            x + width / 2,
            full_context_scores,
            width,
            label="Full Context",
            color="lightcoral",
        )

        ax.set_ylabel("Judge Score (1-5)")
        ax.set_title("Model Performance Comparison by Question")
        ax.set_xticks(x)
        ax.set_xticklabels(question_labels, rotation=45, ha="right")
        ax.set_ylim(0, 5.5)
        ax.legend()
        ax.bar_label(rects1, padding=3, fmt="%.2f")
        ax.bar_label(rects2, padding=3, fmt="%.2f")

        fig.tight_layout()
        plt.savefig(output_path)
        print(f"\nComparison graph saved to {output_path}")

    def generate_tool_usage_chart(
        self, results: List[Dict], output_path: str = "tool_usage_chart.png"
    ):
        """Generate a chart showing which tools were called for each question."""
        question_labels = [f"Q{r['question_id']}" for r in results]

        # Count tool calls per question
        tool_names = set()
        for r in results:
            for tool_call in r["agentic_rag"]["tool_calls_detail"]:
                tool_names.add(tool_call["name"])

        tool_names = sorted(list(tool_names))

        # Create a matrix of tool usage counts
        tool_usage = {tool: [] for tool in tool_names}
        for r in results:
            tool_counts = {tool: 0 for tool in tool_names}
            for tool_call in r["agentic_rag"]["tool_calls_detail"]:
                tool_counts[tool_call["name"]] += 1
            for tool in tool_names:
                tool_usage[tool].append(tool_counts[tool])

        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(question_labels))
        width = 0.6

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
        bottom = np.zeros(len(question_labels))

        for i, tool in enumerate(tool_names):
            counts = tool_usage[tool]
            ax.bar(
                x,
                counts,
                width,
                label=tool,
                bottom=bottom,
                color=colors[i % len(colors)],
            )
            bottom += counts

        ax.set_ylabel("Number of Tool Calls")
        ax.set_xlabel("Questions")
        ax.set_title("Tool Usage per Question (Agentic RAG)")
        ax.set_xticks(x)
        ax.set_xticklabels(question_labels, rotation=45, ha="right")
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        plt.savefig(output_path)
        print(f"Tool usage chart saved to {output_path}")

    def generate_token_usage_charts(
        self, results: List[Dict], output_path_prefix: str = "token_usage"
    ):
        """Generate charts showing token usage comparison."""
        question_labels = [f"Q{r['question_id']}" for r in results]

        # Per-question token usage
        rag_tokens = [r["agentic_rag"]["total_tokens"] for r in results]
        full_tokens = [r["full_context"]["total_tokens"] for r in results]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Subplot 1: Per-question token usage
        x = np.arange(len(question_labels))
        width = 0.35

        rects1 = ax1.bar(
            x - width / 2, rag_tokens, width, label="Agentic RAG", color="skyblue"
        )
        rects2 = ax1.bar(
            x + width / 2, full_tokens, width, label="Full Context", color="lightcoral"
        )

        ax1.set_ylabel("Total Tokens")
        ax1.set_xlabel("Questions")
        ax1.set_title("Token Usage per Question")
        ax1.set_xticks(x)
        ax1.set_xticklabels(question_labels, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)
        ax1.bar_label(rects1, padding=3, fmt="%d")
        ax1.bar_label(rects2, padding=3, fmt="%d")

        # Subplot 2: Total token usage comparison
        total_rag = sum(rag_tokens)
        total_full = sum(full_tokens)

        methods = ["Agentic RAG", "Full Context"]
        totals = [total_rag, total_full]
        colors_total = ["skyblue", "lightcoral"]

        bars = ax2.bar(methods, totals, color=colors_total, width=0.6)
        ax2.set_ylabel("Total Tokens")
        ax2.set_title("Total Token Usage Comparison")
        ax2.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, total in zip(bars, totals):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(total):,}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        # Add savings percentage
        if total_full > 0:
            savings_pct = ((total_full - total_rag) / total_full) * 100
            ax2.text(
                0.5,
                max(totals) * 0.5,
                f"Savings: {savings_pct:.1f}%",
                ha="center",
                fontsize=14,
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        fig.tight_layout()
        plt.savefig(f"{output_path_prefix}.png")
        print(f"Token usage charts saved to {output_path_prefix}.png")

    def run_tests(
        self, document_path: str, output_file: str = "rag_comparison_results.json"
    ):
        """Run complete test suite comparing RAG vs Full Context."""
        print("=" * 80)
        print("RAG vs Full Context Comparison Test")
        print("=" * 80)

        print("\n1. Loading document...")
        self.load_document(document_path)

        print("\n2. Generating test questions...")
        questions = self.generate_test_questions()
        questions.append("Summarize the entire document, capturing the key points.")
        print(f"Generated {len(questions)} questions")
        print("Judge model initialized with full document context.")

        # Phase 1: Test all questions with RAG
        print(f"\n{'=' * 80}")
        print("PHASE 1: Testing all questions with Agentic RAG")
        print("=" * 80)
        rag_results = []
        for i, question in enumerate(questions, 1):
            print(f"\n[RAG {i}/{len(questions)}] Testing: {question[:100]}...")
            rag_metrics = self.test_with_method(question, "rag")
            rag_results.append(rag_metrics)
            print(
                f"  -> Completed in {rag_metrics.time_taken:.2f}s, {rag_metrics.total_tokens} tokens"
            )

        # Phase 2: Test all questions with Full Context
        print(f"\n{'=' * 80}")
        print("PHASE 2: Testing all questions with Full Context")
        print("=" * 80)
        full_results = []
        for i, question in enumerate(questions, 1):
            print(f"\n[Full {i}/{len(questions)}] Testing: {question[:100]}...")
            full_metrics = self.test_with_method(question, "full")
            full_results.append(full_metrics)
            print(
                f"  -> Completed in {full_metrics.time_taken:.2f}s, {full_metrics.total_tokens} tokens"
            )

        # Phase 3: Judge all responses
        print(f"\n{'=' * 80}")
        print("PHASE 3: Judging all responses")
        print("=" * 80)
        results = []
        for i, (question, rag_metrics, full_metrics) in enumerate(
            zip(questions, rag_results, full_results), 1
        ):
            print(
                f"\n[Judge {i}/{len(questions)}] Evaluating responses for: {question[:100]}..."
            )

            # Judge RAG response
            print("  -> Judging RAG response...")
            rag_judgement = self.judge_response(
                rag_metrics.question, rag_metrics.response
            )
            rag_metrics.judge_score = rag_judgement.get("overall_score")
            rag_metrics.judge_rationale = rag_judgement.get("rationale")
            print(f"     RAG Score: {rag_metrics.judge_score}/5")

            # Judge Full Context response
            print("  -> Judging Full Context response...")
            full_judgement = self.judge_response(
                full_metrics.question, full_metrics.response
            )
            full_metrics.judge_score = full_judgement.get("overall_score")
            full_metrics.judge_rationale = full_judgement.get("rationale")
            print(f"     Full Context Score: {full_metrics.judge_score}/5")

            results.append(
                {
                    "question_id": i,
                    "question": question,
                    "agentic_rag": asdict(rag_metrics),
                    "full_context": asdict(full_metrics),
                }
            )

        # Save results
        print(f"\n{'=' * 80}")
        print("Saving results...")
        with open(output_file, "w") as f:
            json.dump(
                {
                    "test_config": {
                        "test_model": self.test_model,
                        "judge_model": self.judge_model,
                        "document": self.file_name,
                    },
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"Results saved to {output_file}")

        self.print_summary(results)
        self.generate_comparison_graph(results)
        self.generate_tool_usage_chart(results)
        self.generate_token_usage_charts(results)
        return results

    def print_summary(self, results: List[Dict]):
        """Print summary statistics."""
        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print("=" * 80)

        rag_total_tokens = sum(r["agentic_rag"]["total_tokens"] for r in results)
        full_total_tokens = sum(r["full_context"]["total_tokens"] for r in results)
        rag_total_time = sum(r["agentic_rag"]["time_taken"] for r in results)
        full_total_time = sum(r["full_context"]["time_taken"] for r in results)
        rag_avg_score = np.mean([r["agentic_rag"]["judge_score"] for r in results])
        full_avg_score = np.mean([r["full_context"]["judge_score"] for r in results])

        print(f"\n{'Metric':<30} {'Agentic RAG':<20} {'Full Context':<20}")
        print("-" * 70)
        print(f"{'Avg Judge Score':<30} {rag_avg_score:<20.2f} {full_avg_score:<20.2f}")
        print(f"{'Total Tokens':<30} {rag_total_tokens:<20} {full_total_tokens:<20}")
        print(
            f"{'Avg Tokens/Query':<30} {rag_total_tokens/len(results):<20.1f} {full_total_tokens/len(results):<20.1f}"
        )
        print(
            f"{'Total Time (s)':<30} {rag_total_time:<20.2f} {full_total_time:<20.2f}"
        )

        if full_total_tokens > 0:
            savings_pct = (
                (full_total_tokens - rag_total_tokens) / full_total_tokens
            ) * 100
            print(f"\nToken Savings with RAG: {savings_pct:.1f}%")


def list_files(folder_path: str) -> List[str]:
    """Return list of file paths in folder (non-recursive)."""
    files = []
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if os.path.isfile(full_path):
            files.append(full_path)
    return files


if __name__ == "__main__":
    # Configuration
    DOCS_FOLDER = "< add document path >"

    doc_paths = list_files(DOCS_FOLDER)
    if not doc_paths:
        print(f"No files found in folder '{DOCS_FOLDER}'. Nothing to test.")
        exit()

    tester = RAGTester()
    all_results = {}

    for doc_path in doc_paths:
        print(f"\n{'#' * 80}")
        print(f"Running tests for: {doc_path}")
        print("#" * 80)
        result = tester.run_tests(doc_path)
        all_results[doc_path] = result

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
    for path, res in all_results.items():
        print(f"{Path(path).name}: {len(res)} questions tested")

    print("\n✓ Test complete!")
