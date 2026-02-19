"""
vLLM Router Benchmark Script
Sends N requests (default 300) and measures total + per-request latency.
"""

import asyncio
import aiohttp
import time
import argparse
import statistics
import json
from dataclasses import dataclass, field
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_BASE_URL   = "http://localhost:8001"
DEFAULT_MODEL      = "Qwen/Qwen2.5-3B-Instruct-AWQ"          # change me
DEFAULT_N_REQUESTS = 300
DEFAULT_CONCURRENCY = 50                        # max simultaneous requests
DEFAULT_MAX_TOKENS = 128

# Mix of prompts across categories, lengths, and complexity levels
PROMPT_POOL = [
    # ── Factual / Short Answer ────────────────────────────────────────────────
    "What is the capital of France?",
    "What is the boiling point of water in Celsius?",
    "Who wrote 'Pride and Prejudice'?",
    "What year did World War II end?",
    "How many bones are in the human body?",
    "What planet is closest to the Sun?",
    "What is the chemical symbol for gold?",
    "Who painted the Mona Lisa?",
    "What is the speed of light in meters per second?",
    "How many continents are there on Earth?",

    # ── Coding ────────────────────────────────────────────────────────────────
    "Write a Python function to check if a number is prime.",
    "Give me a Python one-liner to reverse a string.",
    "Write a SQL query to find the top 5 customers by total spend.",
    "How do I merge two dictionaries in Python?",
    "Write a JavaScript function that debounces another function.",
    "What is the difference between == and === in JavaScript?",
    "Write a bash script that counts lines in all .txt files in a directory.",
    "Explain what a Python decorator is and give a simple example.",
    "Write a regex pattern to validate an email address.",
    "What is the time complexity of quicksort in the average case?",
    "Write a recursive function to compute the nth Fibonacci number in Python.",
    "How do you reverse a linked list in place?",
    "Show me a simple REST API endpoint using Python FastAPI.",
    "What is the difference between a stack and a queue?",

    # ── Reasoning / Math ─────────────────────────────────────────────────────
    "If a train travels 60 mph for 2.5 hours, how far does it go?",
    "What is 17 multiplied by 43?",
    "A rectangle has a perimeter of 36cm and width of 6cm. What is its area?",
    "If I flip a fair coin three times, what is the probability of getting exactly two heads?",
    "What comes next in the sequence: 2, 6, 12, 20, 30, ?",
    "A store marks up prices by 40% then offers a 20% discount. Is the item more or less expensive than original?",

    # ── Summarization / Long Context ─────────────────────────────────────────
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "In a few sentences, explain what the French Revolution was about.",
    "Briefly explain how HTTPS works.",
    "What were the main causes of the 2008 financial crisis?",
    "Summarize the theory of evolution in plain language.",
    "Explain the difference between supervised and unsupervised machine learning.",
    "What is the significance of the Turing Test?",

    # ── Creative Writing ──────────────────────────────────────────────────────
    "Write a haiku about the ocean.",
    "Write a two-sentence horror story.",
    "Write a limerick about a programmer who hates bugs.",
    "Write the opening line of a mystery novel set in Tokyo.",
    "Describe a sunset using only food metaphors.",
    "Write a short poem about the feeling of Monday mornings.",
    "Write a three-sentence fairy tale.",

    # ── Translation / Language ───────────────────────────────────────────────
    "Translate 'Hello, how are you?' into Spanish.",
    "Translate 'Good morning, have a nice day' into Japanese.",
    "Translate 'Where is the nearest hospital?' into French.",
    "What does the Latin phrase 'carpe diem' mean?",
    "What is the origin of the word 'algorithm'?",

    # ── Explanation / Education ───────────────────────────────────────────────
    "Explain quantum computing in simple terms.",
    "Describe the water cycle briefly.",
    "What are three benefits of regular exercise?",
    "How does a vaccine work?",
    "Explain what inflation is and what causes it.",
    "What is the difference between RAM and ROM?",
    "How does GPS know where I am?",
    "What is photosynthesis?",
    "Explain the concept of supply and demand.",
    "What is the difference between a virus and a bacterium?",
    "How does the Internet work at a high level?",
    "What is machine learning and how is it different from traditional programming?",

    # ── Opinion / Open-ended ─────────────────────────────────────────────────
    "What are the pros and cons of remote work?",
    "What are some effective strategies for managing stress?",
    "What skills will be most valuable in the job market in 10 years?",
    "What are the ethical concerns around facial recognition technology?",
    "Should cities prioritize public transit over car infrastructure? Give both sides.",

    # ── Lists / Recommendations ───────────────────────────────────────────────
    "List five programming languages and their main use cases.",
    "Name five must-read books on personal finance.",
    "What are five common mistakes new Python developers make?",
    "Give me five ideas for a healthy weekday lunch.",
    "List three ways to improve the performance of a slow database query.",
    "What are the top five Python libraries for data science?",

    # ── Instruction Following ────────────────────────────────────────────────
    "Give me a step-by-step plan to learn Spanish in six months.",
    "Walk me through how to make a cup of pour-over coffee.",
    "Explain how to set up a Python virtual environment step by step.",
    "How do I write a good resume? Give five actionable tips.",
]
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Result:
    index: int
    success: bool
    latency: float          # seconds
    tokens_generated: int = 0
    error: Optional[str] = None


async def send_request(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    index: int,
    base_url: str,
    model: str,
    max_tokens: int,
) -> Result:
    prompt = PROMPT_POOL[index % len(PROMPT_POOL)]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    url = f"{base_url}/v1/chat/completions"

    async with sem:
        t0 = time.perf_counter()
        try:
            async with session.post(url, json=payload) as resp:
                body = await resp.json()
                latency = time.perf_counter() - t0

                if resp.status != 200:
                    return Result(index, False, latency, error=str(body))

                tokens = body.get("usage", {}).get("completion_tokens", 0)
                return Result(index, True, latency, tokens_generated=tokens)

        except Exception as e:
            latency = time.perf_counter() - t0
            return Result(index, False, latency, error=str(e))


def print_report(results: list[Result], wall_time: float):
    successes = [r for r in results if r.success]
    failures  = [r for r in results if not r.success]
    latencies = [r.latency for r in successes]
    total_tokens = sum(r.tokens_generated for r in successes)

    print("\n" + "═" * 55)
    print("  vLLM Router Benchmark Report")
    print("═" * 55)
    print(f"  Total requests      : {len(results)}")
    print(f"  Successful          : {len(successes)}")
    print(f"  Failed              : {len(failures)}")
    print(f"  Wall-clock time     : {wall_time:.2f}s")
    print(f"  Throughput          : {len(successes)/wall_time:.2f} req/s")
    print(f"  Total tokens out    : {total_tokens}")
    print(f"  Token throughput    : {total_tokens/wall_time:.1f} tok/s")

    if latencies:
        print(f"\n  Latency (seconds):")
        print(f"    min   : {min(latencies):.3f}s")
        print(f"    p50   : {statistics.median(latencies):.3f}s")
        p95_idx = int(0.95 * len(latencies))
        p99_idx = int(0.99 * len(latencies))
        sorted_l = sorted(latencies)
        print(f"    p95   : {sorted_l[p95_idx]:.3f}s")
        print(f"    p99   : {sorted_l[p99_idx]:.3f}s")
        print(f"    max   : {max(latencies):.3f}s")
        print(f"    mean  : {statistics.mean(latencies):.3f}s")
        print(f"    stdev : {statistics.stdev(latencies):.3f}s" if len(latencies) > 1 else "")

    if failures:
        print(f"\n  Sample errors (first 3):")
        for r in failures[:3]:
            print(f"    [{r.index}] {r.error}")

    print("═" * 55 + "\n")


async def main(args):
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=120)

    print(f"Sending {args.n} requests to {args.base_url}  (concurrency={args.concurrency})")
    print(f"Model: {args.model}  |  max_tokens: {args.max_tokens}\n")

    t_start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            send_request(session, sem, i, args.base_url, args.model, args.max_tokens)
            for i in range(args.n)
        ]
        results = await asyncio.gather(*tasks)
    wall_time = time.perf_counter() - t_start

    print_report(list(results), wall_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a vLLM router endpoint")
    parser.add_argument("--base-url",    default=DEFAULT_BASE_URL,    help="Router base URL")
    parser.add_argument("--model",       default=DEFAULT_MODEL,       help="Model name")
    parser.add_argument("-n",            type=int, default=DEFAULT_N_REQUESTS, help="Total requests")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Max parallel requests")
    parser.add_argument("--max-tokens",  type=int, default=DEFAULT_MAX_TOKENS,  help="Max tokens per response")
    args = parser.parse_args()
    asyncio.run(main(args))