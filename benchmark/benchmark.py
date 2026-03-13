"""
vLLM Standard vs Speculative Benchmark Script

1. Run the machines (from start.sh) separately since each of them will take 0.85 gpu
2. Kill the machines after running benchmark to ensure fair comparison

BENCHMARK
Standard: 8001
Speculative: 8003 

Concurrency 10:
python benchmark.py --base-url http://localhost:8001 -n 100 --concurrency 10
python benchmark.py --base-url http://localhost:8003 -n 100 --concurrency 10

Concurrency 20:
python benchmark.py --base-url http://localhost:8001 -n 100 --concurrency 20
python benchmark.py --base-url http://localhost:8003 -n 100 --concurrency 20

Concurrency 50:
python benchmark.py --base-url http://localhost:8001 -n 100 --concurrency 50
python benchmark.py --base-url http://localhost:8003 -n 100 --concurrency 50    


Concurrency 10: Speculative wins

Standard:

═══════════════════════════════════════════════════════
  vLLM Benchmark Report
═══════════════════════════════════════════════════════
  Total requests      : 100
  Successful          : 100
  Failed              : 0
  Wall-clock time     : 225.39s
  Throughput          : 0.44 req/s
  Total tokens out    : 45784
  Token throughput    : 203.1 tok/s

  Latency (seconds):
    min   : 0.800s
    p50   : 22.647s
    p95   : 48.299s
    p99   : 49.245s
    max   : 49.245s
    mean  : 21.243s
    stdev : 13.497s
═══════════════════════════════════════════════════════

Speculative:

═══════════════════════════════════════════════════════
  vLLM Benchmark Report
═══════════════════════════════════════════════════════
  Total requests      : 100
  Successful          : 100
  Failed              : 0
  Wall-clock time     : 170.60s
  Throughput          : 0.59 req/s
  Total tokens out    : 38134
  Token throughput    : 223.5 tok/s

  Latency (seconds):
    min   : 0.752s
    p50   : 15.288s
    p95   : 34.386s
    p99   : 45.547s
    max   : 45.547s
    mean  : 15.668s
    stdev : 11.005s
═══════════════════════════════════════════════════════

Concurrency 20: Standard wins


Standard:
═══════════════════════════════════════════════════════
  vLLM Benchmark Report
═══════════════════════════════════════════════════════
  Total requests      : 100
  Successful          : 100
  Failed              : 0
  Wall-clock time     : 149.17s
  Throughput          : 0.67 req/s
  Total tokens out    : 40463
  Token throughput    : 271.3 tok/s

  Latency (seconds):
    min   : 0.705s
    p50   : 28.628s
    p95   : 55.756s
    p99   : 67.621s
    max   : 67.621s
    mean  : 26.468s
    stdev : 17.109s
═══════════════════════════════════════════════════════

Speuclative:
═══════════════════════════════════════════════════════
  vLLM Benchmark Report
═══════════════════════════════════════════════════════
  Total requests      : 100
  Successful          : 100
  Failed              : 0
  Wall-clock time     : 156.54s
  Throughput          : 0.64 req/s
  Total tokens out    : 39583
  Token throughput    : 252.9 tok/s

  Latency (seconds):
    min   : 2.048s
    p50   : 21.700s
    p95   : 55.606s
    p99   : 70.238s
    max   : 70.238s
    mean  : 23.716s
    stdev : 17.360s
═══════════════════════════════════════════════════════


Concurrency 50: Standard wins

Standard:
═══════════════════════════════════════════════════════
  vLLM Benchmark Report
═══════════════════════════════════════════════════════
  Total requests      : 100
  Successful          : 100
  Failed              : 0
  Wall-clock time     : 89.23s
  Throughput          : 1.12 req/s
  Total tokens out    : 34274
  Token throughput    : 384.1 tok/s

  Latency (seconds):
    min   : 0.842s
    p50   : 22.503s
    p95   : 80.246s
    p99   : 86.420s
    max   : 86.420s
    mean  : 30.850s
    stdev : 25.821s
═══════════════════════════════════════════════════════

Speuclative:

═══════════════════════════════════════════════════════
  vLLM Benchmark Report
═══════════════════════════════════════════════════════
  Total requests      : 100
  Successful          : 100
  Failed              : 0
  Wall-clock time     : 96.47s
  Throughput          : 1.04 req/s
  Total tokens out    : 33758
  Token throughput    : 349.9 tok/s

  Latency (seconds):
    min   : 2.290s
    p50   : 32.621s
    p95   : 76.429s
    p99   : 84.270s
    max   : 84.270s
    mean  : 34.541s
    stdev : 25.306s
═══════════════════════════════════════════════════════


"""

import asyncio
import aiohttp
import time
import argparse
import statistics
import random
from dataclasses import dataclass, field
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "meta-llama/Llama-3.2-3B-Instruct"          # change me
DEFAULT_N_REQUESTS = 100
DEFAULT_CONCURRENCY = 20                        # max simultaneous requests
DEFAULT_MAX_TOKENS = 1024

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
    "Write a 600‑word essay about Google",

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

    # ── 78 additional ────────────────────────────────────────────────
    "Act as a McKinsey senior partner. Tear apart this business idea: [idea]. Be brutally honest.",
    "If this startup fails in 2 years, what will have been the top 5 causes?",
    "Design a moat strategy for this AI product in a highly competitive market.",
    "How would Amazon enter this industry?",
    "Build a zero-to-one roadmap for dominating this niche.",
    "If I wanted to build the Bloomberg Terminal of legal intelligence, what would it look like?",
    "Design a regulatory-safe AI legal assistant architecture.",
    "What are the real risks of deploying a legal LLM in production?",
    "Design a scalable React Native + FastAPI backend architecture for an AI-powered app.",
    "Refactor this code for production-level reliability: [paste code].",
    "How would a FAANG engineer structure this repo?",
    "Rewrite this code as if it needs to support 1 million users.",
    "Profile this backend for performance bottlenecks.",
    "Rewrite this for 10x speed.",
    "What would break at scale?",
    "Make this code enterprise-grade.",
    "Rewrite this using SOLID principles.",
    "Add proper logging, error handling, and observability.",
    "Explain this using first principles.",
    "What would Naval Ravikant think about this strategy?",
    "Analyze this using game theory.",
    "Apply inversion to this problem.",
    "What are the second-order consequences?",
    "Design a 12-month plan to get into FAANG as a high-impact AI product leader.",
    "What projects would make my GitHub irresistible to recruiters?",
    "Rewrite my resume to signal top 1% talent.",
    "If you were a Google hiring manager, what would worry you about my profile?",
    "How do I become antifragile in tech?",
    "Turn this idea into a high-performing LinkedIn post.",
    "Rewrite this in the voice of Paul Graham.",
    "Make this contrarian and bold.",
    "Turn this into a Twitter thread.",
    "Position me as a thought leader in AI x law.",
    "Ask me 10 uncomfortable questions about this idea.",
    "Debate me.",
    "Argue the opposite position.",
    "Simulate a VC pitch meeting.",
    "Simulate a skeptical regulator.",
    "Write a YC-style application answer for this startup.",
    "What traction metrics matter most for this?",
    "Design a pitch deck outline.",
    "What would make this fundable?",
    "If Sequoia rejected this, why?",
    "Break this down into assumptions.",
    "Quantify the TAM realistically.",
    "Build a back-of-the-envelope model.",
    "Where are we likely overconfident?",
    "Stress test this strategy.",
    "Give me 50 ideas in 3 minutes.",
    "Generate 100 startup ideas in AI x law.",
    "List 30 niche markets no one talks about.",
    "Give me 25 unfair advantages I might have.",
    "Generate 40 business models.",
    "Design a daily routine for a future billionaire tech founder.",
    "What habits compound the most over 10 years?",
    "Audit my current workflow.",
    "Design a 90-day deep work sprint.",
    "How do I think bigger?",
    "What if this was illegal?",
    "What if this had to scale to 1 billion users?",
    "What if this had zero budget?",
    "What if OpenAI built this?",
    "What would this look like in 2035?",
    "Explain C++ memory management like I’m building a high-frequency trading system.",
    "Explain this concept visually.",
    "Teach me this using analogies.",
    "Test my understanding with hard questions.",
    "Make the strongest case for quitting this.",
    "Make the strongest case for doubling down.",
    "What information would change this decision?",
    "What are we not seeing?",
    "If this were easy, what would it look like?",
    "Act as a combination of a YC partner, a constitutional lawyer in Hong Kong, and a FAANG staff engineer. Critique this AI legal assistant architecture for scalability, regulatory risk, and defensibility.",
    "Respond in brutal truth mode. No motivational fluff. Only strategic insights.",
    "Break your answer into: Risks / Opportunities / Blind Spots / 10x Move."
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
    prompt = random.choice(PROMPT_POOL)
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
                content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"Prompt: {prompt}\nResponse: {content}\n{'-'*50}\n")
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
    print("  vLLM Benchmark Report")
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
    parser = argparse.ArgumentParser(description="Benchmark a vLLM endpoint")
    parser.add_argument("--base-url",required=True,help="Router base URL (e.g. http://localhost:8001)")
    parser.add_argument("--model",       default=DEFAULT_MODEL,       help="Model name")
    parser.add_argument("-n",            type=int, default=DEFAULT_N_REQUESTS, help="Total requests")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Max parallel requests")
    parser.add_argument("--max-tokens",  type=int, default=DEFAULT_MAX_TOKENS,  help="Max tokens per response")
    args = parser.parse_args()
    asyncio.run(main(args))