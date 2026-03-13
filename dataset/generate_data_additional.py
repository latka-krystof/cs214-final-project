

import argparse
import json
import random
import numpy as np
from dataclasses import dataclass, asdict
from typing import List

SPECULATIVE_REQUESTS = [

    # ── Pure repetition ───────────────────────────────────────────────────────
    # Draft model predicts the repeated token/phrase with ~100% accuracy
    ("Repeat the word 'hello' exactly 200 times separated by spaces. Nothing else.",                            512, 0.0, "spec"),
    ("Repeat the word 'error' exactly 150 times, one per line. No other text.",                                 400, 0.0, "spec"),
    ("Write the phrase 'the quick brown fox' exactly 60 times, each on its own line.",                          400, 0.0, "spec"),
    ("Repeat 'ping 192.168.1.1' exactly 80 times, one per line.",                                              350, 0.0, "spec"),
    ("Repeat the string 'null' exactly 200 times separated by commas.",                                         400, 0.0, "spec"),
    ("Write 'PASS' exactly 100 times, one per line.",                                                           300, 0.0, "spec"),
    ("Repeat '0x00' exactly 120 times separated by spaces.",                                                    300, 0.0, "spec"),
    ("Write the word 'loading' followed by '...' exactly 80 times, one per line.",                             350, 0.0, "spec"),

    # ── Numeric sequences ─────────────────────────────────────────────────────
    # Counting is maximally predictable — draft always knows what comes next
    ("Print integers from 1 to 150, one per line, no other text.",                                              512, 0.0, "spec"),
    ("Count down from 200 to 1, one number per line.",                                                          512, 0.0, "spec"),
    ("List the first 100 even numbers, one per line.",                                                          400, 0.0, "spec"),
    ("List the first 100 odd numbers, one per line.",                                                            400, 0.0, "spec"),
    ("Print every multiple of 3 from 3 to 300, one per line.",                                                  400, 0.0, "spec"),
    ("Write the squares of integers from 1 to 80, one per line as: N^2 = M.",                                  400, 0.0, "spec"),
    ("Write the multiplication table for 9 from 9x1 to 9x40, one equation per line.",                          400, 0.0, "spec"),
    ("List all powers of 2 from 2^1 to 2^30, one per line as: 2^N = M.",                                      350, 0.0, "spec"),
    ("Print a Fibonacci sequence up to the 60th term, one number per line.",                                    400, 0.0, "spec"),

    # ── Structured data formats ───────────────────────────────────────────────
    # Syntax constraints mean the draft model can predict tokens with high accuracy
    ("Generate a CSV with columns id,name,value. Fill 80 rows with sequential data.",                          400, 0.0, "spec"),
    ("Write a JSON array of 40 objects, each with fields: id (int), status (string 'active').",                400, 0.0, "spec"),
    ("Repeat this JSON object 50 times on separate lines: {\"code\": 200, \"status\": \"ok\"}",                400, 0.0, "spec"),
    ("Write a YAML list of 60 items: - item_N: value_N where N increments.",                                   400, 0.0, "spec"),
    ("Generate 50 lines of nginx access log entries in standard format with sequential IPs.",                   400, 0.0, "spec"),
    ("Write a .env file with 50 environment variables: VAR_N=value_N where N increments.",                      350, 0.0, "spec"),
    ("Generate 40 lines of /etc/hosts entries: 192.168.1.N hostname-N.local",                                  300, 0.0, "spec"),
    ("Write 50 cron job entries, one per line: */N * * * * /usr/bin/job_N.sh",                                 350, 0.0, "spec"),
    ("Generate a markdown table with columns ID, Name, Score, Grade. Fill 40 rows sequentially.",              400, 0.0, "spec"),

    # ── Repetitive code patterns ──────────────────────────────────────────────
    # Code syntax is highly constrained — draft model learns the pattern quickly
    ("Write 60 lines of Python: print('Processing item N') where N goes from 1 to 60.",                        400, 0.0, "spec"),
    ("Write 50 Python assert statements: assert result_N == expected_N, 'Test N failed'",                      400, 0.0, "spec"),
    ("Write 40 SQL INSERT statements into a users table with sequential id, name_N, email_N.",                  400, 0.0, "spec"),
    ("Write a bash for loop from 1 to 100 that echoes 'Processing file N of 100'.",                            350, 0.0, "spec"),
    ("Write 50 JavaScript console.log statements: console.log('Step N:', result_N);",                          400, 0.0, "spec"),
    ("Write 40 Python import statements importing module_1 through module_40.",                                  300, 0.0, "spec"),
    ("Write a Python list comprehension expanding to 100 elements, then print each.",                           350, 0.0, "spec"),
    ("Write 30 HTML list items: <li>Item N: Description of item N</li>",                                       300, 0.0, "spec"),

    # ── Fixed templates ───────────────────────────────────────────────────────
    # Template slots are filled predictably — draft model learns the slot pattern
    ("Write 60 lines: 'Step N: Execute command N and verify output N.'",                                        400, 0.0, "spec"),
    ("List 80 server hostnames: server-001.prod.example.com through server-080.",                               400, 0.0, "spec"),
    ("Write 50 git commit messages: 'fix(module-N): resolve issue N in component N'",                          400, 0.0, "spec"),
    ("Generate 40 UUID-like strings: xxxxxxxx-xxxx-4xxx-yxxx-N where N is sequential.",                        350, 0.0, "spec"),
    ("Write 50 ticket IDs and titles: PROJ-N: Fix bug in module N",                                            350, 0.0, "spec"),
    ("List 60 API endpoint paths: /api/v1/resource/N/subresource/N",                                           350, 0.0, "spec"),
    ("Write 40 error messages: 'ERROR [2024-01-N]: Module N failed with code N'",                              350, 0.0, "spec"),
    ("Generate 50 test function names: def test_feature_N_returns_expected_value_N():",                        350, 0.0, "spec"),

    # ── Long structured prose ─────────────────────────────────────────────────
    # Structured enough that the draft model predicts well despite being prose-like
    ("Write a Python class with 15 methods. Each method is named process_step_N and prints its name.",         400, 0.0, "spec"),
    ("Write 30 dictionary definitions in this format: WORD_N (noun): The Nth example of a defined term.",      350, 0.0, "spec"),
    ("List all 50 US states alphabetically, one per line, preceded by their number.",                           300, 0.0, "spec"),
    ("List all 12 months with the number of days each has, repeated 5 times total.",                            300, 0.0, "spec"),
    ("Write the NATO phonetic alphabet 4 times, one letter per line each time.",                                 300, 0.0, "spec"),
]

# ── Standard-favored requests ─────────────────────────────────────────────────
# All: temp >= 1.0 OR short output (max_tokens <= 80), unpredictable content

STANDARD_REQUESTS = [

    # ── High-entropy creative one-liners ─────────────────────────────────────
    # Completely open-ended — any token could plausibly come next
    ("Give me a one-sentence tagline for a startup that sells underwater drones.",                               60,  1.0, "std"),
    ("In one sentence, what would a pirate say about machine learning?",                                         60,  1.0, "std"),
    ("Write a haiku about debugging code at 3am.",                                                               40,  1.0, "std"),
    ("Suggest a name for a coffee shop inside a library. One name only.",                                        20,  1.0, "std"),
    ("Describe the taste of nostalgia in one sentence.",                                                         50,  1.0, "std"),
    ("What would a robot's midlife crisis look like? One sentence.",                                             60,  1.0, "std"),
    ("Invent a new word and define it in one sentence.",                                                         50,  1.0, "std"),
    ("Write the worst possible fortune cookie message.",                                                         30,  1.0, "std"),
    ("What is the worst superpower to have in a library? One sentence.",                                         50,  1.0, "std"),
    ("Describe silence using only food metaphors. One sentence.",                                                 60,  1.0, "std"),
    ("What would Plato say about TikTok? One sentence.",                                                         60,  1.0, "std"),
    ("Invent a name and definition for the fear of slightly damp socks.",                                        70,  1.0, "std"),
    ("Write a 2-sentence legal disclaimer for a time machine rental service.",                                   80,  1.0, "std"),
    ("Describe quantum entanglement using only breakfast food metaphors.",                                        70,  1.0, "std"),
    ("What is the plot of a telenovela set inside a compiler? Two sentences.",                                   80,  1.0, "std"),
    ("Invent a cocktail named after a sorting algorithm. Name and one-sentence description.",                    60,  1.0, "std"),
    ("Write the opening line of a nature documentary about office printers.",                                    80,  1.0, "std"),
    ("What life advice would a very old tortoise give a mayfly? One sentence.",                                  60,  1.0, "std"),
    ("Describe an economy that uses anxiety as currency. Two sentences.",                                         80,  1.0, "std"),
    ("Write a 2-sentence weather forecast for a city that only exists in dreams.",                               70,  1.0, "std"),

    # ── Surreal / abstract ────────────────────────────────────────────────────
    # Maximally unpredictable — abstract concepts with no obvious token continuation
    ("What color is Wednesday? One sentence.",                                                                    40,  1.2, "std"),
    ("Describe the sound of the number 7. One sentence.",                                                        50,  1.2, "std"),
    ("What does déjà vu smell like? One sentence.",                                                              50,  1.2, "std"),
    ("If boredom were a texture, what would it feel like? One sentence.",                                        50,  1.2, "std"),
    ("Describe the weight of a memory. One sentence.",                                                           50,  1.2, "std"),
    ("What does the color blue taste like on a rainy Tuesday? One sentence.",                                    60,  1.2, "std"),
    ("If nostalgia were a piece of furniture, what would it be? One sentence.",                                  50,  1.2, "std"),
    ("Describe ambition as if it were a weather pattern.",                                                        60,  1.2, "std"),
    ("What is the opposite of an echo? One sentence.",                                                           40,  1.2, "std"),
    ("If regret had a postal address, where would it live? One sentence.",                                       50,  1.2, "std"),

    # ── Unexpected combinations ───────────────────────────────────────────────
    # Forces model to combine unrelated domains — unpredictable output
    ("Describe a medieval siege using only cooking metaphors. One sentence.",                                    70,  1.0, "std"),
    ("Explain blockchain to a golden retriever. One sentence.",                                                  70,  1.0, "std"),
    ("What would a GPS say if it developed existential dread? One sentence.",                                    60,  1.0, "std"),
    ("Describe the French Revolution as a Yelp review. Two sentences.",                                          80,  1.0, "std"),
    ("Write a LinkedIn post from the perspective of a medieval blacksmith.",                                     80,  1.0, "std"),
    ("What would a coral reef's Airbnb listing say? Two sentences.",                                             80,  1.0, "std"),
    ("Describe photosynthesis as a heated political debate. One sentence.",                                       70,  1.0, "std"),
    ("Write a one-sentence Yelp review of the concept of time.",                                                 60,  1.0, "std"),
    ("Explain gravity to someone who has only ever lived in water. One sentence.",                               70,  1.0, "std"),
    ("Write a performance review for the sun. Two sentences.",                                                   80,  1.0, "std"),

    # ── Short factual QA ──────────────────────────────────────────────────────
    # Short output — not enough tokens to amortize draft overhead even at temp=0
    ("What is the capital of Iceland?",                                                                          15,  0.0, "std"),
    ("Name three programming languages invented before 1980.",                                                   40,  0.0, "std"),
    ("What does HTTP stand for?",                                                                                15,  0.0, "std"),
    ("In one sentence, what is a binary search tree?",                                                           60,  0.0, "std"),
    ("What year was the first iPhone released?",                                                                 10,  0.0, "std"),
    ("What is the speed of light in meters per second?",                                                         20,  0.0, "std"),
    ("Name the four noble gases.",                                                                                25,  0.0, "std"),
    ("What does CPU stand for?",                                                                                 15,  0.0, "std"),
    ("Who wrote the TCP/IP protocol?",                                                                           30,  0.0, "std"),
    ("What is the time complexity of quicksort in the average case?",                                            30,  0.0, "std"),
]


BASE_PHASES = [
    (0/60,  5/60,  0.05, "low_load"),    # first  5/60 — quiet baseline
    (5/60,  10/60, 0.10, "ramp_up"),     # next   5/60 — increasing load
    (10/60, 15/60, 0.27, "burst_1"),     # next   5/60 — first burst
    (15/60, 30/60, 0.10, "recovery_1"),  # next  15/60 — long quiet recovery
    (30/60, 35/60, 0.08, "steady"),      # next   5/60 — steady mid load
    (35/60, 40/60, 0.30, "burst_2"),     # next   5/60 — peak burst
    (40/60, 60/60, 0.10, "cool_down"),   # last  20/60 — cool down
]


def make_phases(duration_s: int, total_requests: int) -> List[tuple]:
    phases = []
    allocated = 0
    for i, (start_frac, end_frac, proportion, label) in enumerate(BASE_PHASES):
        start_s = int(start_frac * duration_s)
        end_s   = int(end_frac   * duration_s)
        # Last phase gets remainder to ensure total_requests is exact
        if i == len(BASE_PHASES) - 1:
            n = total_requests - allocated
        else:
            n = round(proportion * total_requests)
            allocated += n
        phases.append((start_s, end_s, n, label))
    return phases


@dataclass
class TraceRequest:
    request_id:   int
    arrival_time: float   # seconds from t=0
    prompt:       str
    max_tokens:   int
    temperature:  float
    request_type: str     # "spec" or "std"
    phase:        str



def generate_trace(duration_s: int = 3600,
                   total_requests: int = 300,
                   seed: int = 42) -> List[TraceRequest]:
    
    rng    = random.Random(seed)
    np.random.seed(seed)
    phases = make_phases(duration_s, total_requests)

    requests = []
    req_id   = 0

    for start, end, n_requests, phase_label in phases:
        if n_requests == 0:
            continue
        duration = end - start
        rate     = n_requests / duration

        inter_arrivals = np.random.exponential(1.0 / rate, size=n_requests * 4)
        timestamps     = np.cumsum(inter_arrivals) + start
        timestamps     = timestamps[timestamps < end][:n_requests]

        if len(timestamps) < n_requests:
            extra = np.linspace(start, end, n_requests - len(timestamps) + 2)[1:-1]
            timestamps = np.sort(np.concatenate([timestamps, extra]))[:n_requests]

        for t in timestamps:
            pool = SPECULATIVE_REQUESTS if rng.random() < 0.5 else STANDARD_REQUESTS
            prompt, max_tokens, temperature, req_type = rng.choice(pool)

            requests.append(TraceRequest(
                request_id   = req_id,
                arrival_time = round(float(t), 3),
                prompt       = prompt,
                max_tokens   = max_tokens,
                temperature  = temperature,
                request_type = req_type,
                phase        = phase_label,
            ))
            req_id += 1

    requests.sort(key=lambda r: r.arrival_time)
    return requests

def save_trace(requests: List[TraceRequest], path: str,
               duration_s: int, total_requests: int):
    phases = make_phases(duration_s, total_requests)
    data = {
        "metadata": {
            "total_requests": len(requests),
            "spec_requests":  len([r for r in requests if r.request_type == "spec"]),
            "std_requests":   len([r for r in requests if r.request_type == "std"]),
            "duration_s":     duration_s,
            "duration_min":   duration_s / 60,
            "phases":         [[s, e, n, l] for s, e, n, l in phases],
        },
        "requests": [asdict(r) for r in requests],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(requests)} requests to {path}")


def load_trace(path: str) -> List[TraceRequest]:
    with open(path) as f:
        data = json.load(f)
    return [TraceRequest(**r) for r in data["requests"]]

def print_trace_summary(requests: List[TraceRequest]):
    phase_order = ["low_load", "ramp_up", "burst_1", "recovery_1",
                   "steady", "burst_2", "cool_down"]

    print(f"\n  Total requests : {len(requests)}")
    print(f"  Spec-favored   : {len([r for r in requests if r.request_type == 'spec'])}")
    print(f"  Std-favored    : {len([r for r in requests if r.request_type == 'std'])}")
    dur = max(r.arrival_time for r in requests)
    print(f"  Duration       : {dur:.1f}s ({dur/60:.1f} min)")
    print(f"\n  {'Phase':<15} {'Reqs':>6} {'Start':>9} {'End':>9} {'Gap':>12}")
    print(f"  {'─'*15} {'─'*6} {'─'*9} {'─'*9} {'─'*12}")
    for phase in phase_order:
        group = [r for r in requests if r.phase == phase]
        if not group:
            continue
        t0  = min(r.arrival_time for r in group)
        t1  = max(r.arrival_time for r in group)
        gap = (t1 - t0) / len(group) if len(group) > 1 else 0
        print(f"  {phase:<15} {len(group):>6} {t0:>8.1f}s {t1:>8.1f}s {gap:>10.1f}s/req")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartSpec trace generator")
    parser.add_argument("--output",   type=str, default="trace.json",
                        help="Output JSON file (default: trace.json)")
    parser.add_argument("--duration", type=int, default=60,
                        help="Total duration in MINUTES (default: 60)")
    parser.add_argument("--requests", type=int, default=300,
                        help="Total number of requests (default: 300)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    duration_s = args.duration * 60

    print(f"\n  Generating trace: {args.duration} min | "
          f"{args.requests} requests | seed={args.seed}")

    requests = generate_trace(duration_s=duration_s,
                              total_requests=args.requests,
                              seed=args.seed)
    print_trace_summary(requests)
    save_trace(requests, args.output, duration_s, args.requests)