import asyncio
import httpx
import time
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
ENDPOINTS = {
    "Static Speculative": "http://localhost:8001/v1/completions",
    "Static Standard":    "http://localhost:8002/v1/completions",
    "SmartSpec (Ours)":   "http://localhost:8000/v1/completions"
}

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct-AWQ"
PROMPT = "The future of AI is"
MAX_TOKENS = 64

# Experimental Settings
LOAD_LEVELS = [1, 5, 10, 15, 20, 25]
NUM_TRIALS = 3  # Run each test 3 times to get an average

async def send_request(client, url):
    start = time.time()
    try:
        resp = await client.post(
            url,
            json={
                "model": MODEL_NAME,
                "prompt": PROMPT,
                "max_tokens": MAX_TOKENS,
                "temperature": 0
            },
            timeout=30.0
        )
        resp.raise_for_status()
        return time.time() - start
    except Exception:
        return None

async def warmup(url):
    """Fires a request to ensure the model is loaded and compiled."""
    print(f"  Warmup -> {url}...", end="", flush=True)
    async with httpx.AsyncClient() as client:
        await send_request(client, url)
    print(" Done.")

async def benchmark_system(system_name, url, concurrency):
    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, url) for _ in range(concurrency)]
        latencies = await asyncio.gather(*tasks)
    
    valid = [l for l in latencies if l is not None]
    if not valid:
        return float('nan')
    return np.mean(valid)

async def main():
    results = []
    
    print("=== STARTING AVERAGED BENCHMARK SUITE ===")
    
    # 1. Global Warmup (Fixes the 'Cold Start' anomaly)
    print("--- Phase 1: Warming Up Engines ---")
    for name, url in ENDPOINTS.items():
        await warmup(url)
    
    # 2. The Loop
    print("\n--- Phase 2: Running Experiments ---")
    for concurrency in LOAD_LEVELS:
        print(f"\nLoad Level: {concurrency} Concurrent Requests")
        
        for name, url in ENDPOINTS.items():
            trial_latencies = []
            
            for i in range(NUM_TRIALS):
                print(f"  [{name}] Trial {i+1}/{NUM_TRIALS}...", end="", flush=True)
                lat = await benchmark_system(name, url, concurrency)
                
                if not np.isnan(lat):
                    trial_latencies.append(lat)
                    print(f" {lat:.2f}s")
                else:
                    print(f" FAILED")
                
                # Critical: Let queues drain between trials!
                time.sleep(1)
            
            # Calculate Statistics
            if trial_latencies:
                avg_lat = np.mean(trial_latencies)
                std_dev = np.std(trial_latencies)
                
                results.append({
                    "System": name,
                    "Concurrency": concurrency,
                    "Latency": avg_lat,
                    "StdDev": std_dev
                })
            else:
                print(f"  [{name}] ALL TRIALS FAILED.")

    # 3. Save
    df = pd.DataFrame(results)
    df.to_csv("thesis_data_averaged.csv", index=False)
    print("\nBenchmark complete! Data saved to 'thesis_data_averaged.csv'")

if __name__ == "__main__":
    asyncio.run(main())