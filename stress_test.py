import asyncio
import httpx
import time
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
PROXY_URL = "http://localhost:8000/v1/completions"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct-AWQ"
PROMPT = "The future of AI is"
MAX_TOKENS = 64

async def send_request(client, req_id):
    start = time.time()
    try:
        resp = await client.post(
            PROXY_URL,
            json={
                "model": MODEL_NAME,
                "prompt": PROMPT,
                "max_tokens": MAX_TOKENS,
                "temperature": 0
            },
            timeout=120.0
        )
        duration = time.time() - start
        mode = resp.headers.get("X-System-Mode", "unknown")
        print(f"Req {req_id}: {mode.upper()} took {duration:.2f}s")
        return {"id": req_id, "mode": mode, "latency": duration}
    except Exception as e:
        print(f"Req {req_id}: FAILED ({e})")
        return {"id": req_id, "mode": "failed", "latency": None}

async def run_batch(num_requests):
    print(f"\n--- Launching {num_requests} concurrent requests ---")
    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
    return results

async def main():
    # 1. Warmup
    print("Warming up...")
    await run_batch(1)
    
    # 2. Ramping Up Traffic
    all_data = []
    for concurrency in [1, 5, 10, 20]:
        batch_results = await run_batch(concurrency)
        for res in batch_results:
            res['concurrency'] = concurrency
            all_data.append(res)
            
    # 3. Analyze
    df = pd.DataFrame(all_data)
    print("\nSUMMARY:")
    print(df.groupby(['concurrency', 'mode'])['latency'].mean())
    df.to_csv("smartspec_results.csv", index=False)
    print("Results saved to smartspec_results.csv")

if __name__ == "__main__":
    asyncio.run(main())
