import time
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# --- CONFIGURATION ---
URL_SPECULATIVE = "http://localhost:8002/v1/chat/completions"
URL_STANDARD    = "http://localhost:8001/v1/chat/completions"
CONCURRENCY_THRESHOLD = 10

# --- GLOBAL STATE ---
spec_count = 0
std_count  = 0

@asynccontextmanager
async def lifespan(app):
    global client
    client = httpx.AsyncClient(timeout=60.0)
    yield
    await client.aclose()

app = FastAPI(title="SmartSpec Proxy", lifespan=lifespan)

# --- ROUTING LOGIC ---
def choose_backend() -> tuple[str, str]:
    if spec_count < std_count:
        return URL_SPECULATIVE, "speculative"
    elif std_count < spec_count:
        return URL_STANDARD, "standard"
    else:  # tied — use threshold to pick the most efficient mode
        if spec_count < CONCURRENCY_THRESHOLD:
            return URL_SPECULATIVE, "speculative"
        else:
            return URL_STANDARD, "standard"

# --- MAIN ROUTE ---
@app.post("/v1/chat/completions")
async def smart_route(request: Request):
    global spec_count, std_count

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    target_url, mode = choose_backend()
    
    # Increment the appropriate counter
    if mode == "speculative":
        spec_count += 1
    else:
        std_count += 1

    # Print current counters
    print(f"[INFO] Speculative count: {spec_count}, Standard count: {std_count}")

    start = time.time()
    try:
        response = await client.post(target_url, json=payload)
        duration = time.time() - start

        headers = {
            "X-System-Mode":  mode,
            "X-Spec-Count":   str(spec_count),
            "X-Std-Count":    str(std_count),
            "X-Process-Time": f"{duration:.4f}",
        }
        return JSONResponse(response.json(), status_code=response.status_code, headers=headers)

    except httpx.RequestError as exc:
        return JSONResponse({"error": f"Backend failed: {exc}"}, status_code=502)

    finally:
        # Decrement after request finishes
        if mode == "speculative":
            spec_count -= 1
        else:
            std_count -= 1

        # Print counters after decrement (optional)
        print(f"[INFO] After request: Speculative count: {spec_count}, Standard count: {std_count}")

# --- STATS ---
@app.get("/stats")
async def get_stats():
    return {
        "spec_count":            spec_count,
        "std_count":             std_count,
        "concurrency_threshold": CONCURRENCY_THRESHOLD,
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("new_router:app", host="0.0.0.0", port=8000, reload=True)
# Run with: uvicorn proxy:app --port 8000 --workers 1
