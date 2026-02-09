import time
import httpx
import asyncio
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse

app = FastAPI(title="SmartSpec Proxy")

# --- CONFIGURATION ---
# The ports must match your start_models.sh script
URL_SPECULATIVE = "http://localhost:8001/v1/completions"
URL_STANDARD = "http://localhost:8002/v1/completions"

# THE MAGIC NUMBER: If pending requests > this, switch to Standard Mode
# You will tune this number for your demo graphs!
QUEUE_THRESHOLD = 8 

# Global State
current_queue_size = 0

@app.post("/v1/completions")
async def smart_route(request: Request):
    global current_queue_size
    
    # 1. READ REQUEST
    try:
        payload = await request.json()
    except:
        return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)

    # 2. DECISION LOGIC (The Innovation)
    # If the queue is empty, use Speculative (Latency Optimized)
    # If the queue is full, use Standard (Throughput Optimized)
    if current_queue_size < QUEUE_THRESHOLD:
        target_url = URL_SPECULATIVE
        mode = "speculative"
        backend_name = "Instance A (Hare)"
    else:
        target_url = URL_STANDARD
        mode = "standard"
        backend_name = "Instance B (Tortoise)"

    # 3. INCREMENT LOAD
    current_queue_size += 1
    start_time = time.time()

    try:
        # 4. FORWARD REQUEST (Proxy)
        # We use a 60s timeout because LLMs can be slow
        async with httpx.AsyncClient() as client:
            response = await client.post(
                target_url, 
                json=payload, 
                timeout=60.0
            )
            
        # 5. PROCESS RESPONSE
        duration = time.time() - start_time
        
        # We return the original JSON but inject our own metadata headers
        # so your data analysis scripts can see what happened.
        headers = dict(response.headers)
        headers["X-System-Mode"] = mode
        headers["X-Queue-Depth"] = str(current_queue_size)
        headers["X-Process-Time"] = f"{duration:.4f}"
        
        return JSONResponse(
            content=response.json(),
            status_code=response.status_code,
            headers=headers
        )

    except httpx.RequestError as exc:
        return JSONResponse(
            content={"error": f"Backend failed: {exc}"}, 
            status_code=502
        )
        
    finally:
        # 6. DECREMENT LOAD
        current_queue_size -= 1

@app.get("/stats")
async def get_stats():
    """Helper endpoint to see current load"""
    return {
        "current_queue_size": current_queue_size,
        "threshold": QUEUE_THRESHOLD,
        "status": "overloaded" if current_queue_size >= QUEUE_THRESHOLD else "idle"
    }

# Run with: uvicorn proxy:app --port 8000 --workers 1