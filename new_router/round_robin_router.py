import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from itertools import cycle

# --- CONFIGURATION ---
URL_SPECULATIVE = "http://localhost:8002/v1/chat/completions"
URL_STANDARD    = "http://localhost:8001/v1/chat/completions"

# --- GLOBAL STATE ---
backend_cycle = cycle([("speculative", URL_SPECULATIVE), ("standard", URL_STANDARD)])

@asynccontextmanager
async def lifespan(app):
    global client
    client = httpx.AsyncClient(timeout=60.0)
    yield
    await client.aclose()

app = FastAPI(title="SmartSpec Proxy", lifespan=lifespan)

# --- ROUTING LOGIC ---
def choose_backend() -> tuple[str, str]:
    """Pick the next backend in round-robin order."""
    return next(backend_cycle)

# --- MAIN ROUTE ---
@app.post("/v1/chat/completions")
async def smart_route(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    mode, target_url = choose_backend()
    print(f"[INFO] Routing to {mode} backend: {target_url}")

    try:
        response = await client.post(target_url, json=payload)
        return JSONResponse(response.json(), status_code=response.status_code)
    except httpx.RequestError as exc:
        return JSONResponse({"error": f"Backend failed: {exc}"}, status_code=502)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("round_robin_router:app", host="0.0.0.0", port=8000, reload=True)