import random
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

URL_SPECULATIVE = "http://localhost:8001/v1/chat/completions"
URL_STANDARD    = "http://localhost:8002/v1/chat/completions"

@asynccontextmanager
async def lifespan(app):
    global client
    client = httpx.AsyncClient(timeout=60.0)
    yield
    await client.aclose()

app = FastAPI(title="Random Router", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def random_route(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    # Randomly pick backend
    target_url = random.choice([URL_SPECULATIVE, URL_STANDARD])

    try:
        response = await client.post(target_url, json=payload)
        return JSONResponse(
            response.json(),
            status_code=response.status_code
        )

    except httpx.RequestError as exc:
        return JSONResponse({"error": f"Backend failed: {exc}"}, status_code=502)
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("random_router:app", host="0.0.0.0", port=8000, reload=True)