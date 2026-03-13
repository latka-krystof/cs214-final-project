# CS 214 Project

## 1. Setup
Install requirements:
`pip install -r requirements.txt`

## 2. Launch Backends
This starts two vLLM servers (Speculative & Standard) on ports 8001 and 8002.
`bash start_models.sh`
*Wait ~2 minutes for "Application startup complete" to appear twice.*

## 3. Start the Proxy
In a new terminal:
`uvicorn proxy:app --port 8000 --reload`

## 4. Run the Demo
Send a request to your proxy (Port 8000):
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TheBloke/Llama-2-7b-Chat-AWQ",
    "prompt": "Explain Quantum Physics",
    "max_tokens": 100,
    "temperature": 0
  }'