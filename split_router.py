import re
from pathlib import Path

path = Path('app/api/router.py')
content = path.read_text()

# Replace `app = FastAPI(...)` and `app.add_middleware(...)` with `router = APIRouter(...)`
# We will basically create main.py, state.py, and router.py.

# Actually, let's keep things extremely simple.
# 1. In router.py: Replace `@app.` with `@router.`
content = content.replace("from fastapi import FastAPI, ", "from fastapi import APIRouter, FastAPI, ")
content = content.replace("@app.post(", "@router.post(")
content = content.replace("@app.get(", "@router.get(")
content = content.replace("@app.delete(", "@router.delete(")
content = content.replace("@app.websocket(", "@router.websocket(")

# Inject router = APIRouter() right before the first @router
first_at = content.find("@router.")
if first_at != -1:
    content = content[:first_at] + "router = APIRouter()\n\n" + content[first_at:]
    
path.write_text(content)
print("router.py references updated")
