import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import router, lifespan, state

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Turbofan RUL Prediction API",
    description=(
        "Real-time Remaining Useful Life prediction for turbofan engines "
        "using the NASA C-MAPSS FD001 dataset and pre-trained ML models."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)
@app.get("/")
def home():
    return {
        "message": "Turbofan RUL Prediction API is running"
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Inject the state directly into app.state for access if needed
app.state.turbofan = state
