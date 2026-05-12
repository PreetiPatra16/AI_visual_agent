from fastapi import FastAPI
from app.api.routes.compare import router as compare_router

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Visual Testing Agent API")

# Include routes
app.include_router(compare_router)