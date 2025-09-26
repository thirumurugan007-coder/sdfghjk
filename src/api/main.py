from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"info": f"file '{file.filename}' uploaded successfully"}

@app.post("/analyze")
async def analyze_video(video_name: str):
    # Placeholder for video analysis logic
    # You would implement your video analysis here
    return {"info": f"Analyzing video '{video_name}'"}

@app.get("/playback/{video_name}")
async def playback_video(video_name: str):
    file_location = os.path.join(UPLOAD_DIR, video_name)
    return FileResponse(file_location)