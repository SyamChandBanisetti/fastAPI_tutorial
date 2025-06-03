from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import base64, io, os, requests
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Allow frontend access (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_ENDPOINT = os.getenv("GEMINI_API_ENDPOINT")

# Convert image to base64
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

# Get analysis from Gemini
def get_gemini_analysis(encoded_image):
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [
                {"text": "Analyze this plant leaf image and identify diseases."},
                {"inlineData": {"mimeType": "image/jpeg", "data": encoded_image}}
            ]
        }]
    }
    response = requests.post(
        f"{GEMINI_API_ENDPOINT}?key={GEMINI_API_KEY}",
        headers=headers,
        json=payload
    )
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]

# Main API endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    encoded_image = encode_image(image_data)

    try:
        result_text = get_gemini_analysis(encoded_image)
        return {"result": result_text}
    except Exception as e:
        return {"error": str(e)}
