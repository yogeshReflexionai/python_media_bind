from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from imagebind import data
from mangum import Mangum
from typing import List, Optional
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import Config, Server
import os
import uvicorn
import requests
import tempfile


app = FastAPI(swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"})
handler = Mangum(app)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=['*'],
#     allow_credentials=True,
#     allow_methods=['*'],
#     allow_headers=['*'],
# )

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

def download_file(url):
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        # Extract the file name from the URL
        file_name = os.path.basename(url)

        # Construct the local file path in the temporary directory
        local_file_path = os.path.join(temp_dir, file_name)

        # Download the file from the URL
        response = requests.get(url)
        response.raise_for_status()

        # Save the downloaded file locally
        with open(local_file_path, 'wb') as file:
            file.write(response.content)

        return local_file_path

    except Exception as e:
        print(f"Error downloading file: {e}")
        return None

class InputData(BaseModel):
    text_list: Optional[List[str]] = None
    image_paths: Optional[List[str]] = None  # Use UploadFile for file uploads
    audio_paths: Optional[List[str]] = None  # Use UploadFile for file uploads


@app.post("/process_data",response_model=dict)
async def process_data(input_data: InputData):

    inputs = {}
    text_list = input_data.text_list or []
    audio_paths = input_data.audio_paths or []
    image_paths = input_data.image_paths or []

    

    # Instantiate model
    

    # print(image_paths,audio_paths)


    # Load data
    if text_list:
        inputs[ModalityType.TEXT] = data.load_and_transform_text(text_list, device)
    if image_paths:
        image_paths = [download_file(image_path) for image_path in image_paths]
        inputs[ModalityType.VISION] = data.load_and_transform_vision_data(image_paths, device)
    if audio_paths:
        audio_paths = [download_file(audio_path) for audio_path in audio_paths]
        inputs[ModalityType.AUDIO] = data.load_and_transform_audio_data(audio_paths, device)



    # inputs = {
    #     ModalityType.TEXT: data.load_and_transform_text(text_list, device) if text_list else None,
    #     ModalityType.VISION: data.load_and_transform_vision_data([download_file(image_path) for image_path in image_paths], device) if image_paths else None,
    #     ModalityType.AUDIO: data.load_and_transform_audio_data([download_file(audio_path) for audio_path in audio_paths], device) if audio_paths else None,
    # }
    if not inputs:
        raise HTTPException(status_code=400, detail="Atleast one of the (text , image, audio) paths are required.")

    with torch.no_grad():
        embeddings = model(inputs)

    # print(embeddings)

    response_data = {
        "text": embeddings["text"].tolist() if "text" in embeddings else None,
        "vision": embeddings["vision"].tolist() if "vision" in embeddings else None,
        "audio": embeddings["audio"].tolist() if "audio" in embeddings else None,
    }

    return response_data

if __name__=="__main__":
  uvicorn.run(app,host="0.0.0.0",port=9000)
