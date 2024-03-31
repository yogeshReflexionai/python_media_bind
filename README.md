# python_media_bind
Embed Text, Image and Audio


# installation
git clone https://github.com/facebookresearch/ImageBind.git
cd ImageBind
pip install -e .
cd ..
pip install -r requirements.txt

# Application 
app.py

# Request Optional ..(atleast need one parameter)
{
    "text_list": ["A dog."] ,
    "image_paths": ["https://storage.cloud.google.com/rfxstudio/images/dog_image.jpg"],
    "audio_paths": ["https://storage.cloud.google.com/rfxstudio/audios/dog_audio.wav"]
}

# Response
{
  "text": [[...]],
  "vision": [[...]],
  "audio":[[...]]
}