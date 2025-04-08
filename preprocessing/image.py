from PIL import Image
import requests
from io import BytesIO


def preprocess_and_check_image(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image = image.convert("RGB")
        return image
    except Exception as e:
        print(f"Failed to process image from {image_url}: {e}")
        return None
