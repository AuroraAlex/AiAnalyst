"""
Image processing tools for the analysis system.
"""

from PIL import Image
import io
import base64
from pathlib import Path

class ImageTool:
    @staticmethod
    def load_image(image_path):
        """Load and preprocess an image."""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            return Image.open(image_path)
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")

    @staticmethod
    def resize_image(image, max_size=1024):
        """Resize image while maintaining aspect ratio."""
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image object")
            
        # 如果图片尺寸已经小于最大尺寸，则不需要调整
        if max(image.width, image.height) <= max_size:
            return image
            
        ratio = min(max_size/image.width, max_size/image.height)
        new_size = (int(image.width*ratio), int(image.height*ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    @staticmethod
    def convert_to_base64(image_path):
        """Convert image to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
        except Exception as e:
            raise Exception(f"Error converting image to base64: {str(e)}")

    @staticmethod
    def save_image(image, output_path):
        """Save processed image to disk."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            return str(output_path)
        except Exception as e:
            raise Exception(f"Error saving image: {str(e)}")