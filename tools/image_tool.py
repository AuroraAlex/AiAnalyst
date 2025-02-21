"""
Image processing tools for the analysis system.
"""

from PIL import Image
import io

class ImageTool:
    @staticmethod
    def load_image(image_path):
        """Load and preprocess an image."""
        try:
            return Image.open(image_path)
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")

    @staticmethod
    def resize_image(image, max_size=1024):
        """Resize image while maintaining aspect ratio."""
        ratio = min(max_size/image.width, max_size/image.height)
        if ratio < 1:
            return image.resize((int(image.width*ratio), int(image.height*ratio)))
        return image