import pytesseract
from PIL import Image

# Path to a sample image file (put an image in your project folder or use an existing one)
image_path = "sample image.png"

# Extract text from image using pytesseract
text = pytesseract.image_to_string(Image.open(image_path))

print("Extracted text from image:")
print(text)

