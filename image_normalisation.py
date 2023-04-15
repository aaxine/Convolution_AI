from PIL import Image
import numpy as np

def preprocess_image(input_image, target_size=(224, 224)):
    # Resize the image
    resized_image = input_image.resize(target_size, Image.LANCZOS)
    
    # Convert pixel values to a normalized range (0-1)
    normalized_image = np.array(resized_image) / 255.0

    return normalized_image

uploaded_image = Image.open("./archive/testCat1.jpeg")
preprocessed_image = preprocess_image(uploaded_image)

print(preprocessed_image)
print(np.shape(preprocessed_image))