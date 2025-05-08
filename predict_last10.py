import os

import numpy as np
from skimage import io
from tensorflow.keras.models import load_model

from data import saveResult, testGenerator

# Load the trained model
model = load_model("unet_membrane2.keras")
print("Model loaded successfully!")

# Get test data from the generator - using last 10 training images
testGene = testGenerator("data/membrane/train_last10/image")
# For modern Keras, predict one image at a time
num_test_images = 10  # Using last 10 training images
results = []
print("Making predictions on last 10 training images...")
for i in range(num_test_images):
    img = next(testGene)
    result = model.predict(img, verbose=0)
    results.append(result[0])
    print(f"Processed image {i+1}/{num_test_images}")

# Save results in the specified directory
results_path = "data/membrane/train_last10/results"
print("Saving results...")
saveResult(results_path, np.array(results))
print(f"Prediction complete! Results saved in {results_path}")
