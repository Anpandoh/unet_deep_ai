import os

import numpy as np
from keras.models import load_model

from data import saveResult, testGenerator

# Load the saved model
model = load_model("unet_membrane.keras")
print("Model loaded successfully!")

# Get test data
test_path = "data/membrane/test"
testGene = testGenerator(test_path)

# Predict on test images
num_test_images = 30  # Adjust this to match your test dataset size
results = []
print("Predicting on test images...")
for i in range(num_test_images):
    img = next(testGene)
    result = model.predict(img, verbose=0)

    # Keep the raw probability outputs - these show membrane structures as grayscale
    # No thresholding, just use the raw network output
    pred = result[0]

    results.append(pred)
    print(f"Processed image {i+1}/{num_test_images}")

# Save results
print("Saving results...")
saveResult(test_path, np.array(results))
print("Prediction complete! Results saved in", test_path)
