import numpy as np
from tensorflow.keras.models import load_model

from data import saveResult, testGenerator

# Load the trained model
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
    results.append(result[0])
    print(f"Processed image {i+1}/{num_test_images}")

# Save results
print("Saving results...")
saveResult(test_path, np.array(results))
print("Prediction complete! Results saved in", test_path)
