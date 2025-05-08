import numpy as np

from data import *
from model import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest",
)
myGene = trainGenerator(
    2, "data/membrane/train_20", "image", "label", data_gen_args, save_to_dir=None
)

model = unet()
model_checkpoint = ModelCheckpoint(
    "unet_membrane2.keras", monitor="loss", verbose=1, save_best_only=True
)
model.fit(myGene, steps_per_epoch=2000, epochs=5, callbacks=[model_checkpoint])

# For evaluation, we'll just load the best model and run prediction in the separate predict.py script
print("Training complete! Use predict.py to generate segmentation results.")

# Get test data from the generator - using last 10 training images
testGene = testGenerator("data/membrane/train_last10")
# For modern Keras, predict one image at a time
num_test_images = 10  # Using last 10 training images
results = []
for i in range(num_test_images):
    img = next(testGene)
    result = model.predict(img, verbose=0)
    results.append(result[0])

# Save results in a new directory to keep them separate
saveResult("data/membrane/train_last10_results", np.array(results))
