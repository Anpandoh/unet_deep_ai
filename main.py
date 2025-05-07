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
    2, "data/membrane/train", "image", "label", data_gen_args, save_to_dir=None
)

model = unet()
model_checkpoint = ModelCheckpoint(
    "unet_membrane2.keras", monitor="loss", verbose=1, save_best_only=True
)
model.fit(myGene, steps_per_epoch=2000, epochs=5, callbacks=[model_checkpoint])

# For evaluation, we'll just load the best model and run prediction in the separate predict.py script
print("Training complete! Use predict.py to generate segmentation results.")

# Get test data from the generator
testGene = testGenerator("data/membrane/test")
# For modern Keras, predict one image at a time
num_test_images = 30  # Assuming there are 30 test images as in the original code
results = []
for i in range(num_test_images):
    img = next(testGene)
    result = model.predict(img, verbose=0)
    results.append(result[0])

saveResult("data/membrane/test", np.array(results))
