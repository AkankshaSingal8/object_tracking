import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

DEFAULT_NCP_SEED = 22222

def load_dataset_from_directory(directory, image_shape, batch_size, seed):
    # Load dataset
    dataset = image_dataset_from_directory(
        directory,
        shuffle=True,
        batch_size=batch_size,
        image_size=image_shape[:2],
        seed=seed
    )

    # Normalization and other preprocessing steps can be added here if needed
    # For example, dataset = dataset.map(lambda x, y: (preprocess_function(x), y))

    return dataset

# Load your dataset
train_dataset = load_dataset_from_directory('./Test1', IMAGE_SHAPE, 32, DEFAULT_NCP_SEED)
print(train_dataset)

# Example of how to use the dataset in model training
# mymodel.fit(train_dataset, epochs=num_epochs)
