import tensorflow_datasets as tfds

# TODO: Load the dataset with TensorFlow Datasets. Hint: use tfds.load()
data_set, dataset_info = tfds.load('oxford_flowers102', as_supervised=True, with_info=True)