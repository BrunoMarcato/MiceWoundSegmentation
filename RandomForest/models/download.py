from keras.applications import VGG16
import pickle

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

VGG16_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))

pickle.dump(VGG16_model, open(f'RandomForest/models/vgg16_{IMAGE_HEIGHT}x{IMAGE_WIDTH}.sav', 'wb'))