import os
import numpy as np
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.utils import to_categorical
import cv2

def load_data(data_dir, img_size=(48, 48)):
    X, y = [], []
    class_names = os.listdir(data_dir)
    class_map = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    for cls in class_names:
        cls_path = os.path.join(data_dir, cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Read in RGB
            if img is not None:
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(class_map[cls])

    X = np.array(X, dtype="float32") / 255.0  # Normalize to [0, 1]
    y = to_categorical(y, num_classes=len(class_names))
    return np.array(X), np.array(y), class_map


def get_data_generator_rgb(train_dir, test_dir, img_size=(48, 48), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        shear_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        subset='training',
        shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        subset='validation',
        shuffle=False
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )

    return train_gen, val_gen, test_gen