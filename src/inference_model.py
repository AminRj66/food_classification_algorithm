import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers
from sklearn.metrics import confusion_matrix
from .utils import load_params


class InferenceModel:
    def __init__(self, config_dir: str, n_classes: int):
        # loading config parameters
        params = load_params(os.path.join(config_dir, "config.yml"))

        self.model = None
        self.n_classes = n_classes
        self.save_path = params['saved_models_dir']
        self.training = params['phase'] == "train"
        self.model_name = params['model_name']
        self.input_shape = tuple(params['target_size'] + [3])
        self.batch_size = params['batch_size']
        self.lr = params['learning_rate']
        self.epochs = params['epochs']

    def __create_model(self):
        # defining architecture for training phase and also loading pretrained model for testing phase
        if self.training:
            # defining a CNN using the MobileNetV2 architecture pre-trained on the ImageNet dataset
            pretrained_model = tf.keras.applications.MobileNetV2(
                input_shape=list(self.input_shape),
                include_top=False,
                # weights='imagenet',
                weights="src/mobilenet_v2_weights.h5",
                pooling='avg')

            pretrained_model.trainable = False

            # customizing the inference part of the model
            model = tf.keras.models.Sequential([
                # layers.Input(shape=self.input_shape),
                # layers.RandomRotation(0.125),
                pretrained_model,
                # tf.keras.layers.GlobalAveragePooling2D(),
                # layers.Dropout(0.25),
                # layers.Dense(256, activation='relu'),
                # layers.BatchNormalization(),
                layers.Dropout(0.1),
                layers.Dense(64, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),),
                layers.BatchNormalization(),
                layers.Dropout(0.1),
                layers.Dense(self.n_classes, activation='softmax')])

            # configuring the learning process using Adam optimizer
            model.compile(optimizer=Adam(learning_rate=self.lr),
                          loss=tf.keras.losses.CategoricalCrossentropy(),
                          metrics=['AUC'])
            # model.build()
            print(model.summary())

        else:
            model = load_model(self.save_path + "/" + self.model_name, compile=False)
            model.compile(optimizer=Adam(learning_rate=self.lr),
                          loss=tf.keras.losses.CategoricalCrossentropy(),
                          metrics=['AUC'])

        return model

    def train_model(self, train_images: pd.DataFrame, val_images: pd.DataFrame):
        self.model = self.__create_model()
        # training the model with a predefined batch size
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, restore_best_weights=True)
        history_data = self.model.fit(train_images,
                                      steps_per_epoch=train_images.samples // self.batch_size // 2,
                                      epochs=self.epochs,
                                      validation_data=val_images,
                                      validation_steps=val_images.samples // self.batch_size // 2,
                                      verbose=1,
                                      callbacks=[early_stop],
                                      shuffle=True)

        self.model.save(self.save_path + "/model_" + time.time().__str__().split(".")[0] + ".hdf5")

        return history_data.history

    def test_model(self, test_images: pd.DataFrame):
        # testing a pre-saved model
        self.model = self.__create_model()
        loss, auc = self.model.evaluate(test_images)
        return loss, auc

    @staticmethod
    def plot_training_performance(epochs, data, train_param, val_param):
        plt.figure(figsize=(10, 7))

        plt.plot(epochs, data[train_param], 'g', label=f'Training ({train_param})')
        plt.plot(epochs, data[val_param], 'red', label=f'Validation ({val_param})')

        plt.title("Training performance")
        plt.xlabel('Epochs')
        plt.ylabel(train_param)

        plt.legend()
        plt.show()

    @staticmethod
    def plot_confusion(model, test_images, n_classes):
        predictions = np.argmax(model.predict(test_images), axis=1)
        cm = confusion_matrix(test_images.labels, predictions)
        plt.figure(figsize=(30, 30))
        sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
        plt.xticks(ticks=np.arange(n_classes) + 0.5, labels=test_images.class_indices, rotation=90)
        plt.yticks(ticks=np.arange(n_classes) + 0.5, labels=test_images.class_indices, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
