import time
import tensorflow as tf
from src import load_params, set_device_option, InferenceModel, PrepareData

if __name__ == "__main__":
    start_time = time.time()

    # load parameters
    params = load_params("config/config.yml")

    # set device option
    set_device_option(params['device'])

    with tf.device(params['device']):
        # initiate objects
        data_object = PrepareData(config_dir="config")
        model_object = InferenceModel(config_dir="config", n_classes=data_object.n_classes)

        # begin experiment
        if params['phase'] == "train":
            train_images, val_images = data_object.create_train_val_data()
            history = model_object.train_model(train_images, val_images)

            epochs = range(1, len(history['auc']) + 1)
            model_object.plot_training_performance(epochs, history, 'auc', 'val_auc')
            model_object.plot_training_performance(epochs, history, 'loss', 'val_loss')
        else:
            test_images = data_object.create_test_data()
            loss, auc = model_object.test_model(test_images)
            print("Accuracy of classification is {}%".format(auc*100))
            model_object.plot_confusion(model_object.model, test_images, data_object.n_classes)

        print("--- %s seconds ---" % (time.time() - start_time))
