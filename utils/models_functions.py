import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

# Custom PSNR Metric
def psnr(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * tf.math.log(max_pixel ** 2 / (K.mean(K.square(y_pred - y_true)))) / tf.math.log(10.0)

# Custom SSIM Metric
def ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)
    
def save_history(history, file_path):
    """
    Save the training history object to a file.

    Args:
    history (dict): The training history dictionary.
    file_path (str): Path to the file where the history will be saved.

    """
    with open(file_path, 'wb') as f:
        pickle.dump(history, f)

    
def save_model(model, file_path):
    """
    Saves the given model to the specified file path.

    Args:
    model (tensorflow.keras.Model): The Keras model to be saved.
    file_path (str): The path where the model will be saved.
    """
    model.save(file_path)

def load_model(file_path):
    """
    Loads a Keras model from the specified file path.

    Args:
    file_path (str): The path from where the model will be loaded.

    Returns:
    tensorflow.keras.Model: The loaded Keras model.
    """
    return tf.keras.models.load_model(file_path, custom_objects={
    'psnr': psnr,
    'ssim': ssim
    })

def train_model(model, trainX, trainY, valX, valY, epochs=50, batch_size=16, early_stopping=False, reduce_lr=False, early_stoping_patience=30, reduce_lr_patience = 30):
    """
    Trains the given model using the provided training and validation data.

    Args:
        model (tensorflow.keras.Model): The Keras model to be trained.
        trainX (numpy.ndarray): Training data features (low-resolution inputs).
        trainY (numpy.ndarray): Training data labels (high-resolution outputs).
        valX (numpy.ndarray): Validation data features (low-resolution inputs).
        valY (numpy.ndarray): Validation data labels (high-resolution outputs).
        epochs (int, optional): The number of epochs to train the model. Defaults to 50.
        batch_size (int, optional): The batch size to use during training. Defaults to 16.
        patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 5.
        early_stopping (bool, optional): Whether to use early stopping. Defaults to True.
        reduce_lr (bool, optional): Whether to reduce the learning rate on plateau. Defaults to True.

    Returns:
        tensorflow.keras.callbacks.History: The history object that holds training and validation loss values.
    """
    
    callbacks = []
    if early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=early_stoping_patience, restore_best_weights=False, min_delta=1e-4, verbose=1))
    
    if reduce_lr:
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=reduce_lr_patience, min_lr=1e-5, min_delta=1e-4, verbose=1))

    history = model.fit(trainX, trainY,
                        validation_data=(valX, valY),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks)

    return history
