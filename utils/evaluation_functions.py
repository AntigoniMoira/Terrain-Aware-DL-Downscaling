import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from skimage.metrics import structural_similarity

def measure_inference_time(model, x_test, batch_size=1):
    """
    Measures average inference time per sample in milliseconds.
    
    Parameters:
        model: Trained TensorFlow/Keras model.
        x_test: Test dataset (NumPy array or compatible input).
        batch_size: Batch size to use during inference.

    Returns:
        Average inference time per sample (in milliseconds).
    """
    # Warm-up run (optional but recommended to avoid first-run overhead)
    _ = model.predict(x_test[:batch_size], batch_size=batch_size)

    start_time = time.time()
    _ = model.predict(x_test, batch_size=batch_size)
    end_time = time.time()

    total_time = end_time - start_time  # total time in seconds
    avg_time_ms = (total_time / len(x_test)) * 1000  # convert to milliseconds

    return avg_time_ms


def evaluate_per_pixel(model, low_res_data, high_res_data, max_pixel=1.0):
    """
    Evaluates the SRCNN model per pixel with TensorFlow-based PSNR calculation.

    Parameters:
    - model: Trained SRCNN model.
    - low_res_data: Low-resolution input images (batch, height, width, channels).
    - high_res_data: Ground truth high-resolution images (batch, height, width, channels).
    - max_pixel: Maximum pixel value (1.0 for normalized images, 255 for unnormalized).

    Returns:
    - per_pixel_mse: MSE per pixel.
    - per_pixel_mae: MAE per pixel.
    - per_pixel_psnr: PSNR per pixel using TensorFlow computation.
    - per_pixel_ssim: SSIM per pixel.
    """
    # Predict high-resolution images
    pred_high_res = model.predict(low_res_data)
    
    high_res_data = high_res_data.astype(np.float32)

    # Compute per-pixel MSE and MAE
    per_pixel_mse = np.mean((pred_high_res - high_res_data) ** 2, axis=0)  # (H, W, C)
    per_pixel_mae = np.mean(np.abs(pred_high_res - high_res_data), axis=0)  # (H, W, C)

    # TensorFlow-based PSNR calculation per pixel
    per_pixel_psnr = np.zeros_like(per_pixel_mse)
    for i in range(per_pixel_mse.shape[0]):  # Loop over height
        for j in range(per_pixel_mse.shape[1]):  # Loop over width
            mse_pixel = per_pixel_mse[i, j]
            if mse_pixel > 0:
                per_pixel_psnr[i, j] = 10.0 * tf.math.log(max_pixel ** 2 / mse_pixel) / tf.math.log(10.0)
            else:
                per_pixel_psnr[i, j] = np.inf  # Perfect match

    # Compute SSIM per pixel (on grayscale channel)
    per_pixel_ssim = np.zeros_like(per_pixel_mse)
    for i in range(per_pixel_mse.shape[0]):  # Loop over height
        for j in range(per_pixel_mse.shape[1]):  # Loop over width
            try:
                per_pixel_ssim[i, j] = structural_similarity(
                    pred_high_res[:, i, j, 0], high_res_data[:, i, j, 0], data_range=max_pixel
                )
            except:
                per_pixel_ssim[i, j] = 0  # Handle edge cases

    return per_pixel_mse, per_pixel_mae, per_pixel_psnr, per_pixel_ssim

def baseline_evaluate_per_pixel(pred_high_res, high_res_data, max_pixel=1.0):
    """
    Evaluates the SRCNN model per pixel with TensorFlow-based PSNR calculation.

    Parameters:
    - model: Trained SRCNN model.
    - low_res_data: Low-resolution input images (batch, height, width, channels).
    - high_res_data: Ground truth high-resolution images (batch, height, width, channels).
    - max_pixel: Maximum pixel value (1.0 for normalized images, 255 for unnormalized).

    Returns:
    - per_pixel_mse: MSE per pixel.
    - per_pixel_mae: MAE per pixel.
    - per_pixel_psnr: PSNR per pixel using TensorFlow computation.
    - per_pixel_ssim: SSIM per pixel.
    """

    # Compute per-pixel MSE and MAE
    per_pixel_mse = np.mean((pred_high_res - high_res_data) ** 2, axis=0)  # (H, W, C)
    per_pixel_mae = np.mean(np.abs(pred_high_res - high_res_data), axis=0)  # (H, W, C)

    # TensorFlow-based PSNR calculation per pixel
    per_pixel_psnr = np.zeros_like(per_pixel_mse)
    for i in range(per_pixel_mse.shape[0]):  # Loop over height
        for j in range(per_pixel_mse.shape[1]):  # Loop over width
            mse_pixel = per_pixel_mse[i, j]
            if mse_pixel > 0:
                per_pixel_psnr[i, j] = 10.0 * tf.math.log(max_pixel ** 2 / tf.cast(mse_pixel, tf.float32)) / tf.math.log(10.0) 
            else:
                per_pixel_psnr[i, j] = np.inf  # Perfect match
                # per_pixel_psnr[i, j] = 100

    # Compute SSIM per pixel (on grayscale channel)
    per_pixel_ssim = np.zeros_like(per_pixel_mse)
    for i in range(per_pixel_mse.shape[0]):  # Loop over height
        for j in range(per_pixel_mse.shape[1]):  # Loop over width
            try:
                per_pixel_ssim[i, j] = structural_similarity(
                    pred_high_res[:, i, j, 0], high_res_data[:, i, j, 0], data_range=max_pixel
                )
            except:
                per_pixel_ssim[i, j] = 0  # Handle edge cases

    return per_pixel_mse, per_pixel_mae, per_pixel_psnr, per_pixel_ssim

def overall_evaluate_per_pixel(model_1, model_2, low_res_data, high_res_data, regional=False, max_pixel=1.0):
    medium_res_output = model_1.predict(low_res_data)
    pred_high_res = model_2.predict(medium_res_output)

    if regional:
        pred_high_res = pred_high_res[:, 140:180, 310:350, :]
        high_res_data = high_res_data[:, 140:180, 310:350, :]

    # Compute per-pixel MSE and MAE
    per_pixel_mse = np.mean((pred_high_res - high_res_data) ** 2, axis=0)  # (H, W, C)
    per_pixel_mae = np.mean(np.abs(pred_high_res - high_res_data), axis=0)  # (H, W, C)

    # TensorFlow-based PSNR calculation per pixel
    per_pixel_psnr = np.zeros_like(per_pixel_mse)
    for i in range(per_pixel_mse.shape[0]):  # Loop over height
        for j in range(per_pixel_mse.shape[1]):  # Loop over width
            mse_pixel = per_pixel_mse[i, j]
            if mse_pixel > 0:
                per_pixel_psnr[i, j] = 10.0 * tf.math.log(max_pixel ** 2 / mse_pixel) / tf.math.log(10.0)
            else:
                per_pixel_psnr[i, j] = np.inf  # Perfect match

    # Compute SSIM per pixel (on grayscale channel)
    per_pixel_ssim = np.zeros_like(per_pixel_mse)
    for i in range(per_pixel_mse.shape[0]):  # Loop over height
        for j in range(per_pixel_mse.shape[1]):  # Loop over width
            try:
                per_pixel_ssim[i, j] = structural_similarity(
                    pred_high_res[:, i, j, 0], high_res_data[:, i, j, 0], data_range=max_pixel
                )
            except:
                per_pixel_ssim[i, j] = 0  # Handle edge cases

    return per_pixel_mse, per_pixel_mae, per_pixel_psnr, per_pixel_ssim