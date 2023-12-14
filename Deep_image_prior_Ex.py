# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 18:00:48 2022

@author: Morteza
"""

import os
import numpy as np
from skimage.transform import rescale, resize

import tensorflow as tf
from tqdm.keras import TqdmCallback
print(tf.__version__)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def get_noisy_img(img, sig=30):
    """Task 1: Removing white noise"""
    sigma = sig / 255.
    noise = np.random.normal(scale=sigma, size=img.shape)
    img_noisy = np.clip(img + noise, 0, 1).astype(np.float32)
    return img_noisy

from skimage import data
from matplotlib import pyplot as plt 

img = data.chelsea().astype(np.float32) / 255.

_, axis = plt.subplots(1, 2, figsize=(16, 5))
axis[0].imshow(img); axis[0].set_title("Input $x_0$")
axis[1].imshow(get_noisy_img(img)); axis[1].set_title("Denoising task")
#axis[2].imshow(get_inpainted_img(img)[0]); axis[2].set_title("Inpainting task")
for ax in axis:
    ax.set_axis_off()
plt.show()


def dip_workflow(x0,
                 x_true, 
                 f, 
                 f_input_shape, 
                 z_std=0.1,
                 loss_mask=None,
                 num_iters=1,
                 init_lr=0.01,
                 save_filepath=None):
    """Deep Image prior workflow
    Args:
        * x0: input image
        * x_true: Ground-truth image, only used for metrics comparison
        * f: Neural network to use as a prior
        * f_input_shape: Shape (excluding batch size) of inputs to f
        * loss_mask: if not None, a binary mask with the same shape as x0,
            which is applied to both x and x0 before applying the loss.
            Used for instance in the inpainting task.
        * num_iters: Number of training iterations
        * init_lr: Initial learning rate for Adam optimizer
        * If True, will save the best model in the given filepath
    """
    # Sample input z
    shape = (1,) + f_input_shape
    z = tf.constant(np.random.uniform(size=shape).astype(np.float32) * z_std, name='net_input')

    # Training Loss
    def loss_fn(x_true, x):
        del x_true
        nonlocal x0, loss_mask
        if loss_mask is None:
            return tf.keras.losses.MSE(x, x0)
        else:
            return tf.keras.losses.MSE(x * loss_mask, x0 * loss_mask)
        
    # Output/log information
    # Diff between generated image and true ground-truth
    # as mean squared error and psnr (peak signal to noise ratio)
    def mse_to_gt(x_true, x):
        return tf.reduce_mean(tf.losses.mean_squared_error(x, x_true))
    
    def psnr_to_gt(x_true, x, maxv=1.):
        mse = tf.reduce_mean(tf.losses.mean_squared_error(x, x_true))
        psnr_ = 10. * tf.math.log(maxv** 2 /mse) / tf.math.log(10.)
        return psnr_
    
    # Optimization
    opt = tf.keras.optimizers.Adam(learning_rate=init_lr)
    f.compile(optimizer=opt, loss=loss_fn, metrics=[mse_to_gt, psnr_to_gt])
    # Saving best model
    callbacks = ()
    if save_filepath is not None:
        callbacks = create_saving_callback(save_filepath)
    
    # Training
    history = f.fit(z, 
                    x_true[None, ...], 
                    epochs=num_iters,
                    steps_per_epoch=1, 
                    verbose=0, 
                    callbacks=callbacks+(TqdmCallback(verbose=1),))
    
    # Display results with gridspec
    x = f.predict(z)[0]
    fig = plt.figure(figsize=(10, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    axes = [fig.add_subplot(gs[0, :]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, 0]),
            fig.add_subplot(gs[2, 1])]
    for ax in axes[1:]:
        ax.set_axis_off()
        
    for key in history.history.keys():
        axes[0].plot(range(num_iters), history.history[key], label=key)
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].set_title("Training dynamics")
    axes[1].imshow(x0); axes[1].set_title('Input image')
    axes[2].imshow(x_true); axes[2].set_title('Ground-truth')
    axes[3].imshow(x); axes[3].set_title(f'Last output (PSNR = {psnr_to_gt(x_true, x):.2f})')
    if save_filepath is not None and os.path.exists(save_filepath):
        f.load_weights(save_filepath)
        x_opt = f.predict(z)[0]
        axes[4].imshow(x_opt); axes[4].set_axis_off()
        axes[4].set_title(f'Best model output (PSNR = {psnr_to_gt(x_true, x):.2f})')
    plt.show()
    return x

class GaussianNoiseWithDecay(tf.keras.layers.GaussianNoise):
    
    def __init__(self, stddev, decayrate=0.99999, decaysteps=1, **kwargs):
        super(GaussianNoiseWithDecay, self).__init__(stddev, **kwargs)
        self.num_calls = 0
        self.decayrate = decayrate
        self.decaysteps = decaysteps
        
        
    def call(self, inputs, training=None):
        def noised():
            self.num_calls += 1
            stddev = self.stddev * self.decayrate ** (self.num_calls // self.decaysteps)
            return inputs + tf.keras.backend.random_normal(
                shape=tf.shape(inputs),
                mean=0.,
                stddev=stddev,
                dtype=inputs.dtype)

        return tf.keras.backend.in_train_phase(noised, inputs, training=training)


def create_saving_callback(filepath):
    return (tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath, 
        monitor='loss',
        verbose=0, 
        save_best_only=True,
        mode='min'),)

def deep_image_prior(input_shape,
                     noise_reg=None,
                     layers=(128, 128, 128, 128, 128),
                     kernel_size_down=3,
                     kernel_size_up=3,
                     skip=(0, 4, 4, 4, 4)):
    def norm_and_active(x):
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        return x
    
    model = tf.keras.models.Sequential(name="Deep Image Prior")
    inputs = tf.keras.Input(shape=input_shape)
    
    ## Inputs
    x = inputs
    if noise_reg is not None:
        x = GaussianNoiseWithDecay(**noise_reg)(x)
    
    ## Downsampling layers
    down_layers = []
    for i, (num_filters, do_skip) in enumerate(zip(layers, skip)):
        if do_skip > 0:
            down_layers.append(norm_and_active(tf.keras.layers.Conv2D(
                filters=do_skip, kernel_size=1, strides=1, name=f"conv_skip_depth_{i}")(x)))
        for j, strides in enumerate([2, 1]):
            x = norm_and_active(tf.keras.layers.Conv2D(
                num_filters, kernel_size_down, strides=strides, padding='same',
                name=f"conv_down_{j + 1}_depth_{i}")(x))
        
    ## Upsampling
    for i, (num_filters, do_skip) in enumerate(zip(layers[::-1], skip[::-1])):
        x = tf.keras.layers.UpSampling2D(interpolation='bilinear', name=f"upsample_depth_{i}")(x)
        if do_skip:
            x = tf.keras.layers.Concatenate(axis=-1)([x, down_layers.pop()])
        for j, kernel_size in enumerate([kernel_size_up, 1]):
            x = norm_and_active(tf.keras.layers.Conv2D(
                num_filters, kernel_size, strides=1, padding='same',
                name=f"conv_up_{j + 1}_depth_{i}")(x))
            
    ## Last conv
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1, name="conv_out")(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name="deep_image_prior")


def display_dip_model(input_shape=(256, 256, 3)):
    model = deep_image_prior(input_shape)
    model.build(input_shape)
    print(model.summary())
display_dip_model()


x_true = resize(data.chelsea().astype(np.float32) / 255., (256, 256))
x0 = get_noisy_img(x_true)
_, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(x0); axes[0].set_axis_off()
axes[0].set_title("Input noisy image")
axes[1].imshow(x_true); axes[1].set_axis_off()
axes[1].set_title("Ground-truth image")
plt.show()

input_shape = x0.shape
noise_reg = {'stddev': 1./ 30., 'decayrate': 1.0, 'decaysteps': 100}

model = deep_image_prior(input_shape, noise_reg=noise_reg)
x = dip_workflow(x0, x_true, model, input_shape, num_iters=3000, save_filepath='best_dip_denoising.hdf5')