import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.models import VAE, Sampling
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import argparse, sys
from evaluation import plot_cluster, load_and_enrich_dataset

# Example call: python train.py --epochs 100 --coldstart 30 --beta 0.2 --batchsize 128

parser=argparse.ArgumentParser()

parser.add_argument('--epochs', help='Number of epochs for training')
parser.add_argument('--coldstart', help='Number of epochs during beta is zero')
parser.add_argument('--beta', help='Max beta value')
parser.add_argument('--batchsize', help='Batch size')

args=parser.parse_args()

EPOCHS = int(args.epochs)
COLDSTART = int(args.coldstart)
BETA = float(args.beta)
BATCH_SIZE = int(args.batchsize)


slices_train = np.load("./slices_train.npy")

if __name__ == '__main__':
  latent_dim = 2
  encoder_inputs = keras.Input(shape=(12, 1, 1))
  conv1_filters = 32
  conv2_filters = 64
  conv_kernel_size = 3
  conv_strides = 2
  conv_padding = "same"
  dense_units = 16

  x = layers.Conv2D(conv1_filters, conv_kernel_size, activation="relu", strides=conv_strides, padding=conv_padding)(encoder_inputs)
  x = layers.Conv2D(conv2_filters, conv_kernel_size, activation="relu", strides=conv_strides, padding=conv_padding)(x)
  x = layers.Flatten()(x)
  x = layers.Dense(dense_units, activation="relu")(x)
  z_mean = layers.Dense(latent_dim, name="z_mean")(x)
  z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
  z = Sampling()([z_mean, z_log_var])
  encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
  encoder.summary()

  latent_inputs = keras.Input(shape=(latent_dim,))
  dense_units_decoder = 12 * 1 * 64
  conv_transpose1_filters = 64
  conv_transpose2_filters = 32
  conv_transpose_kernel_size = 1
  conv_transpose_strides = 1
  conv_transpose_padding = "same"
  conv_transpose_output_filters = 1

  x = layers.Dense(dense_units_decoder, activation="relu")(latent_inputs)
  x = layers.Reshape((12, 1, 64))(x)
  x = layers.Conv2DTranspose(conv_transpose1_filters, conv_transpose_kernel_size, activation="relu", strides=conv_transpose_strides, padding=conv_transpose_padding)(x)
  x = layers.Conv2DTranspose(conv_transpose2_filters, conv_transpose_kernel_size, activation="relu", strides=conv_transpose_strides, padding=conv_transpose_padding)(x)
  decoder_outputs = layers.Conv2DTranspose(conv_transpose_output_filters, conv_kernel_size, activation="sigmoid", padding=conv_padding)(x)
  decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
  decoder.summary()

  epochs = EPOCHS
  beta = np.linspace(1e-3, BETA, epochs)
  hist = []
  vae = VAE(encoder, decoder)
  vae.beta_update(max(beta[0], 0))
  cold_start = COLDSTART
  for epoch in range(epochs):
    print("Epoch: " + str(epoch) + "/" + str(epochs) + ", beta = " + str(vae.beta))
    if epoch > cold_start:
      vae.beta_update(max(beta[epoch - cold_start], 0))
      #TO FIX
    """    if epoch%5==0: 
      _, test_slices, test_labels = load_and_enrich_dataset(1000) # Load a synthetic dataset used for plotting the latent space.
      plot_cluster(f"Latent dimension: {latent_dim}", "latent", vae, epoch, test_slices, BATCH_SIZE, test_labels)"""

  # Create a callback that saves the model's weights
  checkpoint_path = "training_1/cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
  vae.compile(optimizer=keras.optimizers.Adam())
  hist.append(vae.fit(slices_train, epochs=epochs, batch_size=BATCH_SIZE, callbacks=[cp_callback]))

  history = []
  name = str(time.time())
  for h in hist:
    history.append(h.history['loss'])
    history.append(h.history['reconstruction_loss'])
    history.append(h.history['kl_loss'])
  df = pd.DataFrame(np.array(history).T, columns=['loss', 'reconstruction_loss', 'kl_loss'])
  # Create the figures folder if it doesn't exist
  os.makedirs("figures", exist_ok=True)
  df.plot()
  plt.savefig("figures/loss.png")
