import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models import VAE, Sampling
import pandas as pd
import time
#TODO: save model
#python train.py --epochs 100 --coldstart 30 --beta 0.2 --batchsize 128

import argparse, sys

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
   x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
   x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
   x = layers.Flatten()(x)
   x = layers.Dense(16, activation="relu")(x)
   z_mean = layers.Dense(latent_dim, name="z_mean")(x)
   z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
   z = Sampling()([z_mean, z_log_var])
   encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
   encoder.summary()

   latent_inputs = keras.Input(shape=(latent_dim,))
   x = layers.Dense(12 * 1 * 64, activation="relu")(latent_inputs)
   x = layers.Reshape((12, 1, 64))(x)
   x = layers.Conv2DTranspose(64, 1, activation="relu", strides=1, padding="same")(x)
   x = layers.Conv2DTranspose(32, 1, activation="relu", strides=1, padding="same")(x)
   decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
   decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
   decoder.summary()

   epochs = EPOCHS
   beta = np.linspace(0,BETA,epochs)
   hist = []
   vae = VAE(encoder, decoder)
   vae.beta_update(max(beta[0],0))
   cold_start = COLDSTART
   for epoch in range(epochs):
     print("Epoch: "+str(epoch)+"/"+str(epochs)+", beta = "+str(vae.beta))
     if epoch > cold_start:
       vae.beta_update(max(beta[epoch-cold_start],0))

     vae.compile(optimizer=keras.optimizers.Adam())
     hist.append(vae.fit(slices_train, epochs=1, batch_size=BATCH_SIZE))

   history = []
   name = str(time.time())
   #vae.save_weights('models/'+name+"/", save_format='tf')
   for h in hist:
      history.append([100 * h.history['loss'][0], 100 * h.history['reconstruction_loss'][0], h.history['kl_loss'][0]])
   pd.DataFrame(history, columns=['loss', 'reconstruction_loss', 'kl_loss']).plot()