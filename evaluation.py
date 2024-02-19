

"""
This module contains functions for evaluating the performance of a Variational Autoencoder (VAE) model trained on the Bach chorales dataset.
It includes functions for generating a synthetic dataset, calculating chord distances, transforming chroma to chord names, and visualizing the latent space using PCA.
"""

import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import colormaps
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.models import load_model
from src.models import Sampling, VAE
from src.utils import *

if __name__ == "__main__":
    # Define the size of the synthetic dataset
    size_dataset = 10000
    slices_train, test_slices, test_labels = load_and_enrich_dataset(size_dataset)

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
    vae = VAE(encoder, decoder)

    checkpoint_path = "training_1/cp.ckpt"
    vae.load_weights(checkpoint_path)

    # Get the encoder and decoder models from the VAE model
    encoder, decoder = vae.encoder, vae.decoder

    # Encode the test slices to obtain the latent space representation
    z_test = encoder.predict(test_slices, batch_size=128)

    # Perform PCA on the latent space representation to reduce dimensionality to 2
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(z_test[2])

    x,y = principalComponents[:,0], principalComponents[:,1]
    x,y = z_test[2][:,0], z_test[2][:,1]
    labels = test_labels

    label_coords = {}

    for i in range(len(x)):
        if labels[i] in label_coords:
            label_coords[labels[i]].append((x[i],y[i]))
        else:
            label_coords[labels[i]] = [(x[i],y[i])]

    label_centroids = {}
    for l in label_coords.keys():
        label_centroids[l] = centroid(label_coords[l])


    c_points = list(label_centroids.values())
    c_labels = list(label_centroids.keys())

    # Create a colormap for coloring the Voronoi regions
    cmap = colormaps['viridis']
    # Get the chords and points for the nearest neighbors calculation
    chords = c_labels
    points = np.array(c_points)

    # Define the number of nearest neighbors to consider
    K = 5

    # Fit the nearest neighbors model to the points
    nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)

    # Calculate the scores for each point based on chord distances
    scores = np.zeros(len(points))
    for k in range(len(points)):
        neighbors = indices[k][1:]
        score = 0
        c1 = chords[k]
        for i in neighbors:
            c2 = chords[i]
            d = chord_distance(c1, c2)
            score += d
        scores[k] = score/K

    scores = scores

    # Compute the Voronoi tessellation
    vor = Voronoi(points)

    # Create a plot for the Voronoi diagram
    fig, ax = plt.subplots()
    ax.set_title("Score : {:.2f}".format(sum(scores)))

    # Colorize the Voronoi regions based on the scores
    regions, vertices = voronoi_finite_polygons_2d(vor)
    fig.set_size_inches(18.5, 10.5)
    for i, region in enumerate(regions):
        polygon = vertices[region]
        c = cmap(scores[i])
        ax.fill(*zip(*polygon), color=c)

    # Plot the points and Voronoi diagram
    ax.plot(points[:,0], points[:,1], 'ko')
    voronoi_plot_2d(vor, ax, show_points=False,show_vertices=False)

    # Annotate each point with its chord and score
    for k in range(len(points)):
        x, y = points[k][0], points[k][1]
        ax.annotate(chords[k]+"\n"+str(truncate(scores[k],2)),(x,y))

    # Set the plot limits
    plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

    # Save the figure to the /figures folder
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig('figures/voronoi_plot.png')

    # Perform reconstruction on a randomly selected test slice
    test_index=random.randrange(0,len(slices_train))
    z_test = encoder.predict(slices_train[0:len(slices_train)], batch_size=128)

    output = decoder.predict(z_test[0])[test_index].squeeze(axis=2)

    # Create a figure for the reconstruction plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    axes[0].imshow(output,origin='lower',cmap="Greens")
    axes[1].imshow(slices_train[test_index].squeeze(axis=2),origin='lower',cmap="Reds")
    fig.tight_layout()

    # Save the figure to the /figures folder
    plt.savefig('figures/reconstruction_plot.png')

    # Create a .md report with metrics
    with open('report.md', 'w') as f:
        f.write("# Report\n\n")
        f.write("## Metrics\n\n")
        f.write("- Score: {:.2f}\n".format(sum(scores)))
        f.write("- Test Index: {}\n".format(test_index))
