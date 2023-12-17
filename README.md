# Dimensionality Reduction Algorithms

This repository contains programs implementing dimensionality reduction algorithms: Variational Autoencoder (VAE), Principal Component Analysis (PCA), Uniform Manifold Approximation and Projection (UMAP), t-distributed Stochastic Neighbor Embedding (t-SNE), and Independent Component Analysis (ICA).

## Usage and Data Loading

ipynb notebooks can be run in order to reproduce the results. To load data needed, unzip the data.zip file. There should be a file final_repo/data/data.csv 
that is used for the code in jupyter. 

### Main.py

Algorithms we ran in a .py file for reference. 

### Main.ipynb

The Main file runs the 5 algorithms on our data and plots the first two components of each algorithm as a matplotlib dot plot. 
Data is loaded in as a gene expression transcriptome with rows as samples and columns as genes

### Time.ipynb

Times the performance of each algorithm on the data with the system package and graphs it

### Metric.ipynb

There are two parts of the metric.ipynb:

1. NMI, SIL, and ARI metrics for the algorithms after performing k-means clustering with two centroids
2. Hyperparameter sensitivity testing and accuracy for t-SNE, UMAP, and VAE