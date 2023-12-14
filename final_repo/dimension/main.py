import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
import umap

def filtered_gene_types():
  data = pd.read_csv('../data/data.csv')
  gene_types = data['gene_type'].unique()

  selected_gene_types = []
  for gene_type in gene_types:
    df = data[data['gene_type'] == gene_type]
    #filter out gene types with low sample count 
    if len(df) >= 1000:
      selected_gene_types.append(gene_type)
  #miRNA gene type has low expression counts
  selected_gene_types.remove('miRNA')
  return selected_gene_types

def expression_data():
   data = pd.read_csv('../data/data.csv')
   data = data.T
   data = data.drop(data.index[:3])
   return data
   
def plt_PCA(data):
  '''PCA'''
  # Standardize the gene expression data (mean=0 and variance=1)
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(data)

  # Apply PCA
  pca = PCA(n_components=2)  # Number of components for visualization (2 for 2D plot)
  pca_result = pca.fit_transform(scaled_data)

  # Create a DataFrame from PCA result for easy plotting
  pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])  # Change column names accordingly

  # Plot PCA results
  plt.figure(figsize=(8, 6))
  plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.title('PCA Plot of Gene Expression Data')
  plt.grid(True)
  plt.ylim(-25,25)
  plt.xlim(-25,25)
  plt.show()

def plt_tsne(data):
  '''TSNE'''
  selected_columns = data 

  tsne = TSNE(n_components=2, perplexity=30)  # Adjust parameters as needed

  # Perform t-SNE
  tsne_result = tsne.fit_transform(selected_columns)

  # Convert t-SNE result to a DataFrame for visualization
  tsne_df = pd.DataFrame(tsne_result, columns=['tsne_1', 'tsne_2'])

  # Plot the t-SNE result
  plt.figure(figsize=(8, 6))
  plt.scatter(tsne_df['tsne_1'], tsne_df['tsne_2'], alpha=0.5)
  plt.title('t-SNE Visualization of scRNA-seq Data')
  plt.xlabel('t-SNE 1')
  plt.ylabel('t-SNE 2')
  plt.show()

def plt_ICA(data):
  gene_expression = data 

  # Assuming you have your data loaded into a variable named 'data'
  # Replace this with your actual data loading process

  # Create an ICA object
  ica = FastICA(n_components=2, random_state=42)

  # Fit the ICA model to your data
  ica.fit(gene_expression)

  # Transform the gene_expression to the independent components
  independent_components = ica.transform(gene_expression)

  # Plot the independent components (assuming it's 2D)
  plt.figure(figsize=(8, 6))
  plt.scatter(independent_components[:, 0], independent_components[:, 1], s=10)
  plt.title('Independent Components')
  plt.xlabel('Component 1')
  plt.ylabel('Component 2')
  plt.show()

def plt_VAE(data):
    # Extract the expression data (assuming it starts from the 4th column)
    expression_data = data.iloc[:, 3:].values

    # Normalize the data
    expression_data = (expression_data - np.min(expression_data)) / (np.max(expression_data) - np.min(expression_data))
    expression_data = tf.convert_to_tensor(expression_data, dtype=tf.float32)

    # Define the VAE architecture
    latent_dim = 2  # Set the number of latent dimensions

    # Encoder
    encoder_inputs = keras.Input(shape=(expression_data.shape[1],))
    x = keras.layers.Dense(256, activation='relu')(encoder_inputs)
    x = keras.layers.Dense(128, activation='relu')(x)
    z_mean = keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = keras.layers.Dense(latent_dim, name='z_log_var')(x)

    # Reparameterization trick to sample from the latent space
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decoder
    decoder_inputs = keras.layers.Dense(128, activation='relu')(z)
    decoder_outputs = keras.layers.Dense(expression_data.shape[1], activation='sigmoid')(decoder_inputs)

    # Define the VAE model
    vae = keras.Model(encoder_inputs, decoder_outputs)

    # Define the VAE loss
    reconstruction_loss = tf.keras.losses.mean_squared_error(encoder_inputs, decoder_outputs)
    reconstruction_loss *= expression_data.shape[1]
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_mean(kl_loss)
    kl_loss *= -0.5
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    # Compile the VAE model
    vae.compile(optimizer='adam')

    # Train the VAE model
    vae.fit(expression_data, epochs=50, batch_size=32)

    # Encode data into the latent space
    encoder = keras.Model(encoder_inputs, z_mean)
    encoded_data = encoder.predict(expression_data)

    # Visualize the latent space
    plt.scatter(encoded_data[:, 0], encoded_data[:, 1], alpha=0.5)
    plt.title('VAE Latent Space')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.show()


def plt_umap(data):
  selected_columns = data.iloc[:, 3:]  
  umap_reducer = umap.UMAP(n_components=2)  

  umap_result = umap_reducer.fit_transform(selected_columns)

  umap_df = pd.DataFrame(umap_result, columns=['umap_1', 'umap_2'])

  plt.figure(figsize=(8, 6))
  plt.scatter(umap_df['umap_1'], umap_df['umap_2'], alpha=0.5)
  plt.title('UMAP Visualization of scRNA-seq Data')
  plt.xlabel('UMAP 1')
  plt.ylabel('UMAP 2')
  plt.show()


def plt_pca_variance(data):
  # Separate gene ID and gene name columns
  gene_info = data[['gene_id', 'gene_name', 'gene_type']]
  # Select only the expression data columns for PCA
  expression_data = data.drop(['gene_id', 'gene_name', 'gene_type'], axis=1)

  # Standardize the data
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(expression_data)

  # Apply PCA
  pca = PCA(n_components=10)  # You can change the number of components as needed
  principal_components = pca.fit_transform(scaled_data)

  # Get the explained variance ratios for each principal component
  explained_variances = pca.explained_variance_ratio_

  # Calculate cumulative explained variance
  cumulative_explained_variance = explained_variances.cumsum()

  # Print explained variance ratios for each principal component
  for i, explained_variance in enumerate(explained_variances):
      print(f"Principal Component {i + 1}: Explained Variance Ratio = {explained_variance:.4f}")

  # Print cumulative explained variance
  print(f"\nCumulative Explained Variance:\n{cumulative_explained_variance}")

  plt.figure(figsize=(8, 6))
  plt.plot(range(1, len(explained_variances) + 1), cumulative_explained_variance, marker='o', linestyle='-')
  plt.xlabel('Number of Components')
  plt.ylabel('Cumulative Explained Variance')
  plt.title('Cumulative Explained Variance by Principal Components')
  plt.grid(True)
  plt.show()



def pca(data):
    '''PCA'''
    # Select only the expression data columns for PCA
    expression_data = data

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(expression_data)

    # Apply PCA
    pca = PCA(n_components=2)  # You can change the number of components as needed
    principal_components = pca.fit_transform(scaled_data)
    return principal_components

def tsne(data):
    selected_columns = data
    tsne = TSNE(n_components=2, perplexity=30)  # Adjust parameters as needed

    # Perform t-SNE
    tsne_result = tsne.fit_transform(selected_columns)

def Umap(data):
    selected_columns = data  
    umap_reducer = umap.UMAP(n_components=2)  

    umap_result = umap_reducer.fit_transform(selected_columns)
    return umap_results

def ica(data):
    gene_expression = data
    ica = FastICA(n_components=2, random_state=42)


    ica.fit(gene_expression)

    independent_components = ica.transform(gene_expression)
    return independent_components

def vae(data):
    expression_data = data.values


    expression_data = (expression_data - np.min(expression_data)) / (np.max(expression_data) - np.min(expression_data))
    expression_data = tf.convert_to_tensor(expression_data, dtype=tf.float32)
    latent_dim = 2 

    encoder_inputs = keras.Input(shape=(expression_data.shape[1],))
    x = keras.layers.Dense(256, activation='relu')(encoder_inputs)
    x = keras.layers.Dense(128, activation='relu')(x)
    z_mean = keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = keras.layers.Dense(latent_dim, name='z_log_var')(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    decoder_inputs = keras.layers.Dense(128, activation='relu')(z)
    decoder_outputs = keras.layers.Dense(expression_data.shape[1], activation='sigmoid')(decoder_inputs)

    vae = keras.Model(encoder_inputs, decoder_outputs)

    reconstruction_loss = tf.keras.losses.mean_squared_error(encoder_inputs, decoder_outputs)
    reconstruction_loss *= expression_data.shape[1]
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_mean(kl_loss)
    kl_loss *= -0.5
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    vae.fit(expression_data, epochs=50, batch_size=32)

    encoder = keras.Model(encoder_inputs, z_mean)
    encoded_data = encoder.predict(expression_data)
    return encoded_data
