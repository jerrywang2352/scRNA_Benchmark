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

  selected_gene_types.remove('miRNA')
  return selected_gene_types

def expression_data():
   data = pd.read_csv('../data/data.csv')
   data = data.T
   data = data.drop(data.index[:3])
   return data
   
def plt_PCA(data):
  '''PCA'''
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(data)

  # Apply PCA
  pca = PCA(n_components=2)  
  pca_result = pca.fit_transform(scaled_data)

  # Create a DataFrame from PCA result for easy plotting
  pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])  

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

  tsne = TSNE(n_components=2, perplexity=30)  

  # Perform t-SNE
  tsne_result = tsne.fit_transform(selected_columns)


  tsne_df = pd.DataFrame(tsne_result, columns=['tsne_1', 'tsne_2'])


  plt.figure(figsize=(8, 6))
  plt.scatter(tsne_df['tsne_1'], tsne_df['tsne_2'], alpha=0.5)
  plt.title('t-SNE Visualization of scRNA-seq Data')
  plt.xlabel('t-SNE 1')
  plt.ylabel('t-SNE 2')
  plt.show()

def plt_ICA(data):
  gene_expression = data 


  ica = FastICA(n_components=2, random_state=42)

  ica.fit(gene_expression)

  independent_components = ica.transform(gene_expression)

  plt.figure(figsize=(8, 6))
  plt.scatter(independent_components[:, 0], independent_components[:, 1], s=10)
  plt.title('Independent Components')
  plt.xlabel('Component 1')
  plt.ylabel('Component 2')
  plt.show()

def plt_VAE(data):
    expression_data = data.iloc[:, 3:].values

    expression_data = (expression_data - np.min(expression_data)) / (np.max(expression_data) - np.min(expression_data))
    expression_data = tf.convert_to_tensor(expression_data, dtype=tf.float32)

    latent_dim = 2  # Set the number of latent dimensions

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
  gene_info = data[['gene_id', 'gene_name', 'gene_type']]
  expression_data = data.drop(['gene_id', 'gene_name', 'gene_type'], axis=1)

  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(expression_data)

  # Apply PCA
  pca = PCA(n_components=10)  
  principal_components = pca.fit_transform(scaled_data)

 
  explained_variances = pca.explained_variance_ratio_

  cumulative_explained_variance = explained_variances.cumsum()

  for i, explained_variance in enumerate(explained_variances):
      print(f"Principal Component {i + 1}: Explained Variance Ratio = {explained_variance:.4f}")

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
    expression_data = data

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(expression_data)

    pca = PCA(n_components=2) 
    principal_components = pca.fit_transform(scaled_data)
    return principal_components

def tsne(data):
    selected_columns = data
    tsne = TSNE(n_components=2, perplexity=30)  

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
