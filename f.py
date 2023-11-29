import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.spatial.distance import cdist

class KMeansGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("k-Means Clustering")

        self.canvas = tk.Canvas(self.master, width=400, height=300, relief='raised')
        self.canvas.pack()

        self.label1 = tk.Label(self.master, text='k-Means Clustering')
        self.label1.config(font=('helvetica', 14))
        self.canvas.create_window(200, 25, window=self.label1)

        self.label2 = tk.Label(self.master, text='Type Number of Clusters:')
        self.label2.config(font=('helvetica', 8))
        self.canvas.create_window(200, 120, window=self.label2)

        self.entry1 = tk.Entry(self.master)
        self.canvas.create_window(200, 140, window=self.entry1)

        self.browse_button = tk.Button(self.master, text=" Import Excel File ", command=self.get_excel, bg='green',
                                       fg='white', font=('helvetica', 10, 'bold'))
        self.canvas.create_window(200, 70, window=self.browse_button)

        self.process_button = tk.Button(self.master, text=' Process k-Means ', command=self.get_kmeans, bg='brown',
                                        fg='white', font=('helvetica', 10, 'bold'))
        self.canvas.create_window(200, 170, window=self.process_button)

    def get_excel(self):
        # Open a file dialog to choose an Excel file
        import_file_path = filedialog.askopenfilename()
        # Read the Excel file into a Pandas DataFrame
        read_file = pd.read_excel(import_file_path)
        # Store the DataFrame in the class for further use
        self.df = pd.DataFrame(read_file, columns=['x', 'y'])

    def get_kmeans(self):
        # Get the number of clusters from the user input
        numberOfClusters = int(self.entry1.get())

        # Using scikit-learn's KMeans for initialization and comparison
        kmeans_sklearn = KMeans(n_clusters=numberOfClusters).fit(self.df)
        centroids_sklearn = kmeans_sklearn.cluster_centers_

        # Convert data to TensorFlow tensors
        X_tensor = tf.constant(self.df, dtype=tf.float32)

        # K-means algorithm using TensorFlow and Keras
        kmeans_model = keras.Sequential([
            keras.layers.Input(shape=(self.df.shape[1],)),
            keras.layers.Dense(numberOfClusters, activation='softmax')
        ])

        kmeans_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        kmeans_model.fit(X_tensor, np.zeros(len(self.df)), epochs=10, verbose=0)

        # Get cluster assignments from the Keras model
        clusters_keras = np.argmax(kmeans_model.predict(X_tensor), axis=1)

        # Plot the results
        self.__plot__(self.df, clusters_sklearn, centroids_sklearn,
                       clusters_keras, kmeans_model.get_layer(index=0).get_weights()[0].T)

    def __plot__(self, X, clusters_sklearn, centroids_sklearn, clusters_keras, centroids_keras):
        '''
        Plotting the final cluster using matplotlib
        '''
        figure, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Plot using scikit-learn results
        axs[0].scatter(X['x'], X['y'], c=clusters_sklearn, cmap='viridis', alpha=0.7, edgecolors='k')
        axs[0].scatter(centroids_sklearn[:, 0], centroids_sklearn[:, 1], c='red', marker='X', s=200, label='Centroids')
        axs[0].set_title('Scikit-learn KMeans')

        # Plot using TensorFlow and Keras results
        axs[1].scatter(X['x'], X['y'], c=clusters_keras, cmap='viridis', alpha=0.7, edgecolors='k')
        axs[1].scatter(centroids_keras[:, 0], centroids_keras[:, 1], c='red', marker='X', s=200, label='Centroids')
        axs[1].set_title('TensorFlow and Keras')

        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    kmeans_gui = KMeansGUI(root)
    root.mainloop()
