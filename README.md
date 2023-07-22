# sign-mnist-analysis
Project Description - "Sign MNIST Dataset Analysis"

In this project, we will be exploring the Sign MNIST dataset, which contains images of American Sign Language (ASL) gestures representing letters from A to Z (excluding J and Z) and numbers from 0 to 9. The main objectives of this analysis are as follows:

# Dataset Preparation:
We will download and load the Sign MNIST dataset, dividing it into training, validation, and test sets in the same proportion as in previous tasks (50k images for training, 10k images for validation, and 10k images for testing).

# Dimensionality Reduction using PCA:
We will apply Principal Component Analysis (PCA) to the training portion of the dataset to reduce the number of features while preserving as much variance as possible.
The number of principal components required to retain 95% of the variance will be determined.

# Random Forest Classifier:
A Random Forest classifier will be trained on the reduced dataset obtained through PCA.
We will evaluate the classifier's performance on the test set and compare it to the results obtained in previous tasks (if applicable).

# Visualizing the Impact of PCA:
To understand the impact of PCA on image representation, we will randomly select and plot 10 images in their original form and compare them with the same images after dimensionality reduction using PCA to retain 95% of variance.

# Explained Variance with First Two Principal Components:
We will calculate and report the amount of variance explained by the first two principal components, which can provide insights into the dataset's structure.

# Visualization using t-SNE, LLE, and MDS:

We will reduce the dimensionality of the dataset to only two dimensions using t-SNE (t-Distributed Stochastic Neighbor Embedding), LLE (Locally Linear Embedding), and MDS (Multidimensional Scaling).Scatter plots will be created to visualize 1000 random images from the training set, with different colors representing each image's target class (letter/number).A comparison will be made between the visualizations obtained using different dimensionality reduction techniques, and the preferred method will be discussed.

# Clustering with K-Means:

We will sample 10,000 images from the training portion of the Sign MNIST dataset and use PCA to reduce the dimensionality.
K-Means clustering will be applied to the reduced dataset, and an appropriate number of clusters will be determined using techniques discussed in class.
We will visualize the clusters, showcasing a subset of images in each cluster to observe similar clothing items.
Clustering with Gaussian Mixture Model (GMM):

Similar to the K-Means approach, we will use PCA to reduce the dimensionality of 10,000 images from the training set.
A Gaussian Mixture Model (GMM) will be employed to cluster the reduced dataset, aiming to identify groups of similar images.
We will visualize the clusters and observe similarities between clothing items within each cluster.
Image Generation using GMM:

The GMM model will be used to generate 20 new clothing items from the Sign MNIST dataset.
We will visualize these generated images by applying the inverse_transform() method to the PCA-reduced representations.
Building Fully Connected Neural Network:

A fully connected (dense) feedforward neural network will be constructed using Keras within TensorFlow.
The neural network will have two hidden layers with ReLU activation functions, containing 200 and 50 neurons, respectively.
The network will be trained on the Fashion MNIST training images for 100 epochs, using three different learning rates, and the training and validation loss and accuracy will be plotted.
Evaluation of Neural Network:

The trained neural network will be evaluated on the test portion of the Sign MNIST dataset, using the best-performing learning rate.
Loss and accuracy metrics will be reported.

# Analysis of Activation Functions:

The previous step will be repeated with hidden layers having linear activation functions.The impact of linear activation functions on accuracy will be discussed.
Through this extensive analysis of the Sign MNIST dataset, we aim to gain valuable insights into the effectiveness of dimensionality reduction techniques, clustering algorithms, and neural network models for image classification and visualization tasks. The results obtained will provide a better understanding of the dataset's structure and help in developing efficient models for future image-based applications.
