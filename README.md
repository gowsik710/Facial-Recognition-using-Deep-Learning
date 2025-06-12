# Facial-Recognition-using-Deep-Learning

### Data Preparation:

To improve the model’s ability to generalize and prevent overfitting, data augmentation is applied to the training dataset. The training images are first rescaled by dividing pixel values by 255 to normalize them between 0 and 1, which helps the neural network train more efficiently. Additionally, random transformations such as shear (small rotations/skews), zooming in/out, and horizontal flipping are applied. These augmentations artificially increase the diversity of the training data by simulating different viewing angles and conditions, allowing the model to learn more robust and invariant features.
For the testing dataset, only rescaling is performed to ensure that evaluation is done on unaltered images, providing a realistic measure of the model's performance.

### Model Architecture:

The model uses a Convolutional Neural Network (CNN) architecture, which is well-suited for image classification tasks. It consists of the following layers:

* **Convolutional layers:** Two convolutional layers with ReLU activation detect local features like edges, textures, and patterns in the images. The first layer uses 32 filters of size 5x5, and the second uses 64 filters of the same size, allowing the model to learn increasingly complex features.
* **Max-Pooling layers:** Each convolutional layer is followed by a max-pooling layer, which reduces the spatial size of the feature maps by taking the maximum value over non-overlapping regions. This helps reduce computation, controls overfitting, and focuses on the most prominent features.
* **Flattening layer:** After feature extraction, the multi-dimensional output is flattened into a single vector to feed into the dense (fully connected) layers.
* **Fully connected layer:** A dense layer with 32 units and ReLU activation combines the extracted features to learn high-level representations.
* **Dropout layer:** A dropout rate of 0.4 is applied, randomly turning off 40% of neurons during training to reduce overfitting by preventing the model from relying too heavily on any one feature.
* **Output layer:** The final layer has one neuron with sigmoid activation, producing a probability score for binary classification (e.g., face vs. non-face).

### Training:

The model is compiled with the Adam optimizer, which adapts the learning rate during training for efficient convergence. The loss function used is binary cross-entropy, appropriate for two-class problems. Training occurs over 300 epochs with batches of 32 images, each resized to 64x64 pixels for uniformity. The model's performance is evaluated on validation data after each epoch to monitor learning progress.

### Results:

Initially, the validation accuracy improves and reaches around 70% after 100 epochs, indicating the model is learning useful features. However, beyond this point, the validation accuracy begins to decline despite continued training. By 300 epochs, accuracy drops to about 60%, and the validation loss increases significantly. This pattern indicates **overfitting**, where the model memorizes the training data instead of generalizing to unseen data, resulting in poorer performance on validation images.

### Conclusion:

Overfitting occurs when a model is trained for too many epochs on limited or insufficiently varied data. To mitigate overfitting and improve generalization, several strategies can be employed:

* Implement **early stopping** to halt training when validation performance stops improving.
* Increase the size and diversity of the training dataset, if possible.
* Apply more aggressive or varied **data augmentation** techniques.
* Experiment with regularization methods such as higher dropout rates or L2 regularization.
* Consider simplifying the model architecture if it is too complex for the available data.
