# Facial-Recognition-using-Deep-Learning
Data Preparation:
The training images are augmented by rescaling, rotating (shear), zooming, and flipping to help the model learn better. The testing images are only rescaled.

Model Architecture:
The model is a Convolutional Neural Network with two convolutional layers followed by max-pooling layers to extract important features from the images. After flattening, there is one fully connected layer with dropout to reduce overfitting, and a final output layer with sigmoid activation for binary classification.

Training:
The model is trained using the Adam optimizer and binary cross-entropy loss for 300 epochs with images resized to 64x64 pixels.

Results:
The validation accuracy improves initially (about 70% after 100 epochs), but starts to decrease after more epochs. By 300 epochs, accuracy drops to 60%, and validation loss increases, showing the model is overfitting (memorizing training data instead of learning general patterns).

Conclusion:
Overfitting happens when the model trains too long on limited data. To improve, use early stopping, increase data size, or add more augmentation.

