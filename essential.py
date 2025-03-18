import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display as widgvis


def load_mnist_data() -> tuple:
    """
    Load the MNIST dataset from the keras library.
    """
    X = np.load('./data/images.npy')
    X = X.reshape(70000, 784)
    y = np.load('./data/labels.npy')

    return X, y

def normalize(X:list, Y:list) -> tuple:
    """
    Normalize the dataset by dividing each pixel value by 255.
    """
    X = X/255
    Y = Y/255

    X = X - 0.5
    Y = Y - 0.5

    X = X * 2
    Y = Y * 2
    
    return X, Y

def test_train_split(X:list, Y:list, splitting_ratio:float) -> tuple:
    """
    Split the dataset into training and testing sets. 
    """
    X_train = X[:int(len(X)*splitting_ratio)]
    X_test = X[int(len(X)*splitting_ratio):]
    y_train = Y[:int(len(Y)*splitting_ratio)]
    y_test = Y[int(len(Y)*splitting_ratio):]
    return X_train, y_train, X_test, y_test

def displayImage(image:list):
    """
    Display an MNIST image.
    
    Parameters:
    -----------
    image : ndarray
        Flattened image with shape (784,)
    """
    # Reshape from (784,) to (28,28)
    img = image.reshape(28, 28)
    
    # Create a new figure with specific size
    plt.figure(figsize=(0.5, 0.5))
    
    # Display the image with grayscale colormap (similar to what's in your notebook)
    plt.imshow(img, cmap='gray')
    
    # Remove axes for cleaner display
    plt.axis('off')
    
    # Show the plot
    plt.show()

def displayImages(images, labels, grid_size):
    """
    Display multiple MNIST images in a square grid with labels below each image.
    """
    plt.figure(figsize=(grid_size, grid_size))
    
    for i in range(grid_size**2):
        # Create a subplot at the i-th position
        plt.subplot(grid_size, grid_size, i+1)
        

        random_index = np.random.randint(0, images.shape[0])
        img = images[random_index].reshape(28, 28)
        
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(labels[random_index])
    
    plt.tight_layout()
    plt.show()

def displayPredictions(images:list, labels:list, predictions:list, grid_size:int):
    """
    Display multiple MNIST images in a square grid with labels as well as predictions below each image.
    """
    plt.figure(figsize=(grid_size, grid_size))
    for i in range(grid_size**2):
        plt.subplot(grid_size, grid_size, i+1)
        random_index = np.random.randint(0, images.shape[0])
        img = images[random_index].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f'{predictions[random_index]} | {labels[random_index]}')
    plt.tight_layout()
    plt.suptitle("Prediction | Label", fontsize=16)
    plt.subplots_adjust(top=0.9)  # Add space for the title
    plt.show()
    
def DisplayErrors(images:list, labels:list, predictions:list):
    """
    Display all the MNIST images that were predicted incorrectly in a grid as well as the prediction and label below each image.
    """
    incorrect_indices = np.where(predictions != labels)[0]
    grid_size = int(np.ceil(np.sqrt(len(incorrect_indices))))
    plt.figure(figsize=(grid_size, grid_size))
    for i in range(grid_size**2):
        plt.subplot(grid_size, grid_size, i+1)
        if i < len(incorrect_indices):
            img = images[incorrect_indices[i]].reshape(28, 28)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.title(f'{predictions[incorrect_indices[i]]} | {labels[incorrect_indices[i]]}')
    plt.tight_layout()
    plt.show()
    print(f"Total errors: {len(incorrect_indices)} out of {len(images)}")

# def displayConfusionMatrix(confusion_matrix):
#     """
#     Display a confusion matrix as a heatmap.
#     """
#     plt.imshow(confusion_matrix, cmap='viridis')
#     plt.colorbar()
#     plt.xlabel('Predicted label')
#     plt.ylabel('True label')
#     plt.title('Confusion matrix')
#     plt.show()