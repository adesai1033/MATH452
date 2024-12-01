from sklearn.datasets import fetch_openml
import numpy as np
# Load MNIST dataset - contains 70,000 images of handwritten digits
# Each image is 28x28 pixels, flattened to 784 dimensions
def load():
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    x = x.astype(float)  # Convert x to float to allow division
    x /= 255.0
    #np.random.seed(1)
    
    # Randomly select 10,000 indices
    #indices = np.random.choice(len(x), 10000, replace=False)
    
    # Select samples based on these indices
    #x_selected = x[indices]
    #y_selected = y[indices]
    
    return x, y

def main():
    x, y = load()
    print(x.shape)
    print("Maximum value in x:", np.max(x))
    print("Success")

if __name__ == '__main__':
    main()