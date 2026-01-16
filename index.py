import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silences TF logs
import numpy as np
from keras.datasets import mnist
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


x= np.random.rand(1, 32*32*3)
y= np.random.rand(1, 32*32*3)

def load_and_prep_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0],-1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0],-1).astype('float32')

    x_train /= 255
    x_test /= 255
    return x_train , y_train , x_test , y_test

x_train, y_train, x_test, y_test = load_and_prep_data()


def predict(test_img , x_train , y_train , k=5):

    dists = compute_distance(test_img , x_train)
    closest_indices = np.argsort(dists[0])[:k]
    closest_labels = y_train[closest_indices]

    counts = np.bincount(closest_labels , minlength=10)
    probabilities = counts /k
    prediction = np.argmax(probabilities)

    return prediction , probabilities

def visualize(image_vector , label, proba):
    plt.imshow(image_vector.reshape(28,28), cmap="gray")
    plt.title(f"Predicted: {label} | conf: {proba[label]*100:.1f}%" )
    plt.axis('off')
    plt.savefig('prediction.png')
    plt.show()


def compute_distance(test, train):

    test_sum = np.sum(test, axis=1 , keepdims=True)
    train_sum = np.sum(train, axis=1)

    inner_product = 2 * np.dot(train , test.T)

    dists = np.sqrt(np.maximum(0 , test_sum + train_sum - inner_product))
    return dists

test_img = x_test[0:1]
pred,proba = predict(test_img , x_train , y_train)

print(f"predicted digit: {pred}")
print(f"probabilities {proba}")
visualize(x_test[0], pred, proba)
