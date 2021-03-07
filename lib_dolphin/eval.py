import numpy as np
import matplotlib.pyplot as plt


def plot_tensorflow_hist(hist, output):
    plt.plot(hist.history['mse'],     label="mse")
    plt.plot(hist.history['val_mse'], label="val_mse")
    plt.title("MSE")
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.savefig(output)
    plt.clf()


def visualize_dataset(instances, output):
    plt.figure(figsize=(20, 20))
    for i in range(0, 100):
        xi = np.random.randint(0, len(instances))
        plt.subplot(10, 10, i + 1)
        plt.imshow(1.0 - instances[xi].T, cmap='gray')
    plt.savefig(output)
    plt.clf()

    
def reconstruct(ae, instances, output):    
    plt.figure(figsize=(5, 10))
    t = instances[0][0]
    d = instances[0][1]
    for i in range(25):
        idx = np.random.randint(len(instances))
        reconstruction = ae.predict(x_train[idx].reshape(1, t, d, 1))
        reconstruction = reconstruction.reshape(t, d)
        plt.subplot(5, 5, i + 1)
        plt.imshow(reconstruction.T)
    plt.savefig(output)
    plt.clf()   
