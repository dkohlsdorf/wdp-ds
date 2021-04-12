import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write    


def label(x, label_dict):
    n_labels = len(label_dict)
    n_labels = np.zeros(n_labels)
    for l in x:
        n_labels[l] += 1
    return label_dict[np.argmax(n_labels)]


def export_audio(c, labels, windows, label_dict, out, min_support = 25):
    windows_by_cluster = {}
    labels_by_cluster = {}
    for i, k in enumerate(c):
        win = windows[i]
        lab = labels[i]
        if k not in windows_by_cluster:
            windows_by_cluster[k] = []
            labels_by_cluster[k] = []
        windows_by_cluster[k].append(win)
        labels_by_cluster[k].append(lab)

    reverse = dict([(v, k) for k, v in label_dict.items()])
    
    whitelist = {}
    cur = 0
    for c, instances in windows_by_cluster.items():
        l = label(labels_by_cluster[c], reverse)
        audio = []
    
        if len(instances) > min_support:
            whitelist[c] = cur
            cur += 1
            for instance in instances:
                for x in instance:
                    audio.append(x)
                for i in range(0, 1000):
                    audio.append(0)
            audio = np.stack(audio)
            write('{}/{}_{}.wav'.format(out, l, whitelist[c]), 44100, audio.astype(np.int16)) 
        return whitelist
    

def plot_result_matrix(confusion, classes, predictions, title, cmap=plt.cm.Blues):
    fig, ax = plt.subplots()
    im = ax.imshow(confusion, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(confusion.shape[1]),
           yticks=np.arange(confusion.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=predictions, yticklabels=classes,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fmt = '.1f'
    thresh = confusion.max() / 2.
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, format(confusion[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plot_tensorflow_hist(hist, output):
    if 'mse' in hist.history:
        plt.plot(hist.history['mse'],     label="mse")
        plt.plot(hist.history['val_mse'], label="val_mse")
        plt.title("MSE")
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('mse')
    elif 'accuracy' in hist.history:
        plt.plot(hist.history['accuracy'],     label="accuracy")
        plt.plot(hist.history['val_accuracy'], label="val_accuracy")
        plt.title("Accuracy")
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
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
    t = instances[0].shape[0]
    d = instances[0].shape[1]
    for i in range(25):
        idx = np.random.randint(len(instances))
        reconstruction = ae.predict(instances[idx].reshape((1, t, d, 1)))
        reconstruction = reconstruction.reshape((t, d))
        plt.subplot(5, 5, i + 1)
        plt.imshow(reconstruction.T)
    plt.savefig(output)
    plt.clf()   


def enc_filters(enc, n_filters, output):
    plt.figure(figsize=(10, 10))
    w = enc.weights[0].numpy()
    for i in range(w.shape[-1]):
        weight = w[:, :, 0, i]
        plt.subplot(n_filters / 8, 8, i + 1)
        plt.imshow(weight.T, cmap='gray')
    plt.savefig(output)
    plt.clf()
