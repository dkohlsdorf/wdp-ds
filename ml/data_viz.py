import matplotlib.pyplot as plt
from audio import *
import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("python data_viz.py WIN FOLDER1 ... FOLDERN")
    else:        
        win     = int(sys.argv[1])
        folders = sys.argv[2:]
        for x in data_gen(folders, win):
            plt.imshow(1.0 - x[:, :, 0].T, cmap='gray')
            plt.show()