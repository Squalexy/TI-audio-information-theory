import scipy.io.wavfile as spiowf
import sounddevice as sd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

def histograma(filename):


def main():
    filename = input("Nome do ficheiro : ")
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img = mpimg.imread(filename)
    elif filename.lower().endswith(('.wav', '.mp3')):
        [samplerate, data] = spiowf.read(filename)
    elif filename.lower().endswith(('.txt')):
        with open("Alpha_Particle.txt") as f:
            data = f.read()
            data = data.split('\n')

if __name__ == "__main__":
    