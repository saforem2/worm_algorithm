import os
import subprocess
import sys
import argparse

def download_mnist():
    """Download mnist dataset into 'mnist/' directory."""
    current_dir = os.getcwd()
    mnist_data_dir = '../data/mnist/'
    if not os.path.exists(mnist_data_dir):
        os.makedirs(mnist_data_dir)
    os.chdir(mnist_data_dir)
    os.system('curl -O http://deeplearning.net/data/mnist/mnist.pkl.gz')
    os.chdir(current_dir)
    mnist_data_file = mnist_data_dir.join('mnist.pkl.gz')
    if os.path.isfile(mnist_data_file):
        print(f"Successfully downloaded mnist data file to: {mnist_data_file}")


if __name__ == '__main__':
    download_mnist()


