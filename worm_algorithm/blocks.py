from __future__ import division
import random
import time
import sys
import os
import getopt
import argparse

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from worm_algorithm.bonds import Bonds

class Blocks(Bonds):
    """ Class: Blocks, used to implement `coarse-graining`
    renormalization-group transformation on worm configurations. 
