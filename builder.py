
import sys, random
import os
import time
import math

import pymunk
import pymunk.pygame_util
import pygame

import pickle

import numpy as np
import matplotlib.pyplot as plt

from scene_tools import *
import argparse

def main():
    bscene = BuilderScene()
    bscene.run()


main()