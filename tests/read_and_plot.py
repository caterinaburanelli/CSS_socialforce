from contextlib import contextmanager
from matplotlib.cbook import flatten
import numpy as np
import pytest
import socialforce
import pandas as pd
from matplotlib import pyplot as plt

layouts = ["benchmark", "single", "pillars", "horizontal", "angled"]
data=[]

for layout in layouts :
    data.append(pd.read_csv('data/df_{}_test'.format(layout)))
    

