# -*- coding: utf-8 -*-

import pytest
from pandalyse.selector import Selector

__author__ = "Simon Wehle"
__copyright__ = "Simon Wehle"
__license__ = "mit"


import pandas as pd 
import numpy as np

def create_test_df(size=10):
    df = pd.DataFrame({'A' : np.arange(size),
                      'B' : np.random.randint(2,size=size),}
                     )
    return df

def test_selection_init_str():
    df = create_test_df(10)
    s = Selector("A > 5")
    assert len(s(df)) == 4

def test_selection_init_list():
    df = create_test_df(10)
    s = Selector(['A>5', 'A<9'])
    assert len(s(df)) == 3

def test_selector_add_rm():
    df = create_test_df(10)
    s = Selector()
    assert len(s.cutlist) == 0
    s.add_cut("A>5")
    s.add_cut("A<9")
    assert len(s.cutlist) == 2
    s.rm_cut(1)
    assert len(s.cutlist) == 1
    # Only first cut left
    assert s.cutlist[0] == 'A>5'
