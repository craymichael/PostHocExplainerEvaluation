"""
generate_data.py - A PostHocExplainerEvaluation file
Copyright (C) 2020  Zach Carmichael
"""
import sympy as sp
from sympy import stats

U = stats.Uniform('U', -1, +1)
interval = sp.Interval(-.001, sp.oo)
constraint = interval.contains(U)
