"""
generate_data.py - A PostHocExplainerEvaluation file
Copyright (C) 2020  Zach Carmichael
"""
import sympy as S
from sympy import stats

U = stats.Uniform('U', -1, +1)
interval = S.Interval(-.001, S.oo)
constraint = interval.contains(U)
