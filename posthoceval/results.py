"""
results.py - A PostHocExplainerEvaluation file
Copyright (C) 2020  Zach Carmichael
"""
from collections import namedtuple

ExprResult = namedtuple('ExprResult',
                        'symbols,expr,domains,state,kwargs')
