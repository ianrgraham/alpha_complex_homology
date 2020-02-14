from numba import njit
import numpy as np

@njit
def pc_b(b, d):
    return 1
@njit
def pc_ml(b, d):
    return (b+d)/2
@njit
def pc_l(b, d):
    return d-b
@njit
def pc_mul(b, d):
    return d/b

@njit
def sum_persistence_func(birth, death, func, points=None, os=1):
    if os != 0:
        birth = birth+os#[b+os for b in birth]
        death = death+os#[d+os for d in death]
        if points is None:
            points = np.linspace(-.5+os, 1.5+os, 151)
    elif points is None:
        points = np.linspace(-.5, 1.5, 151)

    s = np.zeros_like(points)
    for idx, i in enumerate(points):
        for b, d in zip(birth, death):
            if b >= i or d <= i:
                continue
            s[idx] += func(b, d)
    return points, s

@njit
def sum_pers_ent_func(birth, death, func, points=None, os=1):
    if os != 0:
        birth = birth+os#[b+os for b in birth]
        death = death+os#[d+os for d in death]
        if points is None:
            points = np.linspace(-.5+os, 1.5+os, 151)
    elif points is None:
        points = np.linspace(-.5, 1.5, 151)
    s = np.zeros_like(points)
    for idx, i in enumerate(points):
        tmp_terms = []
        for b, d in zip(birth, death):
            if b >= i or d <= i:
                continue
            tmp_terms.append((b, d))
        vfunc = np.vectorize(func)
        if len(tmp_terms) == 0:
            s[idx] = np.nan
            continue
        t2 = np.sum(vfunc(*zip(*tmp_terms)))
        for b, d in tmp_terms:  
            tmp = func(b, d)/t2
            s[idx] -= tmp*np.log(tmp)
    return points, s