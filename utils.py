from typing import List, Tuple

import numpy as np
from scipy import stats

def pearsonr_ci(x: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> Tuple[float, float, float, float]:
    ''' 
    Taken from https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/

    Calculate Pearson correlation along with the confidence interval using scipy and numpy

    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.1 by default d
    
    Returns
    -------
    r : float  =  Pearson's correlation coefficient
    pval : float  =  The corresponding p value
    lo, hi : float  =  The lower and upper bound of confidence intervals
    '''
    # TODO: More principled response to this?
    if max(x) == min(x) or max(y) == min(y):
        return -0.5, -0.5, -0.5, 1.0

    r, p = stats.pearsonr(x,y) # blank value is p-value
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return float(r), float(lo), float(hi), float(p)