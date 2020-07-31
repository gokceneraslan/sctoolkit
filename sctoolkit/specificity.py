from .utils import get_expression_per_group

import numpy as np
import pandas as pd

_eps = 1e-20

def BCE(y, yhat):
    return -(y*np.log2(yhat+_eps)) - ((1-y)*np.log2(1-yhat+_eps))


def BCE_asym(y, yhat):
    res = -(y*np.log2(yhat+_eps)) - ((1-y)*np.log2(1-yhat+_eps))
    return res * (((y>yhat).astype(float)*2.)-1.)    


def KL_pq(p, q):
    return -(p*np.log2((q/(p+_eps))+_eps)) - ((1-p)*np.log2(((1-q)/(1-p))+_eps))


def KL_pq_asym(p, q):
    res = -(p*np.log2((q/(p+_eps))+_eps)) - ((1-p)*np.log2(((1-q)/(1-p))+_eps))
    return res * (((p>q).astype(float)*2.)-1.)


def KL_qp(p, q):
    return -(q*np.log2((p/q)+_eps) - ((1-q)*np.log2(((1-p)/(1-q))+1e-50)))


def JS(i, j):
    m = ((i+j)/2)
    return (KL_pq(i,m) + KL_pq(j,m)) /2.


def JS_asym(i, j):
    m = ((i+j)/2)
    res = (KL_pq(i,m) + KL_pq(j,m)) /2. 
    return res * (((i>j).astype(float)*2.)-1.)


def JS_asym2(i,j):
    m = ((i+j)/2)
    res = -(i*np.log2((m/(i+_eps))+_eps)) -(j*np.log2((m/(j+_eps))+_eps))
    return res * (((i>j).astype(float)*2.)-1.)


_all_spec_funcs = {
    'BCE_asym': BCE_asym,    
    'KL_pq_asym': KL_pq_asym,
    'JS_asym': JS_asym,
    'JS_asym2': JS_asym2
}


def get_gene_specificity_metrics(adata, group_key, group, reference='rest', genes=None):
    
    if reference == 'rest':
        reference = f'Non-{group}'
    else:
        assert reference in adata.obs[group_key].cat.categories, f'Reference is not a valid category in {group_key}'
        assert reference != group, 'Group and reference cannot be the same'

        adata = adata[adata.obs[group_key].isin([group, reference])].copy()
        
    adata.obs['__specificity_temp'] = [group if x else reference for x in adata.obs[group_key] == group]        

    if genes is None:
        genes = adata.var_names.tolist()
        
    _, p = get_expression_per_group(adata, genes, '__specificity_temp', long_form=False, scale_percent=False)
    adata.obs.drop('__specificity_temp', inplace=True, axis=1)
    
    res = pd.DataFrame({k: v(p.loc[:, group], 
                             p.loc[:, f'Non-{group}']) for k,v in _all_spec_funcs.items()})
    
    return res.assign(group=p.loc[:, group], nongroup=p.loc[:, f'Non-{group}'])