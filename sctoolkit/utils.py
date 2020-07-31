import pandas as pd
import scanpy as sc
import numpy as np

def get_expression_per_group(ad, genes, groupby, threshold=0, use_raw=False, layer=None, long_form=True, scale_percent=True):
    
    if isinstance(groupby, str):
        groupby = [groupby]
    
    if layer is not None:
        x = ad[:, genes].copy().layers[layer].A
    elif use_raw:
        x = ad.raw[:, genes].copy().X.A
    else:
        x = ad[:, genes].copy().X.A

    x = pd.DataFrame(x, index=ad.obs.index, columns=genes)
    key_df = sc.get.obs_df(ad, keys=groupby)
    genedf = pd.concat([x, key_df], axis=1)

    #genedf = sc.get.obs_df(ad, keys=[*groupby, *genes], use_raw=use_raw) #too slow
        
    grouped = genedf.groupby(groupby, observed=True)
    percent_scaler = 100 if scale_percent else 1
    
    exp = grouped.agg(lambda x: np.nanmean(x[x>thres])).fillna(0)
    exp.index.name = None
    percent = grouped.agg(lambda x: np.mean(x>thres)*percent_scaler).fillna(0)
    percent.index.name = None    

    if long_form:
        percent = percent.reset_index().melt(id_vars=groupby, value_name='percent_expr', var_name='gene')
        exp = exp.reset_index().melt(id_vars=groupby, value_name='mean_expr', var_name='gene')
        df = percent.merge(exp)

        return df
    else:
        return exp.T, percent.T