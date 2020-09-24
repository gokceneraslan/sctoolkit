import pandas as pd
import scanpy as sc
import numpy as np
from plotnine import *
import warnings

def get_expression_per_group(ad, genes, groupby=None, groups=None, threshold=0, use_raw=False, layer=None, long_form=True, scale_percent=True):

    if groups is not None:
        assert isinstance(groups, dict), 'groups must be a dict'

        for k,v in groups.items():
            ad = ad[ad.obs[k].isin(v)]
    
    if layer is not None:
        x = ad[:, genes].copy().layers[layer].A
    elif use_raw:
        x = ad.raw[:, genes].copy().X.A
    else:
        x = ad[:, genes].copy().X.A

    x = pd.DataFrame(x, index=ad.obs.index, columns=genes)

    if groupby  is not None:
        if isinstance(groupby, str):
            groupby = [groupby]
        
        key_df = sc.get.obs_df(ad, keys=groupby)
        genedf = pd.concat([x, key_df], axis=1)

        #genedf = sc.get.obs_df(ad, keys=[*groupby, *genes], use_raw=use_raw) #too slow
            
        grouped = genedf.groupby(groupby, observed=True)
    else:
        grouped = x

    percent_scaler = 100 if scale_percent else 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        exp = grouped.agg(lambda x: np.nanmean(x[x>threshold])).fillna(0)
        percent = grouped.agg(lambda x: np.mean(x>threshold)*percent_scaler).fillna(0)

    if long_form:
        if groupby is not None:
            percent = percent.reset_index().melt(id_vars=groupby, value_name='percent_expr', var_name='gene')
            exp = exp.reset_index().melt(id_vars=groupby, value_name='mean_expr', var_name='gene')
            df = percent.merge(exp)
        else:
            df = pd.DataFrame(dict(percent_expr=percent, mean_expr=exp)).reset_index()
            df.rename(columns={'names': 'gene'}, inplace=True)
        return df
    else:
        return exp.T, percent.T



def plot_significance_dotplot(
    df,
    xcol='variable', 
    ycol='compartment',
    title='', 
    size='neglog_pval_adj',
    fill='coefficient', 
    color='significant', 
    color_values=('#808080', '#990E1D'),
    fill_limit=(-2,2),
    size_limit=10,
    dot_size_limit=10,
    width_scale=1.0,
    height_scale=1.0,
):

    from statsmodels.stats.multitest import multipletests
    
    df = df.copy()
    
    #ct = df.groupby(xcol)['significant'].sum() > 0
    #ct = ct[ct].index
    #df = df[df[xcol].isin(ct)]

    df.loc[df[fill] < fill_limit[0], fill] = fill_limit[0]
    df.loc[df[fill] > fill_limit[1], fill] = fill_limit[1]
    df.loc[df[size] > size_limit, size] = size_limit

    df[ycol] = pd.Categorical(df[ycol], categories = reversed(sorted(df[ycol].unique())))
    limit = max(df[fill].abs()) * np.array([-1, 1])

    g = (
        ggplot(aes(x=xcol, y=ycol), data=df) +
        geom_point(aes(size=size, fill=fill, color=color))+
        scale_fill_distiller(type='div', limits=limit, name='Effect size') + 
        scale_color_manual(values=color_values) + 
        labs(size = "-log10(adj. P value)", y=ycol, x=xcol, title=title) +
        guides(size = guide_legend(reverse=True)) +
        theme_bw() +
        scale_size(range = (1,dot_size_limit)) +
        scale_y_discrete(drop=False) +
        #scale_x_discrete(drop=False) +
        theme(
          figure_size=(9*width_scale,12*height_scale),
          legend_key=element_blank(),
          axis_text_x = element_text(rotation=45, hjust=1.),
        )
    )

    return g


def run_spring(ad, key, groups=None, varm_key=None):
    from scrublet.helper_functions import rank_enriched_genes, sparse_zscore
    from scipy.sparse import issparse, csr_matrix
    from tqdm.auto import tqdm

    E = ad.X if issparse(ad.X) else csr_matrix(ad.X)
    z = sparse_zscore(E)

    if groups is None:
        groups = ad.obs[key].cat.categories

    dfs = []
    for group in tqdm(groups):
        cell_mask = (ad.obs[key] == group).values
        scores = z[cell_mask,:].mean(0).A.squeeze()
        o = np.argsort(-scores)

        df = pd.DataFrame(dict(names=ad.var_names[o], spring_score=scores[o]))
        dfs.append(df.assign(group=group))

    dfs = pd.concat(dfs, axis=0).reset_index(drop=True)
    dfs['group'] = pd.Categorical(dfs.group, categories=ad.obs[key].cat.categories)

    if varm_key is None:
        varm_key = f'spring_{key}'

    ad.varm[varm_key] = dfs.pivot(index='names', values='spring_score', columns='group').loc[ad.var_names]

    return dfs


def dotplot_spring(adata, key, groups=None, n_genes=10, update=False, *args, **kwargs):
    if groups is None:
        groups = adata.obs[key].cat.categories

    if f'spring_{key}' in adata.varm_keys() and not update:
        df = adata.varm[f'spring_leiden'].copy()
        df.columns = df.columns.astype(str)
        df = df.reset_index().melt(id_vars='index', value_name='spring_score')
        df = df.rename(columns={'index': 'names'})
        df.group = pd.Categorical(df.group, categories=adata.obs[key].cat.categories)
        df = df.sort_values(['group', 'spring_score'], ascending=[True, False]).reset_index(drop=True)
    else:
        df = run_spring(adata, key, groups)

    d = {k: df[df.group == k].names[:n_genes] for k in groups}

    return sc.pl.dotplot(adata, var_names=d, groupby=key, *args, **kwargs)
