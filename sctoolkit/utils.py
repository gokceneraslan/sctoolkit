import pandas as pd
import scanpy as sc
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import zscore

from plotnine import *
import warnings

# requirements for _indexed_expression_df and summarized_expression_df
from typing import Optional, Iterable, Tuple, Mapping, Union, Sequence, Literal
from anndata import AnnData
_VarNames = Union[str, Sequence[str]]
from pandas.api.types import is_categorical_dtype


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
    xlabel='',
    ylabel='',
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
        labs(size = "-log10(adj. P value)", y=ylabel if ylabel else ycol, x=xlabel if xlabel else xcol, title=title) +
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


def run_spring(ad, key, groups=None, varm_key=None, store_in_varm=True, layer=None):
    from scrublet.helper_functions import rank_enriched_genes, sparse_zscore
    from scipy.sparse import issparse, csr_matrix
    from tqdm.auto import tqdm

    E = ad.X if layer is None else ad.layers[layer]
    sparse = issparse(E)
    z = sparse_zscore(E).A if sparse else zscore(E, axis=0)

    if groups is None:
        groups = ad.obs[key].cat.categories

    dfs = []
    for group in tqdm(groups):
        cell_mask = (ad.obs[key] == group).values
        scores = z[cell_mask,:].mean(0).squeeze()
        o = np.argsort(-scores)

        df = pd.DataFrame(dict(names=ad.var_names[o], spring_score=scores[o]))
        dfs.append(df.assign(group=group))

    dfs = pd.concat(dfs, axis=0).reset_index(drop=True)
    dfs['group'] = pd.Categorical(dfs.group, categories=ad.obs[key].cat.categories)

    if store_in_varm:

        if varm_key is None:
            varm_key = f'spring_{key}'

        varm = dfs.pivot(index='names', values='spring_score', columns='group').loc[ad.var_names]
        varm.index.name = ad.var.index.name
        varm.columns.name = None

        # workaround for https://github.com/theislab/anndata/issues/459
        ad.varm.dim_names = ad.var_names
        ad.varm[varm_key] = varm

    return dfs


def dotplot_spring(adata, key, groups=None, n_genes=10, spring_cutoff=None, update=False, *args, **kwargs):
    if groups is None:
        groups = adata.obs[key].cat.categories

    if f'spring_{key}' in adata.varm_keys() and not update:
        df = adata.varm[f'spring_{key}'].copy()
        df.columns = df.columns.astype(str)
        df = df.reset_index().melt(id_vars='index', value_name='spring_score', var_name='group')
        df = df.rename(columns={'index': 'names'})
        df['group'] = pd.Categorical(df.group, categories=adata.obs[key].cat.categories)
        df = df.sort_values(['group', 'spring_score'], ascending=[True, False]).reset_index(drop=True)
    else:
        df = run_spring(adata, key, groups)

    if spring_cutoff is None:
        d = {k: df[df.group == k].names[:n_genes] for k in groups}
    else:
        d = {k: df[(df.group == k) & (df.spring_score>=spring_cutoff)].names for k in groups}

    return sc.pl.dotplot(adata, var_names=d, groupby=key, *args, **kwargs)


def sort_by_correlation(mat, rows=True, metric='correlation', method='complete', optimal_ordering=True):
    if not rows:
        mat = mat.T
    Z = linkage(mat, metric=metric, method=method, optimal_ordering=optimal_ordering)
    dn = dendrogram(Z, no_plot=True)
    return np.array([int(x) for x in dn['ivl']])


def plot_enrichment(
    genes,
    num_pathways=20,
    title='',
    ordered=True,
    cutoff=0.05,
    sources=('GO:BP', 'HPA', 'REAC'),
    organism='hsapiens',
    return_df=False,
):
    en_df = sc.queries.enrich(genes, org=organism, gprofiler_kwargs=dict(no_evidences=False, ordered=ordered, all_results=True, user_threshold=cutoff, sources=sources))
    en_df['name'] = en_df['name'].str.capitalize()
    en_df['intersections'] = ['(' + ','.join(x[:3]) + ')' for x in en_df.intersections]
    en_df['name'] = en_df['name'].astype(str) + ' ' + en_df['intersections'].astype(str)
    en_df = en_df.drop_duplicates('name')[:num_pathways]
    en_df['neglog10_pval'] = -np.log10(en_df['p_value'])
    en_df['name'] = pd.Categorical(en_df['name'], categories=en_df['name'], ordered=True)

    figsize = (7,len(en_df)/4)
    text_start = (en_df.neglog10_pval.max()*0.01)

    g = (
        ggplot(en_df, aes(x='name', y='neglog10_pval')) +
        geom_bar(aes(fill='significant'), stat='identity', color='#0f0f0f', size=0.1) +
        geom_hline(yintercept=-np.log10(cutoff), size=0.05, color='black') +
        geom_text(aes(x='name', y=text_start, label='name'), size=8, ha='left') +coord_flip() +
        scale_x_discrete(limits=list(reversed(en_df.name.cat.categories))) +
        scale_fill_manual({True:'#D3D3D3', False:'#efefef'}) +
        theme_classic() +
        theme(
            figure_size=figsize, panel_spacing_x=1.,
            axis_text_y = element_blank(),
            legend_position = 'none',
        ) +
        labs(y='Gene Set Enrichment (-log10(adj. P value))', x='Pathways', title=title)
    )

    if not return_df:
        return g
    else:
        return g, en_df


def knn2mnn(adata):
    d = adata.uns['neighbors']['distances'] != 0.
    mnn_mask = d.multiply(d.T)
    adata.uns['neighbors']['distances'] = adata.uns['neighbors']['distances'].multiply(mnn_mask)
    adata.uns['neighbors']['connectivities'] = adata.uns['neighbors']['connectivities'].multiply(mnn_mask)


# from https://github.com/theislab/scanpy/pull/1390

def _indexed_expression_df(
    adata: AnnData,
    var_names: Optional[Union[_VarNames, Mapping[str, _VarNames]]] = None,
    groupby: Optional[Union[str, Sequence[str]]] = None,
    use_raw: Optional[bool] = None,
    log: bool = False,
    num_categories: int = 7,
    layer: Optional[str] = None,
    gene_symbols: Optional[str] = None,
    concat_indices: bool = True,
):
    """
    Given the anndata object, prepares a data frame in which the row index are the categories
    defined by group by and the columns correspond to var_names.

    Parameters
    ----------
    adata
        Annotated data matrix.
    var_names
        `var_names` should be a valid subset of `adata.var_names`. All genes are used if no
        given.
    groupby
        The key of the observation grouping to consider. It is expected that
        groupby is a categorical. If groupby is not a categorical observation,
        it would be subdivided into `num_categories`.
    use_raw
        Use `raw` attribute of `adata` if present.
    log
        Use the log of the values
    num_categories
        Only used if groupby observation is not categorical. This value
        determines the number of groups into which the groupby observation
        should be subdivided.
    gene_symbols
        Key for field in .var that stores gene symbols.
    concat_indices
        Concatenates categorical indices into a single categorical index, if 
        groupby is a sequence. True by default.

    Returns
    -------
    Tuple of `pandas.DataFrame` and list of categories.
    """
    from scipy.sparse import issparse

    adata._sanitize()
    if use_raw is None and adata.raw is not None:
        use_raw = True
    if isinstance(var_names, str):
        var_names = [var_names]
    if var_names is None:
        if use_raw:
            var_names = adata.raw.var_names.values
        else:
            var_names = adata.var_names.values

    if groupby is not None:
        if isinstance(groupby, str):
            # if not a list, turn into a list
            groupby = [groupby]
        for group in groupby:
            if group not in adata.obs_keys():
                raise ValueError(
                    'groupby has to be a valid observation. '
                    f'Given {group}, is not in observations: {adata.obs_keys()}'
                )

    if gene_symbols is not None and gene_symbols in adata.var.columns:
        # translate gene_symbols to var_names
        # slow method but gives a meaningful error if no gene symbol is found:
        translated_var_names = []
        # if we're using raw to plot, we should also do gene symbol translations
        # using raw
        if use_raw:
            adata_or_raw = adata.raw
        else:
            adata_or_raw = adata
        for symbol in var_names:
            if symbol not in adata_or_raw.var[gene_symbols].values:
                logg.error(
                    f"Gene symbol {symbol!r} not found in given "
                    f"gene_symbols column: {gene_symbols!r}"
                )
                return
            translated_var_names.append(
                adata_or_raw.var[adata_or_raw.var[gene_symbols] == symbol].index[0]
            )
        symbols = var_names
        var_names = translated_var_names
    if layer is not None:
        if layer not in adata.layers.keys():
            raise KeyError(
                f'Selected layer: {layer} is not in the layers list. '
                f'The list of valid layers is: {adata.layers.keys()}'
            )
        matrix = adata[:, var_names].layers[layer]
    elif use_raw:
        matrix = adata.raw[:, var_names].X
    else:
        matrix = adata[:, var_names].X

    if issparse(matrix):
        matrix = matrix.toarray()
    if log:
        matrix = np.log1p(matrix)

    obs_tidy = pd.DataFrame(matrix, columns=var_names)
    if groupby is None:
        groupby = ''
        obs_tidy_idx = pd.Series(np.repeat('', len(obs_tidy))).astype('category')
        idx_categories = obs_tidy_idx.cat.categories
    else:
        if len(groupby) == 1 and not is_categorical_dtype(adata.obs[groupby[0]]):
            # if the groupby column is not categorical, turn it into one
            # by subdividing into  `num_categories` categories
            obs_tidy_idx = pd.cut(adata.obs[groupby[0]], num_categories)
            idx_categories = obs_tidy_idx.cat.categories
        else:
            assert all(is_categorical_dtype(adata.obs[group]) for group in groupby)
            if concat_indices:
                obs_tidy_idx = adata.obs[groupby[0]]
                if len(groupby) > 1:
                    for group in groupby[1:]:
                        # create new category by merging the given groupby categories
                        obs_tidy_idx = (
                            obs_tidy_idx.astype(str) + "_" + adata.obs[group].astype(str)
                        ).astype('category')
                obs_tidy_idx.name = "_".join(groupby)
                idx_categories = obs_tidy_idx.cat.categories
            else:
                obs_tidy_idx = [adata.obs[group] for group in groupby] # keep as multiindex
                idx_categories = [x.cat.categories for x in obs_tidy_idx]

    obs_tidy.set_index(obs_tidy_idx, inplace=True)
    if gene_symbols is not None:
        # translate the column names to the symbol names
        obs_tidy.rename(
            columns={var_names[x]: symbols[x] for x in range(len(var_names))},
            inplace=True,
        )

    return idx_categories, obs_tidy


def summarized_expression_df(
    adata: AnnData,
    groupby: Union[str, Sequence[str]],
    ops: Optional[Literal['mean_expressed', 'var_expressed', 'fraction']] = None,
    long_format: bool = True,
    var_names: Optional[Union[_VarNames, Mapping[str, _VarNames]]] = None,
    use_raw: Optional[bool] = None,
    log: bool = False,
    layer: Optional[str] = None,
    threshold: float = 0.,
    gene_symbols: Optional[str] = None,
) -> pd.DataFrame:
    """\
    Creates a dataframe where gene expression is grouped by key(s) in adata.obs (`groupby`)
    and aggregated using given functions (`ops`).

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        The key of the observation grouping to consider. It is expected that
        groupby is a categorical.
    ops
        Operations to execute on the grouped dataframe. By default mean and variance of
        expression above the specified threshold (`mean_expressed` and `var_expressed`)
        and fraction of cells expressing given genes above threshold (`fraction`) are
        returned.
    long_format
        Whether to keep the gene names in columns (False) or in rows (True). True by default.
    var_names
        `var_names` should be a valid subset of `adata.var_names`. All genes are used if no
        given.
    use_raw
        Use `raw` attribute of `adata` if present.
    log
        Use the log of the values
    layer
        Layer to use instead of adata.X.
    threshold
        Expression threshold for mean_expressed and var_expressed ops.
    gene_symbols
        Key for field in .var that stores gene symbols.

    Returns
    -------
    `pandas.DataFrame`

    Example
    -------
    >>> import scanpy as sc
    >>> adata = sc.datasets.paul15()
    >>> adata.obs['somecat'] = pd.Categorical(['A' if x == '3Ery' else 'B' for x in adata.obs.paul15_clusters])
    >>> df = sc.get.summarized_expression_df(adata, groupby=['paul15_clusters', 'somecat'])
    """
    if isinstance(groupby, str):
        groupby = [groupby]
    assert all(is_categorical_dtype(adata.obs[group]) for group in groupby)
    _, df = _indexed_expression_df(
        adata,
        groupby=groupby,
        var_names=var_names,
        use_raw=use_raw,
        log=log,
        layer=layer,
        gene_symbols=gene_symbols,
        concat_indices=False,
    )

    if ops is None:
        ops = ['mean_expressed', 'var_expressed', 'fraction']
    if isinstance(ops, str):
        ops = [ops]
    assert all(np.isin(ops, ['mean_expressed', 'var_expressed', 'fraction'])), 'Undefined op'
    assert len(ops) > 0, 'No ops given'

    res = {}
    # .agg is super slow even for mean and var, so do it separately using .mean and .var
    if 'mean_expressed' in ops or 'var_expressed' in ops:
        nonzero_group = df.mask(df<=threshold).groupby(level=df.index.names, observed=True)
        if 'mean_expressed' in ops:
            res['mean_expressed'] = nonzero_group.mean()
        if 'var_expressed' in ops:
            res['var_expressed'] = nonzero_group.var()
    if 'fraction' in ops:
        res['fraction'] = (df>threshold).groupby(level=df.index.names, observed=True).mean()

    res = pd.concat(res.values(), axis=1, keys=res.keys(), names=[None, 'gene'])

    if long_format:
        res = res.stack(level=1).reset_index('gene')

    return res


def bin_pval(pvals):
    return pd.cut(pvals,
                  [0, 0.001, 0.01, 0.05, 0.1, 1],
                  labels=['***', '**', '*', '.', ' '],
                  include_lowest=True)

