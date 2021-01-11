from scipy.cluster.hierarchy import cut_tree, linkage
from tqdm.auto import tqdm
from math import floor
from scipy.stats import zscore
from natsort import natsorted

import scanpy as sc
import pandas as pd
import numpy as np
import scipy as sp


def find_modules(
    adata,
    n_levels=1000,
    level_start=2,
    level_end_size=2,
    n_pcs=100,
    layer=None,
    corr='pearson',
    corr_threshold=0,
    corr_power=2,
    method='paris',
    metric='correlation',
    smallest_module=3,
    key_added=None,
):
    
    if n_pcs is not None:
        print('Fitting PCA...')
        X = sc.pp.pca(adata, n_comps=n_pcs, copy=True).varm['PCs'].T
    else:
        if layer is None:
            X = adata.X
        else:
            X = adata.layers[layer]
        X = X.A if sp.sparse.issparse(X) else X

    key_added = '' if key_added is None else '_' + key_added
    
    if method == 'paris':
        from sknetwork.hierarchy import Paris
        
        if corr == 'pearson':
            corr_df = np.corrcoef(X, rowvar=False)
        elif corr == 'spearman':
            corr_df = sp.stats.spearmanr(X)[0]
        else:
            raise ValueError('Unknown corr')

        corr_df = pd.DataFrame(corr_df, columns=adata.var_names, index=adata.var_names)

        adata.varp[f'paris_corr_raw{key_added}'] = corr_df.copy()

        if corr_threshold is not None:
            corr_df[corr_df<corr_threshold] = corr_threshold

        if corr_power is not None:
            corr_df = corr_df.pow(corr_power)

        adata.varp[f'paris_corr{key_added}'] = corr_df.values

        print('Fitting the Paris model...')
        # TODO: consider BiParis on the tp10k matrix
        model = Paris()
        corr_mat = sp.sparse.csr_matrix(corr_df.values)
        dendro = model.fit_transform(corr_mat)
        
    else:
        print('Hierarchical clustering...')
        dendro = linkage(X.T, method=method, metric=metric)
    
    level_end = round(adata.n_vars/level_end_size)
    dfs = []
    n_cl_list = np.linspace(level_start, level_end, n_levels, dtype=int)

    if len(set(n_cl_list)) != len(n_cl_list):
        n_cl_list = pd.Series(n_cl_list)
        n_cl_list = n_cl_list[~n_cl_list.duplicated()].tolist()
        print(f'Not enough clusters for n_levels={n_levels}, reducing to {len(n_cl_list)}...')
    
    print('Cutting trees :( ...')
    cl = cut_tree(dendro, n_clusters=n_cl_list)
    dfs = pd.DataFrame(cl.astype(str), index=adata.var_names, columns=[f'level_{i}' for i in range(len(n_cl_list))])
    assert np.all(dfs.nunique().values == n_cl_list)

    for key in dfs.columns:
        dfs[key] = pd.Categorical(dfs[key], categories=natsorted(np.unique(dfs[key])))
    
    adata.varm[f'paris_partitions{key_added}'] = dfs
    adata.uns[f'paris{key_added}'] = {
        'dendrogram': dendro,
        'params': {layer: layer, n_levels:n_levels, corr:corr},
    }
    
    _build_module_dict(adata, level=None, paris_key=None if not key_added else key_added[1:])
    
    print('Removing duplicates and small modules...')
    # remove duplicate modules 
    df = pd.DataFrame(adata.uns[f'paris{key_added}']['module_dict']).reset_index()
    df = df.melt(id_vars=['index'], value_name='genes', var_name='level').rename(columns={'index': 'module'})
    df = df[~df.genes.isnull()]
    df['size'] = [len(x) for x in df.genes]

    small_idx = df['size'] < smallest_module
    dups_idx = df.duplicated('genes')
    
    print(f'{dups_idx.sum()} duplicates found...')
    print(f'{small_idx.sum()} small modules found...')
    
    rms = list(df[small_idx | dups_idx].reset_index(drop=True)[['module', 'level']].itertuples())
    for rm in rms:
        del adata.uns[f'paris{key_added}']['module_dict'][rm.level][rm.module]
        
    # remove empty levels
    empty_levels = [k for k,v in adata.uns[f'paris{key_added}']['module_dict'].items() if len(v) == 0]
    print(f'{len(empty_levels)} empty levels found...')
    newp = adata.varm[f'paris_partitions{key_added}'].iloc[:, ~adata.varm[f'paris_partitions{key_added}'].columns.isin(empty_levels)]
    adata.varm[f'paris_partitions{key_added}'] = newp
    
    for l in empty_levels:
        del adata.uns[f'paris{key_added}']['module_dict'][l]
        
    print(f'{len(df[~(small_idx | dups_idx)])} total modules found.')
    print('Calculating module dependencies...')

    deps = _calculate_module_dependencies(adata, paris_key=None if not key_added else key_added[1:])
    adata.uns[f'paris{key_added}']['module_dependencies'] = deps


def _build_module_dict(adata, level=None, paris_key=None):
    paris_key = '' if paris_key is None else '_' + paris_key
    if level is None:
        level = adata.varm[f'paris_partitions{paris_key}'].columns
    
    final_dict = {}
    print('Building the module dictionary...')
    for i, l in enumerate(tqdm(level)):
        l = l if str(l).startswith('level') else f'level_{l}'

        d = adata.varm[f'paris_partitions{paris_key}'].reset_index().groupby(l)['index'].agg(tuple).to_dict()
        final_dict[l] = d

    adata.uns[f'paris{paris_key}']['module_dict'] = final_dict


def _calculate_module_dependencies(adata, paris_key=None, only_upper=True):
    paris_key = '' if paris_key is None else '_' + paris_key

    df = pd.DataFrame(adata.uns[f'paris{paris_key}']['module_dict']).reset_index()
    df = df.melt(id_vars=['index'], value_name='genes', var_name='level').rename(columns={'index': 'module'})
    df = df[~df.genes.isnull()].reset_index(drop=True)

    df['levelint'] = [int(x.split('_')[1]) for x in df.level]
    df['genes'] = [set(x) for x in df['genes']]

    level_min = df.levelint.min()
    parents = []

    for t in tqdm(list(df[df.levelint>level_min].itertuples())):
        for cand in df[df.levelint<t.levelint].itertuples():
            if t.genes.issubset(cand.genes):
                parents.append((t.level, t.module, tuple(t.genes), cand.level, cand.module, tuple(cand.genes)))
    df = pd.DataFrame(parents, columns=['child_level', 'child_module', 'child_genes', 'parent_level', 'parent_module', 'parent_genes'])
    
    if only_upper:
        df['parent_size'] = [len(x) for x in df.parent_genes]
        df = df.loc[df.groupby(['child_level', 'child_module'])['parent_size'].idxmin()]
        df['child_level_int'] = [int(x.split('_')[1]) for x in df.child_level]
        df = df.sort_values(['child_level_int', 'child_module']).reset_index(drop=True)
    
    return df


def tag_with_score(adata, key, level=None, zcutoff=2.0, layer=None, paris_key=None):
    ad = adata.copy()
    paris_key = '' if paris_key is None else '_' + paris_key
    assert key in adata.obs_keys()
    
    if layer is not None:
        ad.X = ad.layers[layer] #sc.tl.score_genes doesn't suppert layers

    if level is None:
        level = adata.varm[f'paris_partitions{paris_key}'].columns.tolist()
        
    if not isinstance(level, (list, tuple)):
        level = [level]

    if isinstance(key, str):
        key = [key]
        
    print('Scoring all modules...')
    
    d = adata.uns[f'paris{paris_key}']['module_dict']
    
    result = []
    for l in tqdm(level):
        l = l if str(l).startswith('level') else f'level_{l}'
        
        clusters = d[l]
        dfs = []
        for cluster in clusters.keys():
            genes = list(clusters[cluster])
            if len(genes) == 1:
                score = ad.obs_vector(genes[0], layer=layer)
            else:
                ctrl_size = np.maximum(50, len(genes))
                sc.tl.score_genes(ad, genes, score_name='sc', ctrl_size=ctrl_size)
                score = ad.obs.sc.values.copy()
            score = zscore(score)
            dfs.append(pd.DataFrame(score, index=adata.obs_names, columns=[cluster]))

        class_df = pd.concat(dfs, axis=1)
        adata.obsm[f'paris_scores{paris_key}_{l}'] = class_df
        
        for k in key:
            cdf = class_df.groupby(ad.obs[k]).mean()
            cdf = cdf.reset_index().melt(id_vars=k, var_name='module', value_name='zscore')
            cdf[k] = pd.Categorical(cdf[k], categories=ad.obs[k].cat.categories)
            cdf.rename(columns={k: 'category'}, inplace=True)
            cdf['module'] = pd.Categorical(cdf['module'], categories=clusters.keys())
            cdf = cdf[cdf.zscore>zcutoff].sort_values(['module', 'category']).reset_index(drop=True)
            cdf = cdf.assign(level=l, key=k)
            cdf = cdf[['key', 'category', 'level', 'module', 'zscore']]
            result.append(cdf)
        
    result = pd.concat(result, axis=0).reset_index(drop=True)
    adata.uns[f'paris{paris_key}']['score_tags'] = result
    
    return result

        
def sort_module_dict(adata, level=None, paris_key=None, corr='pearson', layer=None):        

    paris_key = '' if paris_key is None else '_' + paris_key
    if level is None:
        level = adata.varm[f'paris_partitions{paris_key}'].columns
    
    if not np.any([x.startswith(f'paris_scores{paris_key}') for x in adata.obsm_keys()]):
        print('Paris scores not found, genes will not be sorted...')
        return

    print('Sorting the module dictionary...')
    
    for i, l in enumerate(tqdm(level)):
        l = l if str(l).startswith('level') else f'level_{l}'
        d = adata.uns[f'paris{paris_key}']['module_dict'][l]
        
        for module in d.keys():
            genes = list(d[module])
            if len(genes) == 1:
                continue

            X = adata[:, genes].X.toarray() if layer is None else adata.layers[layer][:, genes].X.toarray()
            df1 = pd.DataFrame(X, columns=genes)
            # ugly trick to compare only between dfs and not within
            df2 = pd.DataFrame(adata.obsm[f'paris_scores{paris_key}_{l}'][module].values[:, None].repeat(len(genes), axis=1), columns=genes)

            genes = df1.corrwith(df2, method=corr).sort_values(ascending=False).index.tolist()
            d[module] = tuple(genes)
