from scipy.cluster.hierarchy import cut_tree
from scipy.stats import zscore
from tqdm.auto import tqdm


def find_modules(
    adata,
    n_levels=20,
    level_start=3,
    level_end_size=3,
    n_pcs=100,
    layer=None,
    corr='pearson',
    corr_threshold=0,
    corr_power=2,
):
    from sknetwork.hierarchy import Paris, LouvainHierarchy
    
    if n_pcs is not None:
        X = sc.pp.pca(adata, n_comps=n_pcs, copy=True).varm['PCs']
        exp_df = pd.DataFrame(X, index=adata.var_names).T
        corr_df = exp_df.corr(method=corr)
    else:
        if layer is None:
            exp_df = pd.DataFrame(adata.X, columns=adata.var_names)
        else:
            exp_df = pd.DataFrame(adata.layers[layer], columns=adata.var_names)

        corr_df = exp_df.corr(method=corr)

    adata.varp['paris_corr_raw'] = corr_df.copy()

    corr_df[corr_df<corr_threshold] = 0
    if corr_power is not None:
        corr_df = corr_df.pow(corr_power)
        
    adata.varp['paris_corr'] = corr_df.values        

    # TODO: consider BiParis on the tp10k matrix
    model = Paris()
    corr_mat = sp.sparse.csr_matrix(corr_df.values)
    dendro = model.fit_transform(corr_mat)
    
    level_end = round(adata.n_vars/level_end_size)
    dfs = []
    n_cl_list = np.linspace(level_start, level_end, n_levels, dtype=int)

    if len(set(n_cl_list)) != len(n_cl_list):
        n_cl_list = pd.Series(n_cl_list)
        n_cl_list = n_cl_list[~n_cl_list.duplicated()].tolist()
        print(f'Not enough clusters for n_levels={n_levels}, reducing to {len(n_cl_list)}...')
        
    for i, n_cl in enumerate(tqdm(n_cl_list)):
        cl = cut_tree(dendro, n_clusters=n_cl).ravel()
        assert n_cl == len(set(cl))
        cl = pd.Categorical(cl.astype(str), categories=[str(x) for x in sorted(np.unique(cl))])
        df = pd.DataFrame(cl, index=ad.var_names, columns=[f'level_{i}'])
        dfs.append(df)

    adata.varm['paris_partitions'] = pd.concat(dfs, axis=1)
    adata.uns['paris'] = {'dendrogram': dendro}


def tag_with_score(adata, key, level=None, zcutoff=2.0, layer=None):
    ad = adata.copy()
    
    if layer is not None:
        ad.X = ad.layers[layer]

    if level is None:
        level = adata.varm['paris_partitions'].columns.tolist()
        
    if not isinstance(level, (list, tuple)):
        level = [level]

    if not isinstance(key, (list, tuple)):
        key = [key]
        
    result = []
    for l in tqdm(level):
        clusters = adata.varm['paris_partitions'][l if str(l).startswith('level') else f'level_{l}']
        dfs = []
        for cluster in tqdm(clusters.cat.categories, leave=False):
            genes = clusters[clusters == cluster].index.tolist()
            if len(genes) == 1:
                score = ad.obs_vector(genes[0], layer=layer)
            else:
                ctrl_size = np.maximum(50, len(genes))
                sc.tl.score_genes(ad, genes, score_name='sc', ctrl_size=ctrl_size)
                score = ad.obs.sc.values.copy()
            score = zscore(score)
            dfs.append(pd.DataFrame(score, index=adata.obs_names, columns=[cluster]))

        class_df = pd.concat(dfs, axis=1)
        adata.obsm[f'paris_scores_level_{l}'] = class_df
        
        for k in key:
            cdf = class_df.groupby(ad.obs[k]).mean()
            cdf = cdf.reset_index().melt(id_vars=k, var_name='module', value_name='zscore')
            cdf[k] = pd.Categorical(cdf[k], categories=ad.obs[k].cat.categories)
            cdf.rename(columns={k: 'category'}, inplace=True)
            cdf['module'] = pd.Categorical(cdf['module'], categories=clusters.cat.categories)
            cdf = cdf[cdf.zscore>zcutoff].sort_values(['module', 'category']).reset_index(drop=True)
            cdf = cdf.assign(level=l, key=k)
            cdf = cdf[['key', 'category', 'level', 'module', 'zscore']]
            result.append(cdf)
        
    result = pd.concat(result, axis=0).reset_index(drop=True)
    adata.uns['paris']['score_tags'] = result
    
    return result
