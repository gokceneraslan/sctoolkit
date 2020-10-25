
import pandas as pd
from plotnine import *
import scanpy as sc
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .topicmodel.sbmtm import sbmtm
from .utils import sort_by_correlation


def plot_topic(topic, n_genes=15):
    df = pd.DataFrame(topic[:n_genes], columns=['Gene', "Prob"])
    df.Gene = df.Gene.astype('category').cat.reorder_categories(df.Gene, ordered=True )
    return qplot('Gene', 'Prob', data=df) + theme_minimal() +  theme(axis_text_x=element_text(rotation=90, hjust=1))


def plot_topics(topics, figsize=(25, 15), scale='free', highlight=None, ncols=10, panel_spacing_x=1., panel_spacing_y=1., x_label_map=None, **kwargs):

    if isinstance(topics, sc.AnnData):
        topics = get_topic_dict(topics)

    df = pd.concat([pd.DataFrame([list(y) + [topic] for y in x], columns=['Gene', "Prob", 'Topic']) for topic, x in topics.items()], axis=0)
    df['Topic'] = df.Topic.astype('category').cat.reorder_categories(topics.keys(), ordered=True)
    
    if highlight is not None:
        if isinstance(highlight, (list, tuple, np.ndarray, pd.Series, pd.Index)):
            df['Highlight'] = [x in highlight for x in df.Gene]
        elif isinstance(highlight, pd.DataFrame):
            df['Gene'] = df.Gene.astype(str)
            highlight['Highlight'] = highlight['Highlight'].astype(str)
            df = df.merge(highlight, how='left', on='Gene') #.fillna('None')
        else:
            raise ValueError
        
        df['Gene'] = df.Gene.astype('category').cat.reorder_categories(df.Gene, ordered=True )
        return (qplot('Gene', 'Prob', data=df, color='Highlight') + 
                facet_wrap('Topic', scales=scale, ncol=ncols) + 
                theme_minimal() + 
                theme(axis_text_x=element_text(rotation=90, hjust=0.5), 
                      panel_spacing_x=panel_spacing_x,
                      panel_spacing_y=panel_spacing_y,
                      figure_size=figsize, **kwargs))

            
    else:
        df['Gene'] = df.Gene.astype('category').cat.reorder_categories(df.Gene, ordered=True)
        x_reversed = pd.CategoricalDtype(categories=reversed(df['Gene'].cat.categories), ordered=True)
        df['Gene'] = df.Gene.astype(x_reversed)
        
        g = (
            qplot('Gene', 'Prob', data=df) + 
            facet_wrap('Topic', scales=scale, ncol=ncols) + 
            theme_minimal() + 
            theme(figure_size=figsize,
                  panel_spacing_x=panel_spacing_x,
                  panel_spacing_y=panel_spacing_y,
                  **kwargs) +                 #axis_text_x=element_text(rotation=90, hjust=1., size=7),
            coord_flip()
            )
        
        if x_label_map:
            g += scale_x_discrete(labels=x_label_map)
            
        return g


def fit_topic_model(adata, n_hvg=10000, count_layer='counts', n_init_jobs=1, n_init=1, **kwds):
    ad = adata.copy()

    print(f'Finding {n_hvg} HVGs')
    sc.pp.highly_variable_genes(ad, n_top_genes=n_hvg, flavor='seurat_v3', subset=True, layer=count_layer)
    print(ad)

    ad.X = (ad.layers[count_layer] > 0).astype(float)
    sc.pp.filter_cells(ad, min_genes=1)

    model = sbmtm()

    print('Constructing bipartite graph')
    model.make_graph_anndata(ad, weight_property=None)

    print('Fitting SBM topic model')
    model.fit(n_init_jobs=n_init_jobs, n_init=n_init, **kwds)

    L = len(model.state.levels)
    dict_groups_L = {}

    print('Estimating topic probabilities')
    ## only trivial bipartite structure
    if L == 2:
        model.L = 1
        for l in range(L-1):
            dict_groups_l = model.get_groups(l=l)
            dict_groups_L[l] = dict_groups_l

    # omit trivial levels: l=L-1 (single group), l=L-2 (bipartite)
    else:
        model.L = L-2
        for l in range(L-2):
            dict_groups_l = model.get_groups(l=l)
            dict_groups_L[l] = dict_groups_l

    model.groups = dict_groups_L

    assert np.all(ad.var_names == model.words)
    assert np.all(ad.obs_names == model.documents)

    print('Saving topic results to adata varm/obsm')
    for k, v in model.groups.items():
        ad.varm[f'topic_{k}'] = v['p_w_tw'].copy()
        ad.obsm[f'topic_{k}'] = v['p_tw_d'].T.copy()

    return ad, model


def get_topic_dict(adata, level=0, top=10):
    mat = adata.varm[f'topic_{level}']
    max_idx = (-mat).argsort(axis=0)
    ret = {}

    for i in range(mat.shape[1]):
        scores = mat[max_idx[:, i], i][:top]
        names = adata.var_names[max_idx[:, i]][:top]
        ret[i] = [(x,y) for x,y in zip(names, scores) if y>0]

    return ret


def plot_topic_umap(ad, topic_dict, n_cols=10, width=35, height_scale=3.2, sort=True):
    ad = ad.copy()
    ret = topic_dict

    for i, program in ret.items():
        genes = [x[0] for x in program]
        score_name = f'Score Topic {i}'
        sc.tl.score_genes(ad, genes, score_name=score_name)
        ad.obs[score_name] = zscore(ad.obs[score_name])

    if sort:
        topic_score_mat = np.vstack([ad.obs[f'Score Topic {i}'].values for i in range(len(ret))])
        topic_order = sort_by_correlation(topic_score_mat)
    else:
        topic_order = list(range(len(ret)))

    n_topics = len(topic_order)
    n_rows = int(np.ceil(n_topics/n_cols))
    f, axs = plt.subplots(n_rows, n_cols, figsize=(width, height_scale*n_rows))
    axs = axs.flatten()

    for topic, ax in zip(topic_order, axs):
        sc.pl.umap(ad, color=f'Score Topic {topic}', cmap='RdBu_r', vmin=-3, vmax=3, ax=ax, show=False, title=f'Topic {topic}', legend_loc='none', frameon=False)

    for ax in axs[n_topics:]:
        ax.set_visible(False)


def plot_topic_gene_set_enrichment(
    adata,
    level=0,
    top_topic_genes=30,
    num_pathways=15,
    title='',
    ordered=True,
    cutoff=0.05,
    sources=('GO:BP', 'HPA', 'REAC'),
    organism='hsapiens',
    return_df=False,
    figure_width=35,
    figure_height_scale=0.6,
):
    topic_dict = get_topic_dict(adata, top=top_topic_genes, level=level)

    dfs = []
    for topic, g in tqdm(list(topic_dict.items())):
        genes = [x[0] for x in g]

        df = sc.queries.enrich(genes, org=organism, gprofiler_kwargs=dict(no_evidences=False, ordered=ordered, all_results=True, user_threshold=cutoff, sources=sources))
        df['name'] = df['name'].str.capitalize()
        df['intersections'] = ['(' + ','.join(x[:3]) + ')' for x in df.intersections]
        df['name'] = df['name'].astype(str) + ' ' + df['intersections'].astype(str)
        df = df.drop_duplicates('name')[:num_pathways]
        df['neglog10_pval'] = -np.log10(df['p_value'])

        dfs.append(df.assign(topic=topic))

    df = pd.concat(dfs, axis=0)
    df['topic_name'] = df.topic.astype(str) + '-' + df.name

    df = df.merge(df.groupby('topic')[['neglog10_pval']].max().rename(columns={'neglog10_pval': 'neglog10_pval_topic_max'}).reset_index())
    df['topic_name'] = pd.Categorical(df['topic_name'], categories=df['topic_name'][::-1], ordered=True)

    n_topics = df.topic.nunique()
    figsize = (figure_width, (figure_height_scale*n_topics))

    text_start = (df.neglog10_pval_topic_max*0.01)

    g = (
        ggplot(df, aes(x='topic_name', y='neglog10_pval')) +
        geom_bar(aes(fill='significant'), stat='identity', color='#0f0f0f', size=0.1) +
        geom_hline(yintercept=-np.log10(cutoff), size=0.05, color='black') +
        geom_text(aes(x='topic_name', y=text_start, label='name'), size=8, ha='left') + coord_flip() +
        facet_wrap('topic', scales='free', ncol=6) +
        scale_fill_manual({True:'#D3D3D3', False:'#efefef'}) +
        theme_classic() +
        theme(
            figure_size=figsize,
            panel_spacing_x=0.3,
            panel_spacing_y=0.3,
            axis_text_y = element_blank(),
            legend_position = 'none',
        ) +
        labs(y='Gene Set Enrichment (-log10(adj. P value))', x='Pathways')
    )

    if not return_df:
        return g
    else:
        return g, df
