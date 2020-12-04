from .rtools import r2py, py2r

import logging
import numpy as np
import pandas as pd
from plotnine import *
from plotnine.data import diamonds as ddata
import matplotlib.gridspec as gridspec



def get_proportions_per_channel(adata, sample_key, proportion_key, covariates=None):

    prop_df = pd.DataFrame(adata.obs.groupby([sample_key, proportion_key]).size(), columns=['ncell']).reset_index()

    prop_df = prop_df.pivot(index=sample_key, columns=proportion_key, values='ncell').fillna(0)
    prop_df.columns.name = None
    prop_df.columns = prop_df.columns.astype(str)
    prop_df /= prop_df.sum(1).values[:, None]
    prop_df.index = prop_df.index.astype(str)

    if covariates is not None:
        assert np.all(np.isin(covariates, adata.obs.columns))

        # check if all categoricals are nested in sample_key
        cat_covariates = [x for x in covariates if adata.obs[x].dtype.kind not in 'biufc']
        if cat_covariates:
            assert len(adata.obs[[sample_key] + cat_covariates].drop_duplicates()) == adata.obs[sample_key].nunique()

        covar_df = adata.obs.groupby(sample_key)[covariates].agg(**{x: pd.NamedAgg(x, 'first') if x in cat_covariates else pd.NamedAgg(x, 'mean') for x in covariates})
        covar_df = covar_df.loc[prop_df.index.values]
        covar_df.index = covar_df.index.astype(str)

        for c in cat_covariates:
            if adata.obs[c].dtype.name == 'category':
                covar_df[c] = pd.Categorical(covar_df[c], categories=adata.obs[c].cat.categories)

        assert np.all(prop_df.index == covar_df.index)

        return prop_df, covar_df
    else:
        return prop_df



def dirichletreg(adata, sample_key, proportion_key, covariates, formula, onevsrest_category=None, return_reg_input=False):

    from rpy2.robjects import r, Formula
    from rpy2.robjects.packages import importr
    from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

    adata._sanitize()
    prop_df, covar_df = get_proportions_per_channel(adata, sample_key, proportion_key, covariates)
    dr_df = pd.concat([prop_df, covar_df], axis=1)

    dr = importr('DirichletReg')

    f = Formula(formula)

    rpy2_logger.setLevel(logging.ERROR)   # will display errors, but not warnings
    f.environment['y'] = dr.DR_data(py2r(prop_df))
    rpy2_logger.setLevel(logging.WARNING)   # will display errors, but not warnings

    if onevsrest_category is None:
        fit = dr.DirichReg(f, py2r(dr_df))
    else:
        assert onevsrest_category in adata.obs[proportion_key].cat.categories
        cat_index = adata.obs[proportion_key].cat.categories.tolist().index(onevsrest_category) + 1
        fit = dr.DirichReg(f, py2r(dr_df), model='alternative', **{'sub.comp': cat_index})

    r.sink(r.tempfile()) # quietly
    u = r.summary(fit)
    r.sink()

    if onevsrest_category is None:
        varnames = u.rx2('varnames')
    else:
        varnames = [onevsrest_category] * 2

    coef_mat = u.rx2('coef.mat')
    rows = r2py(r('rownames')(coef_mat))
    coef_df = r2py(r('as.data.frame')(coef_mat)).reset_index(drop=True)
    coef_df.columns = ['coefficient', 'se', 'zval', 'pval']

    coef_df['compartment'] = np.repeat(varnames, r2py(u.rx2('n.vars')))
    coef_df['variable'] = rows
    coef_df['significance'] = bin_pval(coef_df.pval)

    if onevsrest_category is not None:
        coef_df['coef_type'] = np.repeat(['mean', 'precision'], r2py(u.rx2('n.vars')))

    if return_reg_input:
        return dr_df, coef_df
    else:
        return coef_df


def bin_pval(pvals):
    return pd.cut(pvals,
                  [0, 0.001, 0.01, 0.05, 0.1, 1],
                  labels=['***', '**', '*', '.', ' '],
                  include_lowest=True)


def plot_proportion_barplot(
    adata,
    yaxis,
    fill, 
    fill_breakdown=None,
    yaxis_label=None,
    fill_label=None,
    percent_limit=5., 
    show_percent=True,
    height_scale=1., 
    width_scale=1.,
    legend_position=(-0.3, 0.5),
    return_df=False,
    normalize_by=None,
):

    import mizani
    import matplotlib.patheffects as pe

    if yaxis_label is None: yaxis_label = yaxis
    if fill_label is None: fill_label = fill

    adata._sanitize()

    fill_dict = {k:v for k,v in zip(adata.obs[fill].cat.categories, adata.uns[f'{fill}_colors'])}

    df_level1 = pd.DataFrame(adata.obs.groupby([yaxis, fill] + ([fill_breakdown] if fill_breakdown else []), observed=True).size(), columns=['counts'])

    if normalize_by:
        scales = df_level1.reset_index().groupby(normalize_by)[['counts']].sum()
        scales = scales.sum().div(scales)

        df_level1 = df_level1.multiply(scales)

    df_level0 = df_level1.reset_index().groupby(yaxis)[['counts']].sum()
    df = df_level1.div(df_level0, level=yaxis).reset_index()
    
    df[fill]  = pd.Categorical(df[fill], categories=reversed(adata.obs[fill].cat.categories))
    df[yaxis] = pd.Categorical(df[yaxis], categories=reversed(adata.obs[yaxis].cat.categories))

    df['counts_coarse'] = df.groupby([yaxis, fill], observed=True)['counts'].transform('sum')
    df['counts_coarse_round_percent'] = (df.counts_coarse*100).round().astype(int)

    df['_show_text'] = df.counts_coarse_round_percent >= percent_limit
    df['_show_breakdown'] = (df.counts_coarse_round_percent >= percent_limit) if fill_breakdown else False
        
    # collapse breakdown of small groups
    if fill_breakdown:
        df = df[(~df.duplicated([yaxis, fill])) | (df._show_breakdown)].copy()
        df.loc[~df._show_breakdown, 'counts'] = df.loc[~df._show_breakdown, 'counts_coarse']
        df['_show_breakdown'] = True
        
    cs = df.sort_values([yaxis, fill], ascending=False).drop_duplicates([yaxis, fill]).groupby(yaxis, observed=True)['counts_coarse'].transform(pd.Series.cumsum)
    df['cumsum_mean'] = cs - df['counts_coarse'] + (df['counts_coarse']/2)        

    g = (
        ggplot(aes(x=yaxis, y='counts', fill=fill, group=fill), data=df) +
        geom_bar(position='fill', stat='identity', mapping=aes(color='_show_breakdown'), size=0.08) +
        scale_y_continuous(labels=mizani.formatters.percent) +
        coord_flip() +
        theme_minimal() +
        theme(figure_size=(8*width_scale, 
                           0.4*df[yaxis].nunique()*height_scale),
             legend_position=legend_position) + 
        scale_color_manual(values={True: 'black', False: 'none'}) +
        scale_fill_manual(values=fill_dict) +        
        labs(x=yaxis_label, y=fill_label) +
        guides(fill = guide_legend(reverse=True), color=None)
    )

    if show_percent:
        g += geom_text(aes(label='counts_coarse_round_percent', y='cumsum_mean'), data=df[df._show_text],
                  color='white', size=8, fontweight='bold',
                  path_effects=(pe.Stroke(linewidth=1, foreground='black'), pe.Normal()))

    if return_df:
        return g, df
    else:
        return g

    
def plot_proportion_barplot_cellcounts(
    adata,
    yaxis,
    height_scale=1., 
    width_scale=1.,
    legend_position=None,
):

    import mizani
    import matplotlib.patheffects as pe
    
    adata._sanitize()

    df = pd.DataFrame(adata.obs.groupby([yaxis], observed=True).size(), columns=['ncell']).reset_index()
    df['counts'] = 1
    df[yaxis] = pd.Categorical(df[yaxis], categories=reversed(adata.obs[yaxis].cat.categories))    
    

    g = (
        ggplot(aes(x=yaxis, y='counts', fill='ncell.astype(float)'), data=df) +
        geom_col() +
        coord_flip() +
        theme_minimal() +
        theme(figure_size=(1.*width_scale, 
                           0.4*df[yaxis].nunique()*height_scale),
              axis_text_y=element_blank(),
              legend_position=legend_position) +  
        labs(x=None, y='Cell counts', fill='Cell counts') +
        geom_text(aes(label='ncell', y=0.5),
                  color='white', size=8, fontweight='bold',
                  path_effects=(pe.Stroke(linewidth=1, foreground='black'), pe.Normal())) + 
        scale_fill_continuous(trans='log10', cmap_name='magma')
    )

    return g


def plot_proportion_barplot_single_categorical(
    adata,
    yaxis,
    fill,
    height_scale=1., 
    width_scale=1.,
    legend_position=None,
):

    import mizani
    import matplotlib.patheffects as pe
    
    adata._sanitize()

    fill_dict = {k:v for k,v in zip(adata.obs[fill].cat.categories, adata.uns[f'{fill}_colors'])}
    
    df = adata.obs[[yaxis, fill]].drop_duplicates().reset_index()
    df['counts'] = 1
    df[yaxis] = pd.Categorical(df[yaxis], categories=reversed(adata.obs[yaxis].cat.categories))    

    g = (
        ggplot(aes(x=yaxis, y='counts', fill=fill), data=df) +
        geom_col() +
        coord_flip() +
        theme_minimal() +
        theme(figure_size=(1.*width_scale, 
                           0.4*df[yaxis].nunique()*height_scale),
              axis_text_y=element_blank(),
              legend_position=legend_position) +  
        labs(x=None, y=fill, fill=fill) +
        scale_fill_manual(values=fill_dict)
    )

    return g


def merge_ggplots(*plots, units, figsize):
    
    # Empty plotnine figure to place the subplots on. Needs junk data (for backend "copy" reasons).
    fig = (ggplot()+geom_blank(data=ddata)+theme_void() + theme(figure_size=figsize)).draw()

    # Create gridspec for adding subpanels to the blank figure
    gs = gridspec.GridSpec(1,np.sum(units))
    prev = 0
    for p, u in zip(plots, np.cumsum(units)):
        ax = fig.add_subplot(gs[0, prev:u])
        prev = u
        _ = p._draw_using_figure(fig, [ax])

    return fig


def plot_proportion_barplot_with_ncells(
    adata,
    yaxis,
    fill, 
    fill_breakdown=None,
    yaxis_label=None,
    fill_label=None,
    percent_limit=5., 
    show_percent=True,
    height_scale=1., 
    width_scale=1.,
    legend_position=(-0.2, 0.5),
    normalize_by=None,
):
    
    g1, df = plot_proportion_barplot(
        adata,
        yaxis,
        fill, 
        fill_breakdown=fill_breakdown,
        yaxis_label=yaxis_label,
        fill_label=fill_label,
        percent_limit=percent_limit, 
        show_percent=show_percent,
        height_scale=height_scale, 
        width_scale=width_scale,
        legend_position=legend_position,
        return_df=True,
        normalize_by=normalize_by,
    )
    
    g2 = plot_proportion_barplot_cellcounts(adata, yaxis)
    figsize = (8*width_scale*1.5, 0.4*df[yaxis].nunique()*height_scale)
    
    return merge_ggplots(g1, g2, units=[9, 1], figsize=figsize)


def plot_proportions(adata, sample_key, proportion_key, covariates, fill, return_input_df=False, kind='boxplot', dotplot_binwidth=0.001, width_scale=1., height_scale=1.):

    adata._sanitize()
    p, c = get_proportions_per_channel(adata, sample_key, proportion_key, covariates)
    dr_df = pd.concat([p, c], axis=1)

    proportion_df = dr_df.reset_index().melt(id_vars=[sample_key] + covariates,
                                             value_vars=adata.obs[proportion_key].cat.categories,
                                             var_name='categorical',
                                             value_name='proportion').set_index(sample_key)

    proportion_df['categorical'] = pd.Categorical(proportion_df['categorical'], categories=adata.obs[proportion_key].cat.categories)
    color_dict = {k:v for k,v in zip(adata.obs[fill].cat.categories, adata.uns[f'{fill}_colors'])}

    if kind == 'dotplot':
        geom = geom_dotplot(position='dodge', binaxis = "y", stackdir = "center", binwidth = dotplot_binwidth)
    else:
        geom = geom_boxplot()

    g = (
        ggplot(proportion_df, aes(x='categorical', y='proportion', fill=fill)) +
        geom +
        scale_fill_manual(values=color_dict) +
        labs(y='Proportions', x='', fill=fill) +
        theme_classic() +
        theme(figure_size=(8*width_scale,6*height_scale), axis_text_x = element_text(angle = 45, hjust=1.))
    )

    if return_input_df:
        return g, proportion_df
    else:
        return g
