from .rtools import r2py, py2r

import logging
import numpy as np
import pandas as pd
from plotnine import *



def get_proportions_per_channel(adata, sample_key, proportion_key, covariates):

    prop_df = pd.DataFrame(adata.obs.groupby([sample_key, proportion_key]).size(), columns=['ncell']).reset_index()

    prop_df = prop_df.pivot(index=sample_key, columns=proportion_key, values='ncell').fillna(0)
    prop_df.columns.name = None
    prop_df.columns = prop_df.columns.astype(str)
    prop_df /= prop_df.sum(1).values[:, None]
    prop_df.index = prop_df.index.astype(str)

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


def plot_proportion_barplot(adata, first, second, first_label, second_label, height_scale=1., width_scale=1.):

    import mizani
    import matplotlib.patheffects as pe

    df = pd.DataFrame(adata.obs.groupby([first, second], observed=True).size(), columns=['counts']).reset_index()

    df[second] = df[second].astype(str)
    df = df.pivot_table(index=first, columns=second, values='counts')
    df = ((df.T / df.sum(1)).T).reset_index()
    df = df.melt(id_vars=first, value_name='counts')

    if adata.obs[first].dtype.name == 'category':
        df[first]  = pd.Categorical(df[first], categories=reversed(adata.obs[first].cat.categories))
    else:
        df[first]  = pd.Categorical(df[first], categories=reversed(sorted(df[first].unique())))
        
    if adata.obs[second].dtype.name == 'category':
        df[second]  = pd.Categorical(df[second], categories=reversed(adata.obs[second].cat.categories))
    else:
        df[second]  = pd.Categorical(df[second], categories=reversed(sorted(df[second].unique())))


    df['cumsum'] = df.groupby(first, observed=True)['counts'].transform(pd.Series.cumsum)
    df['cumsum_mean'] = df['cumsum'] - df['counts'] + (df['counts']/2)

    cols = {k:v for k,v in zip(adata.obs[second].cat.categories, adata.uns[f'{second}_colors'])}

    g = (
        ggplot(aes(x=first, y='counts', fill=second), data=df) +
        geom_bar(position='fill', stat='identity') +
        geom_text(aes(label='round(counts*100).astype(int)', y='cumsum_mean'), data=df[df.counts>0.03],
                  color='white', size=8, fontweight='bold',
                  path_effects=(pe.Stroke(linewidth=1, foreground='black'), pe.Normal())) +
        scale_y_continuous(labels=mizani.formatters.percent) +
        coord_flip() +
        theme_minimal() +
        theme(figure_size=(8*width_scale,
                           0.4*df[first].nunique()*height_scale)) + scale_fill_manual(cols) +
        labs(x=first_label, y=second_label) +
        guides(fill = guide_legend(reverse=True))
    )

    return g


def plot_proportion_dotplot(adata, sample_key, proportion_key, covariates, fill):

    adata._sanitize()
    dr_df = get_proportions_per_channel(sample_key, proportion_key, covariates)

    proportion_df = dr_df.reset_index().melt(id_vars=[sample_key] + covariates,
                                             value_vars=adata.obs[proportion_key].cat.categories,
                                             var_name='categorical',
                                             value_name='proportion').set_index(sample_key)

    proportion_df['categorical'] = pd.Categorical(proportion_df['categorical'], categories=adata.obs[proportion_key].cat.categories)

    g = (
        ggplot(proportion_df, aes(x='categorical', y='proportion', fill=fill)) +
        geom_dotplot(position='dodge', binaxis = "y", stackdir = "center", binwidth = 0.001) +
        scale_fill_manual(values=['#4F9B6C', '#3C75AF',  '#EF8636']) +
        labs(y='Proportions', x='', fill=fill) +
        theme_classic() +
        theme(figure_size=(30,6), axis_text_x = element_text(angle = 90, hjust = 1))
    )

    return g
