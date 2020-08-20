import scanpy as sc
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# R integration

from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri, numpy2ri, r, Formula
from rpy2.robjects.vectors import StrVector, FloatVector, ListVector
import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError


def fit_lme(formula, df, family='gaussian', optimizer='nloptwrap', random_effect=True, **fit_kwargs):
    f = Formula(formula)
    
    lme4 = importr('lme4')
    lmer = importr('lmerTest') # overloads lmer function from lme4 package
    base = importr('base')
    stats = importr('stats')

    with localconverter(ro.default_converter + pandas2ri.converter):
        if family == 'gaussian':
            if random_effect:
                control = lme4.lmerControl(**{'calc.derivs': True,
                                              'check.rankX': 'silent.drop.cols', 
                                              'check.conv.singular': r('lme4::.makeCC')(action = "ignore",  tol = 1e-4)})
                fit = lmer.lmer(f, df, control=control, **fit_kwargs)
            else:
                fit = stats.lm(f, df, **fit_kwargs)
        elif family in ('binomial', 'poisson'):
            if random_effect:
                if optimizer == 'nloptwrap':
                    control = lme4.glmerControl(**{'optimizer': 'nloptwrap',
                                                   'calc.derivs': True,
                                                   'check.rankX': 'silent.drop.cols',
                                                   'check.conv.singular': r('lme4::.makeCC')(action = "ignore",  tol = 1e-4)})
                else:
                    control = lme4.glmerControl(**{'check.rankX': 'silent.drop.cols',
                                                   'check.conv.singular': r('lme4::.makeCC')(action = "ignore",  tol = 1e-4)})

                fit = lme4.glmer(f, df, control=control, family=family, **fit_kwargs)

            else:
                fit = stats.glm(f, df, family=family, **fit_kwargs)
        else:
            if random_effect:
                if optimizer == 'nloptwrap':
                    control = lme4.glmerControl(**{'optimizer': 'nloptwrap',
                                       'calc.derivs': True,
                                       'check.rankX': 'silent.drop.cols',
                                       'check.conv.singular': r('lme4::.makeCC')(action = "ignore",  tol = 1e-4)})
                    fit = r('lme4::glmer.nb')(f, df, **{'nb.control': control}, **fit_kwargs)
                else:
                    fit = r('lme4::glmer.nb')(f, df, **fit_kwargs)
            else:
                fit = r('MASS::glm.nb')(f, df, **fit_kwargs)
        
        anova_df = stats.anova(fit)
    
    coef_df = r['as.data.frame'](stats.coef(base.summary(fit)))
    coef_df = pandas2ri.rpy2py(coef_df)

    return coef_df, anova_df


def _fit(formula, gene, adata, obs_features, use_raw, family, random_effect):
    gene_vec = adata[:, gene].X if not use_raw else adata.raw[:, gene].X
    covariates = adata.obs[obs_features].copy()
    covariates.loc[:, 'gene'] = gene_vec.A.squeeze()

    try:
        coefs, anova = fit_lme(formula, covariates, family=family, random_effect=random_effect)
    except RRuntimeError:
        try:
            coefs, anova = fit_lme(formula, covariates, family=family, random_effect=random_effect, optimizer='default')
        except RRuntimeError:
            print(f'Exception in R... ({gene})')
            coefs, anova = None, None

    return coefs, anova


def fit_lme_adata(adata, genes, formula, obs_features, random_effect, family='gaussian', use_raw=False, n_jobs=4, parallel=None):

    adata = adata.copy()
    for f in obs_features:
        if adata.obs[f].dtype.name == 'category' and len(adata.obs[f].cat.categories) < 2:
            print(f'\tRemoving {f}... ({adata.obs[f].cat.categories.to_list()})')
            adata.obs[f] = adata.obs[f].cat.codes
            if f == random_effect:
                random_effect = False

    if n_jobs == 1:
        para_result = [_fit(formula, x, adata, obs_features, use_raw, family,random_effect) for x in tqdm(genes)]
    else:    
        if parallel is None:
            parallel = Parallel(n_jobs=n_jobs) 
        
        para_result = parallel(delayed(_fit)(formula,
                                             x,
                                             adata,
                                             obs_features,
                                             use_raw,
                                             family,
                                             random_effect) for x in tqdm(genes))
    
    coef_df = {k:v[0] for k, v in zip(genes, para_result) if v[0] is not None}
    anova_df = {k:v[1] for k, v in zip(genes, para_result) if v[1] is not None}   

    coef_df = pd.concat([df.assign(gene=gene) for gene, df in coef_df.items()], axis=0)
    coef_df = coef_df.reset_index().rename(columns={'index': 'fixed_effect'})

    anova_df = pd.concat([df.assign(gene=gene) for gene, df in anova_df.items()], axis=0)
    anova_df = anova_df.reset_index().rename(columns={'index': 'fixed_effect'})
   
    return coef_df, anova_df


def fit_coexpression_model(adata, label, celltype_key, anchor_gene, sample_key, subsample_ref=True, n_jobs=20):
    from tqdm import tqdm

    adata = adata.copy()
    adata.X = (adata.X>0).astype(int) # binarize for binomial GLMM
    adata.obs[f'bin{anchor_gene}'] = adata.obs_vector(anchor_gene)
    
    res = []
    
    ct = (adata.obs.groupby(celltype_key).size() >= 5) & (adata.obs.groupby(celltype_key)[f'bin{anchor_gene}'].sum() >= 5)
    ct = ct[ct].index.to_list()
    
    if not ct:
        print(f'!!! Dataset {label} No available cell types...')
        return

    for celltype in tqdm(ct):
        ad = adata[(adata.obs[celltype_key] == celltype)].copy()

        if subsample_ref:
            ad_pos = ad[ad.obs[f'bin{anchor_gene}'] == 1]
            ad_neg = sc.pp.subsample(ad[ad.obs[f'bin{anchor_gene}'] == 0], n_obs=len(ad_pos), copy=True)
            ad = ad_pos.concatenate(ad_neg)        
        
        genes = ad.var_names.difference([anchor_gene]).to_list()
        genes = (len(ad) > sc.get.obs_df(ad, genes).sum(0)) & (sc.get.obs_df(ad, genes).sum(0) >= 5)
        genes = genes[genes].index.to_list()
        
        if len(genes) == 0:
            continue
            
        print(f'*** Dataset: {label:>60} Cell type: {celltype:>40} # genes: {len(genes):>5} # cells: {len(ad):>6} ***')
        ad.obs[sample_key] = ad.obs[sample_key].astype('category')
        
        coef, anova = fit_lme_adata(ad, 
                                    genes, 
                                    f'gene ~ 1 + bin{anchor_gene} + (1|{sample_key})',
                                    [f'bin{anchor_gene}', sample_key],
                                    family='binomial',
                                    random_effect = sample_key,
                                    n_jobs=min(n_jobs, len(genes)))

        sig, pval, _, _ = multipletests(coef['Pr(>|z|)'], method='fdr_bh', alpha=0.1)
        coef['significant'] = sig
        coef['pval_adj'] = pval
        coef['neglog_pval_adj'] = -np.log10(coef.pval_adj+1e-300)
        res.append(coef.assign(celltype=celltype))
        
    if not res:
        return None

    res = pd.concat(res, axis=0)
    res = res[res.fixed_effect == f'bin{anchor_gene}'].sort_values('pval_adj').reset_index(drop=True).assign(dataset=label)
    
    return res