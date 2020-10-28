import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import urllib.request
import matplotlib.pyplot as plt
from pyannotables import tables
from plotnine import *

from .rtools import py2r, r2py, r_is_installed, r_set_seed, rpy2_check

from rpy2.robjects import r, StrVector
from rpy2.robjects.packages import importr
import rpy2


@rpy2_check
def mgsa(observed, sets, seed=0, alpha=None, beta=None, p=None, report_intersection=True, **kwds):
    r_is_installed('mgsa')
    mgsa = importr('mgsa')
    
    r_set_seed(seed)

    if alpha is not None:
        alpha = py2r(np.array(alpha))
        kwds['alpha'] = alpha

    if beta is not None:
        beta = py2r(np.array(beta))
        kwds['beta'] = beta

    if p is not None:
        p = py2r(np.array(p))
        kwds['p'] = p        
        
    if isinstance(sets, dict):
        set_names = r('`names<-`')

        l = [py2r(np.array(x)) for x in sets.values()]
        l = set_names(l, list(sets.keys()))
    
        sets = r('new')('MgsaSets', sets=l)

    elif isinstance(sets, rpy2.robjects.methods.RS4):
        assert r2py(r('class')(sets))[0] == 'MgsaGoSets'
    else:
        raise Exception('invalid set type')

    unmapped = np.array(observed)[~np.isin(observed, r2py(r('names')(sets.slots['itemName2ItemIndex'])))]
    if len(unmapped) > 0:
        print(f'{len(unmapped)} observation(s) could not be mapped... ({unmapped})')
    
    res_obj = mgsa.mgsa(StrVector(observed), sets, **kwds)
    res = {
        'sets': r2py(res_obj.slots['setsResults']).sort_values('estimate', ascending=False).rename(columns={'std.error': 'se'}),
        'alpha': r2py(res_obj.slots['alphaPost']).rename(columns={'std.error': 'se'}),
        'beta': r2py(res_obj.slots['betaPost']).rename(columns={'std.error': 'se'}),
        'p': r2py(res_obj.slots['pPost']).rename(columns={'std.error': 'se'}),
    }
    
    if report_intersection:
        term2index = {k: r2py(v) for k,v in zip(r2py(r('names')(sets.slots['sets'])), sets.slots['sets'])}
        all_symbols = r2py(r('rownames')(sets.slots['itemAnnotations']))
        term2index = {k: all_symbols[v-1].tolist() for k, v in term2index.items()}

        res['sets']['intersection'] = [sorted(list(set(term2index[x]) & set(observed))) for x in res['sets'].index]

    return res


def get_go_gaf(organism='human', evidence=None, aspect=('P', 'F', 'C'), uniprot2symbol=True, return_df=False):
    
    gaf_url = f'ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/{organism.upper()}/goa_{organism}.gaf.gz'
    if evidence is None:
        evidence = rpy2.rinterface.NULL
    else:
        evidence = StrVector(evidence)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        full_path = Path(tmpdir) / 'file.gaf.gz'
        urllib.request.urlretrieve(gaf_url, full_path)
        
        mgsa_r = importr('mgsa')
        go_sets = mgsa_r.readGAF(str(full_path), evidence=evidence, aspect=StrVector(aspect))

    if uniprot2symbol:
        if organism == 'human':
            uni2gene = tables['homo_sapiens-GRCh38-ensembl100'].copy()
            uni2gene = uni2gene[~uni2gene['UniProtKB_Accession'].isnull()]
            uni2gene = uni2gene.set_index('UniProtKB_Accession')[['gene_name']].to_dict()['gene_name']
            
        elif organism == 'mouse':
            uni2gene = tables['mus_musculus-GRCm38-ensembl100'].copy()
            uni2gene = uni2gene[~uni2gene['UniProtKB_Accession'].isnull()]
            uni2gene = uni2gene.set_index('UniProtKB_Accession')[['gene_name']].to_dict()['gene_name']
            
        else:
            raise Exception(f'no uniprot2gene conversion for {organism}')
        
        # use gene symbols instead of uniprot
        go_uni_ids = r2py(r('names')(go_sets.slots['itemName2ItemIndex']))
        uniq_symbols = r('make.unique')(StrVector([uni2gene.get(x,x) for x in go_uni_ids]))
        set_names = r('`names<-`')
        go_sets.slots['itemName2ItemIndex'] = set_names(go_sets.slots['itemName2ItemIndex'], uniq_symbols)
        set_row_names = r('`rownames<-`')        
        go_sets.slots['itemAnnotations'] = set_row_names(go_sets.slots['itemAnnotations'], uniq_symbols)

    if return_df:
        term2index = {k: r2py(v) for k,v in zip(r2py(r('names')(go_sets.slots['sets'])), go_sets.slots['sets'])}
        all_symbols = r2py(r('rownames')(go_sets.slots['itemAnnotations']))
        term2index = {k: all_symbols[v-1].tolist() for k, v in term2index.items()}
        
        df = r2py(go_sets.slots['setAnnotations'])
        df['genes'] = term2index.values()
        
        return go_sets, df

    return go_sets


def plot_mgsa_diagnostics(res, figure_size = (15,3), x_spacing=1.):

    df = pd.concat([res[x].assign(var=x) for x in ('p', 'beta', 'alpha')], axis=0)

    g = (
        ggplot(aes(x='value', y='estimate'), df) +
        geom_path(color='gray') +
        geom_point(size=2) +
        geom_pointrange(aes(ymin='estimate-se', ymax='estimate+se')) +
        facet_wrap('var', nrow=1, scales='free') +
        theme_minimal() +
        labs(x='', y='Posterior probability') +
        theme(figure_size=figure_size, panel_spacing_x=x_spacing)
    )

    return g


def plot_mgsa(res, top=20, textwidth=40, x='term', figure_size=(5, 12), n_max_genes=5):
    import textwrap

    df = res['sets'].reset_index().copy().head(top)
    cap_keep = lambda s: s[:1].upper() + s[1:]
    df[x] = [cap_keep(s) for s in df[x]]
    if 'intersection' in df.columns:
        df[x] = df[x].astype(str) + ' (' + [','.join(i[:n_max_genes]) for i in df['intersection']] + ')'

    if textwidth:
        df[x] = [textwrap.fill(t, textwidth) for t in df[x]]

    g = (
        ggplot(aes(x=f'reorder({x}, estimate)', y='estimate'), df) +
        geom_pointrange(aes(ymin='estimate-se', ymax='estimate+se')) +
        geom_point(aes(size='inStudySet', fill='inPopulation')) +
        theme_minimal() +
        theme(figure_size=figure_size) +
        scale_fill_distiller(palette='Reds', direction=1, trans='log10') +
        labs(x='Pathway', y='Posterior probability', size='Intersection', fill='Term size') +
        coord_flip()
    )

    return g
