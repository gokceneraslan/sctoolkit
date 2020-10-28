import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import urllib.request
import matplotlib.pyplot as plt
from pyannotables import tables

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
        'sets': r2py(res_obj.slots['setsResults']).sort_values('estimate', ascending=False),
        'alpha': r2py(res_obj.slots['alphaPost']),
        'beta': r2py(res_obj.slots['betaPost']),
        'p': r2py(res_obj.slots['pPost']),
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


def plot_mgsa_diagnostics(res):
    f, axs = plt.subplots(1, 3, figsize=(20,5))    
    res['p'].plot(kind='scatter', x='value', y='estimate', ax=axs[0], title='p posterior', color='#990E1D')
    res['alpha'].plot(kind='scatter', x='value', y='estimate', ax=axs[1], title='alpha posterior', color='#990E1D')
    res['beta'].plot(kind='scatter', x='value', y='estimate', ax=axs[2], title='beta posterior', color='#990E1D')

    res['p'].plot(kind='line', x='value', y='estimate', ax=axs[0], color='#808080', legend=False)
    res['alpha'].plot(kind='line', x='value', y='estimate', ax=axs[1], color='#808080', legend=False)
    res['beta'].plot(kind='line', x='value', y='estimate', ax=axs[2], color='#808080', legend=False)
    
    return f


def plot_mgsa(res, top=10, x='term'):
    df = res['sets'].head(top).reset_index()
    ax = df.plot(kind='scatter', x=x, y='estimate', figsize=(8,5), rot=90, color='#990E1D', title='Set posteriors')
    ax = df.plot(kind='line', x=x, y='estimate', rot=90, color='#808080', ax=ax, legend=False)
    
    return ax
