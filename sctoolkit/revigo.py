import pandas as pd

# taken mostly from https://github.com/alexvpickering/revigoR/blob/master/inst/python/scrape_revigo.py
def revigo(
    goterms,
    pvals=None,
    url='http://revigo.irb.hr',
    cutoff='small',
    isPValue='yes',
    whatIsBetter='higher',
    goSizes='0',
    measure='SIMREL',
    run_twice=True,
):

    import re
    from io import StringIO
    import pandas as pd

    import werkzeug
    werkzeug.cached_property = werkzeug.utils.cached_property
    from robobrowser import RoboBrowser

    br = RoboBrowser(parser="lxml")
    br.open(url)
    form = br.get_form()

    def _rev(c):
        if pvals is None:
            golist = '\n'.join(goterms)
        else:
            golist = '\n'.join([f'{go} {pval}' for go,pval in zip(goterms, pvals)])
        form['goList'].value = golist

        cd = {'large': '0.90', 'medium': '0.70', 'small': '0.50', 'tiny': '0.40'}
        form['cutoff'].value = cd[c]

        form['isPValue'].value = isPValue
        form['whatIsBetter'].value = whatIsBetter
        form['goSizes'].value = str(goSizes)
        form['measure'].value = measure

        br.submit_form(form)
        download_csv_link = br.find("a", href=re.compile("export.jsp"))
        br.follow_link(download_csv_link)
        csv_content = br.response.content.decode("utf-8")

        df = pd.read_csv(StringIO(csv_content))
        df['neglog10'] = -df['log10 p-value']
        df['frequency'] = [float(x[:-1]) for x in df['frequency']]

        return df

    if run_twice and cutoff != 'large':
        rev1 = _rev(c=cutoff)
        rev2 = _rev(c='large')

        rev = rev1.drop(columns=['plot_X', 'plot_Y']).merge(rev2[['term_ID', 'plot_X', 'plot_Y']], on='term_ID')

        return rev

    return _rev(c=cutoff)


def _test_revigo(**kwds):
    golist = '''GO:0009268 1e-14
GO:0010447 1e-14
GO:0000027 1e-297
GO:0042255 1e-297
GO:0042257 1e-297
GO:0042273 1e-297'''

    go = [x.split()[0] for x in golist.split('\n')]
    pvals = [float(x.split()[1]) for x in golist.split('\n')]

    df = revigo(go, pvals, **kwds)

    df2 = revigo(go)

    return df


def plot_revigo(
    rev, 
    outline=2,
    expand_points=(1.05, 1.2), 
    figure_size=(8,8),
    font_size=8,
    point_size=3,
    point_alpha=0.7,
    palette='RdPu',
    dispensability_cutoff=1.,
    show_all_labels=False,
):

    import plotnine as p9
    import matplotlib.patheffects as path_effects
    
    pe = [path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()]
    lbl_df = rev[(rev.eliminated==0) & (rev.dispensability < dispensability_cutoff)] if not show_all_labels else rev
    g = (
        p9.ggplot(p9.aes(x='plot_X', y='plot_Y'), data=rev) + 
        p9.geom_point(p9.aes(fill='neglog10', size='frequency'), color='black', alpha=point_alpha) + 
        p9.geom_text(p9.aes(label='description'), data=lbl_df, size=font_size,
                     adjust_text={'expand_points': expand_points, 'arrowprops': {'arrowstyle': '-'}, 'x':rev.plot_X.values, 'y':rev.plot_Y.values},
                     path_effects=pe) + 
        p9.theme_bw() + 
        p9.scale_fill_distiller(type='seq', palette=palette, direction=1) +
        p9.labs(x='Semantic similarity space', y='', fill='-log10(adj. p-value)', size='Term frequency') + 
        p9.scale_size_continuous(range=(2, 7), trans='log10') +
        p9.theme(figure_size=figure_size, 
                 axis_text_x=p9.element_blank(),
                 axis_text_y=p9.element_blank(),
                 axis_ticks=p9.element_blank())
    )
    
    return g


def enrich_and_simplify(
    sets,
    intersections=True,
    sources=('GO:BP',),
    organism='hsapiens',
    reduce_limit=0,
    **revigo_kwds
):
    from gprofiler import GProfiler

    gprofiler = GProfiler(user_agent="scanpy", return_dataframe=True)
    gprofiler_kwargs = {'no_evidences': not intersections, 'sources':sources}

    df = gprofiler.profile(sets, organism=organism, **gprofiler_kwargs)
    revs = {}

    if reduce_limit is not None:
        dfs = []
        for q in df['query'].unique():
            df_sub = df[df['query'] == q].copy()
            go = df_sub.native.tolist()
            pvals = df_sub.p_value.tolist()

            if len(go) > reduce_limit:
                r = revigo(go, pvals, **revigo_kwds)
                revs[q] = r

                r = r.rename(columns={'term_ID': 'native'}).drop(columns='description').assign(query=q)
                dfs.append(df_sub.merge(r))
            else:
                dfs.append(df.assign(eliminated=0))

        df = pd.concat(dfs, axis=0).reset_index(drop=True)

    return df, revs
