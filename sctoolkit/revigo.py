# taken mostly from https://github.com/alexvpickering/revigoR/blob/master/inst/python/scrape_revigo.py
def revigo(
    goterms,
    pvals=None,
    url='http://revigo.irb.hr',
    cutoff='medium',
    isPValue='yes',
    whatIsBetter='higher',
    goSizes='0',
    measure='SIMREL',
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

    if pvals is None:
        golist = '\n'.join(goterms)
    else:
        golist = '\n'.join([f'{go} {pval}' for go,pval in zip(goterms, pvals)])
    form['goList'].value = golist

    cd = {'large': '0.90', 'medium': '0.70', 'small': '0.50', 'tiny': '0.40'}
    form['cutoff'].value = cd[cutoff]

    form['isPValue'].value = isPValue
    form['whatIsBetter'].value = whatIsBetter
    form['goSizes'].value = str(goSizes)
    form['measure'].value = measure

    br.submit_form(form)
    download_csv_link = br.find("a", href=re.compile("export.jsp"))
    br.follow_link(download_csv_link)
    csv_content = br.response.content.decode("utf-8")

    df = pd.read_csv(StringIO(csv_content))

    return df


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
