
import pandas as pd
from plotnine import *

def plot_topic(topic, n_genes=15):
    df = pd.DataFrame(topic[:n_genes], columns=['Gene', "Prob"])
    df.Gene = df.Gene.astype('category').cat.reorder_categories(df.Gene, ordered=True )
    return qplot('Gene', 'Prob', data=df) + theme_minimal() +  theme(axis_text_x=element_text(rotation=90, hjust=1))


def plot_topics(topics, figsize=(10, 4), scale='free', highlight=None, ncols=10, panel_spacing_x=1., panel_spacing_y=1., x_label_map=None, **kwargs):
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
                      figure_size=figsize, **kwargs))

            
    else:
        df['Gene'] = df.Gene.astype('category').cat.reorder_categories(df.Gene, ordered=True)
        x_reversed = pd.CategoricalDtype(categories=reversed(df['Gene'].cat.categories), ordered=True)
        df['Gene'] = df.Gene.astype(x_reversed)
        
        g = (
            qplot('Gene', 'Prob', data=df) + 
            facet_wrap('Topic', scales=scale, ncol=ncols) + 
            theme_minimal() + 
            theme(figure_size=figsize, **kwargs) +                 #axis_text_x=element_text(rotation=90, hjust=1., size=7),
            coord_flip()
            )
        
        if x_label_map:
            g += scale_x_discrete(labels=x_label_map)
            
        return g