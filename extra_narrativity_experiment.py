# %%

# this is to analyze the relationship between narrativity scores and imageability/visuality/concreteness
# it's like an extra analysis to show the usefullness of the semantic concepts

import pandas as pd
# %%
df = pd.read_csv('/Users/au324704/Downloads/annotated_dataset_401.csv')
df.head()
# %%
# df = df[['genre','avg_overall', 'text', 'avg_agency']]
# df.head()
# %%

import re

def clean_whitespace(text: str) -> str:
    # check if text is a string
    if not isinstance(text, str):
        logger.warning(f"Expected string, got {type(text)}. Returning original text.")
        return text
    # rm newline characters
    text = text.replace('\n', ' ')
    # multiple spaces -> single space
    text = re.sub(r'\s+', ' ', text)
    # rm spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # rm excess spaces after punctuation (.,!? etc.)
    text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
    # leading and trailing spaces
    text = text.strip()
    return text

# apply clean whitespace
df['text'] = df['text'].apply(clean_whitespace)

# %%
from functions import get_dict_scores
# get dict scores

sample_texts = df['text'].tolist()

df['imageability'], df['norm_imageability'] = get_dict_scores(sample_texts, 'resources/mrc_psychol_dict.json', 'imag', 'lemma_', normalize_by_tokens=False)
df['visuality'], df['norm_visuality'] = get_dict_scores(sample_texts, 'resources/sensorimotor_norms_dict.json', 'Visual.mean', 'lemma_', normalize_by_tokens=False)
df['concreteness'], df['norm_concreteness'] = get_dict_scores(sample_texts, 'resources/concreteness_brysbaert.json', None, 'lemma_', normalize_by_tokens=False)
df.head()
# %%

# correlate these with avg_overall
import seaborn as sns
import matplotlib.pyplot as plt

# spearman correlation
from scipy.stats import spearmanr

norm_cols = ['norm_imageability', 'norm_visuality', 'norm_concreteness']
cols = ['imageability', 'visuality', 'concreteness']

for col in norm_cols:
    corr, p_value = spearmanr(df['avg_event_seq'], df[col])
    print(f"Spearman correlation between avg_overall and {col}: {corr:.3f}, p-value: {p_value:.3f}")
    # plot it
    sns.set_style("whitegrid")
    sns.scatterplot(x=df['avg_overall'], y=df[col], alpha=0.5, s=50)
    plt.title(f"Avg. narrativity and {col}: {corr:.3f}, p-value: {['**' if p_value < 0.01 else ''][0]}")
    plt.xlabel('Avg. Narrativity')
    plt.ylabel(col.replace('_', ' ').title())
    plt.tight_layout()
    plt.show()

# %% # %%
