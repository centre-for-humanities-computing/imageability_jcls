# %%
import numpy as np
import json
import os

from scipy.special import softmax
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from datasets import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from functions import generate_clip_embeddings
from functions import get_dict_scores

from transformers import CLIPProcessor, CLIPModel
import torch

model_name = "openai/clip-vit-base-patch32"

# # correlations on the dictionary entries
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

### Last experiment, here we compare imagist to modern love poetry

# %%
# entropy
def calculate_entropy(embedding):
    # Convert the embedding values to probabilities using softmax
    probabilities = softmax(embedding)
    # Calculate the entropy
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))  # Adding a small constant to avoid log(0)
    return entropy


# sparse ratio
# Function to compute the ratio of L1 to L2 norms
def calculate_sparsity_ratio(embedding):
    embedding = np.array(embedding)
    l1_norm = np.sum(np.abs(embedding))  # L1 norm
    l2_norm = np.sqrt(np.sum(embedding**2))  # L2 norm
    sparsity_ratio = l1_norm / (np.sqrt(len(embedding)) * l2_norm)
    return sparsity_ratio

# %%

# CONFIGS

# we can do the analysis on a poem basis or a sentence basis (default, sentences)
analysis_basis = "sentences"
print(f"Analysis basis: {analysis_basis}")

# %%

# open the imagists
with open('data/some_imagists.json', 'r') as f:
    imagists = json.load(f)

poems = []
poem_texts = []
for poet, works in imagists.items():
    for work in works:
        work['poet'] = poet  # Add poet's name to each poem
        poems.append(work)
        poem_texts.append(work['poem'])

# get len
print('no of poems', len(poems))

# clean it

# use poems
if analysis_basis == "poems":
    units = poem_texts
    units_clean = []
    for poem in poems:
        # we want to remove extra spaces
        poem = poem['poem'].replace('\n', ' ').replace('  ', ' ')
        # the stars we put to seperate the lines
        poem = poem.strip().replace('*', '')
        # remove leading and trailing spaces
        poem = poem.strip()
        units_clean.append(poem)
# use sentences
elif analysis_basis == "sentences":
    units = [sentence for poem in poem_texts for sentence in poem.split('*')]
    units_clean = []
    for poem in poems:
        # we want to remove extra spaces
        poem = poem['poem'].replace('\n', ' ').replace('  ', ' ')
        poem = poem.strip().split('*')
        # remove leading and trailing spaces
        poem = [sentence.strip() for sentence in poem]
        units_clean.extend(poem)

else:
    raise ValueError("Invalid analysis_basis")

# remove emtpy strings
units_clean = [unit for unit in units_clean if unit != '']
# remove sentences that are just a number
units_clean = [unit for unit in units_clean if not unit.isdigit()]
print(f'no of {analysis_basis.upper()}', len(units_clean))

# get a list of the poets
poets = list(imagists.keys())
print('no of poets', len(poets))

# %%

# now we get the embeddings for each sentence

# check if embeddings are already saved
if os.path.exists(f'data/embeddings/imagists_{analysis_basis}.json'):
    with open(f'data/embeddings/imagists_{analysis_basis}.json', 'r') as f:
        sentence_embeddings = np.array(json.load(f))
    print('loaded')
else:
    sentence_embeddings = generate_clip_embeddings(units_clean)
    with open(f'data/embeddings/imagists_{analysis_basis}.json', 'w') as f:
        json.dump(sentence_embeddings.tolist(), f)

norms = np.linalg.norm(sentence_embeddings, axis=1)
entropies = [calculate_entropy(embedding) for embedding in sentence_embeddings]
variances = np.var(sentence_embeddings, axis=1)
sparsities = [calculate_sparsity_ratio(embedding) for embedding in sentence_embeddings]

# check if the scores are already saved
if os.path.exists(f'data/measures/imagists_{analysis_basis}.json'):
    df_sentences = pd.read_json(f'data/measures/imagists_{analysis_basis}.json')
    print('loaded')
else:
    # get the imageability scores
    imag, imag_norm = get_dict_scores(units_clean, 'resources/mrc_psychol_dict.json', 'imag', 'lemma_', normalize_by_tokens=False)
    vis, vis_norm = get_dict_scores(units_clean, 'resources/sensorimotor_norms_dict.json', 'Visual.mean', 'lemma_', normalize_by_tokens=False)
    conc, conc_norm = get_dict_scores(units_clean, 'resources/concreteness_brysbaert.json', 'Conc.M', 'lemma_', normalize_by_tokens=False)

    # make a dataframe
    df_sentences = pd.DataFrame({'norm': norms, 'entropy': entropies, 'variance': variances, 'sparsity': sparsities})
    df_sentences['imag'] = imag
    df_sentences['visual'] = vis
    df_sentences['concrete'] = conc
    df_sentences['imag_norm'] = imag_norm
    df_sentences['visual_norm'] = vis_norm
    df_sentences['concrete_norm'] = conc_norm

    # save the measures
    df_sentences.to_json(f'data/measures/imagists_{analysis_basis}.json', orient='records')

# add the text
df_sentences['text'] = units_clean

# %%
# and we get the poetry from:
# https://huggingface.co/datasets/merve/poetry
ds = load_dataset("merve/poetry")

# make df
df = pd.DataFrame(ds['train'])
print(len(df))
df.head()

# %%
# # take only the modern
df_modern = df[df['age'] == 'Modern']
# take only the poems where type is Love
df_love = df_modern[df_modern['type'] == 'Love']
print('len df', len(df_love))
# remove any others that are in the imagist poets
df_love = df_love[~df_love['author'].isin(poets)]
print('len df after filtering authors', len(df_love))

# more cleaning needed from these cause they are scraped data

# take poems
if analysis_basis == "poems":
    units = df_love['content'].tolist()
    units_clean_merve = []
    for poem in units:
        # we want to remove extra spaces
        poem = poem.replace('\n', ' ').replace('  ', ' ')
        poem = poem.strip().replace('*', '')
        # remove leading and trailing spaces
        poem = poem.strip()
        # remove sentences in the poems with copyright or reprint
        poem = poem.split('\n')
        poem = [sentence for sentence in poem if 'Copyright' not in sentence]
        poem = [sentence for sentence in poem if 'reprint' not in sentence.lower()]
        # remove empty strings
        poem = [sentence for sentence in poem if sentence != '']
        poem = [sentence for sentence in poem if sentence != ' ']
        poem = ' '.join(poem)
        # take those with len > 5
        if len(poem) < 5:
            continue
        poem = poem.strip()
        poem = poem.replace('\r', '')
        units_clean_merve.append(poem)
    print('no. of poems', len(units_clean_merve))

# take sentences
elif analysis_basis == "sentences":
    # get all the sentences, seperated by \n
    units = df_love['content'].str.split('\n').explode().tolist()
    # save the author and poem names
    authors = df_love['author'].str.split('\n').explode().tolist()
    print('authors', len(set(authors)))
    poems_names = df_love['poem name'].str.split('\n').explode().tolist()
    #print('poems', len(set(poems)))

    # get the sentences
    units_clean_merve = []
    for line in units:
        # we want to remove extra spaces
        line = line.replace('\n', ' ').replace('  ', ' ')
        line = line.strip().replace('*', '')
        # remove leading and trailing spaces
        line = line.strip()
        units_clean_merve.append(line)
    
    # remove emtpy strings
    units_clean_merve = [unit for unit in units_clean_merve if unit != '']
    # remove sentences that are just a number
    units_clean_merve = [unit for unit in units_clean_merve if not unit.isdigit()]
    # remove empty strings
    units_clean_merve = [unit for unit in units_clean_merve if unit != ' ']
    units_clean_merve = [unit for unit in units_clean_merve if len(unit) > 5]
    # remove \r from strings
    units_clean_merve = [unit.replace('\r', '') for unit in units_clean_merve]
    # remove all sentences containing "copyright"
    units_clean_merve = [unit for unit in units_clean_merve if 'copyright' not in unit.lower()]
    # remove all sentence containing "reprint" or "reprinted"
    units_clean_merve = [unit for unit in units_clean_merve if 'reprint' not in unit.lower()]
    print('no. of lines', len(units_clean_merve))

else:
    raise ValueError("Invalid analysis_basis")


# %%
tag = 'love'

# get the embeddings
if os.path.exists(f'data/embeddings/{tag}_{analysis_basis}.json'):
    with open(f'data/embeddings/{tag}_{analysis_basis}.json', 'r') as f:
        sentence_embeddings = np.array(json.load(f))
    print('loaded')
else:
    sentence_embeddings = generate_clip_embeddings(units_clean_merve)
    with open(f'data/embeddings/{tag}_{analysis_basis}.json', 'w') as f:
        json.dump(sentence_embeddings.tolist(), f)

norms_love = np.linalg.norm(sentence_embeddings, axis=1)
entropies_love = [calculate_entropy(embedding) for embedding in sentence_embeddings]
variances_love = np.var(sentence_embeddings, axis=1)
sparsities_love = [calculate_sparsity_ratio(embedding) for embedding in sentence_embeddings]

# check if the scores are already saved
if os.path.exists(f'data/measures/{tag}_{analysis_basis}.json'):
    df_sentences_love = pd.read_json(f'data/measures/{tag}_{analysis_basis}.json')
    print('loaded')
else:
    # get the imageability scores
    imag_love, imag_love_norm = get_dict_scores(units_clean_merve, 'resources/mrc_psychol_dict.json', 'imag', 'lemma_', normalize_by_tokens=False)
    vis_love, vis_love_norm = get_dict_scores(units_clean_merve, 'resources/sensorimotor_norms_dict.json', 'Visual.mean', 'lemma_', normalize_by_tokens=False)
    conc_love, conc_love_norm = get_dict_scores(units_clean_merve, 'resources/concreteness_brysbaert.json', 'Conc.M', 'lemma_', normalize_by_tokens=False)

    # make a dataframe
    df_sentences_love = pd.DataFrame({'norm': norms_love, 'entropy': entropies_love, 'variance': variances_love, 'sparsity': sparsities_love})
    df_sentences_love['imag'] = imag_love
    df_sentences_love['visual'] = vis_love
    df_sentences_love['concrete'] = conc_love
    df_sentences_love['imag_norm'] = imag_love_norm
    df_sentences_love['visual_norm'] = vis_love_norm
    df_sentences_love['concrete_norm'] = conc_love_norm

    # save the measures
    df_sentences_love.to_json(f'data/measures/{tag}_{analysis_basis}.json', orient='records')

# add the text
df_sentences_love['text'] = units_clean_merve

# %% 
# we get imageable and non imageable sentences as reference points
# cocnat the two dfs
df_sentences_all = pd.concat([df_sentences, df_sentences_love])

# make pd print all
pd.set_option('display.max_colwidth', None)

# print the first 10 most imageable sentences in both dfs
print('most imageable')
# make it a list
imageable = df_sentences_all.sort_values('imag_norm', ascending=True)['text'].tolist()
# print the first 10
for sent in imageable[:300]:
    print(sent)
    print('\n')

# %%
# we choose these sentences (from topmosts and bottommosts) as examples
imag_example = "Homespun, dyed butternuts dark gold color."
non_imag_example = "Of insidious intent"

# retrieve the measures for these from the df
print('imag example exists:', imag_example)
imag_metrics = df_sentences_all[df_sentences_all['text'] == imag_example]
print('non-imag example exists:', non_imag_example)
non_imag_metrics = df_sentences_all[df_sentences_all['text'] == non_imag_example]

# %%
# make distributions of each group for each measure
measures = ['norm', 'entropy', 'sparsity', 
            'imag_norm', 'visual_norm', 'concrete_norm']

sns.set(style='whitegrid')

plt.figure(figsize=(12, 5))

for i, measure in enumerate(measures):
    plt.subplot(2, 3, i+1)
    sns.histplot(df_sentences[measure], label='Imagist', color='blue', kde=True, alpha=0.2, stat="density")#, shade=True)
    sns.histplot(df_sentences_love[measure], label='Modern Love', color='red', kde=True, alpha=0.2, stat="density")#, shade=True)
    # insert the lines for the examples
    plt.axvline(imag_metrics[measure].values[0], color='blue', linestyle='--', label='Imageable')
    plt.axvline(non_imag_metrics[measure].values[0], color='red', linestyle='--', label='Abstract')
    plt.title(measure)
plt.tight_layout()
# add legend
plt.legend()
plt.show()

# print len of both
print('len imag', len(df_sentences))
print('len love', len(df_sentences_love))
# %%
# do the t-test & mannwhitneyu test
for measure in measures:
    # make sure to drop any NaNs
    df_sentences_nonan = df_sentences.dropna(subset=[measure])
    df_sentences_love_nonan = df_sentences_love.dropna(subset=[measure])
    t, p = ttest_ind(df_sentences_nonan[measure], df_sentences_love_nonan[measure])

    # also make a mannwhitneyu test
    t_m, p_m = mannwhitneyu(df_sentences_nonan[measure], df_sentences_love_nonan[measure])

    print(f'{measure}: t = {t:.2f}, p = {p:.2f}', f'mannwhitneyu: t = {t_m:.2f}, p = {p_m:.2f}')
    print('--')

# %%
# do the same but with the bonferroni correction

alpha = 0.05  # Original significance level
n_tests = len(measures)  # Number of comparisons (tests)
bonferroni_alpha = alpha / n_tests  # Adjusted significance level for Bonferroni correction

for measure in measures:
    # Make sure to drop any NaNs
    df_sentences_nonan = df_sentences.dropna(subset=[measure])
    df_sentences_love_nonan = df_sentences_love.dropna(subset=[measure])
    # print the group sizes
    print('group sizes', len(df_sentences_nonan), len(df_sentences_love_nonan))

    # Perform T-test 
    t, p = ttest_ind(df_sentences_nonan[measure], df_sentences_love_nonan[measure])
    # print the group sizes
    print('group sizes', len(df_sentences_nonan), len(df_sentences_love_nonan))
    # Apply Bonferroni correction
    is_significant = p < bonferroni_alpha
    print(f'{measure}: t = {t:.2f}, p = {p:.2f}, significant: {is_significant}')

    # also make a mannwhitneyu test
    t_m, p_m = mannwhitneyu(df_sentences_nonan[measure], df_sentences_love_nonan[measure])
    # Apply Bonferroni correction
    is_significant = p_m < bonferroni_alpha
    print(f'{measure}: mannwhitneyu: t = {t_m:.2f}, p = {p_m:.2f}, significant: {is_significant}')

    print('--')

# %%
# get the numbers of means and stds per group
print('Imagists')
print(df_sentences.describe())
print('Modern Love')
print(df_sentences_love.describe())

# %%
print('All done:)')
# %%
