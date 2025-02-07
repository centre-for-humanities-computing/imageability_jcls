
# %%

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import spacy
import re
import os 

import datetime

from scipy.stats import spearmanr
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy

from functions import get_dict_scores
from functions import generate_clip_embeddings

from transformers import CLIPProcessor, CLIPModel
import torch


### Second part of our study: we take the embeddings of 3 literary sets and
# correlated embedding-based metrics with dictionary-based metrics

# %%
def clean_whitespace(text: str) -> str:
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

# %%

# CONFIGURATION

use_data = 'chicago_9000'

# Model name
model_name = "openai/clip-vit-base-patch32"

print('we use the data:', use_data)
print('with the model:', model_name)

if use_data == 'chicago_9000':
    # read json
    # open chicago
    with open(f'data/{use_data}.json', 'r') as f:
        txt = json.load(f)
    # get a list of all the sentences (these are already tokenized w spacy)
    sample_sentences = [clean_whitespace(sentence) for sentence in txt.values()]
    # make sure odd characters are removed
    sample_sentences = [re.sub(r'[^a-zA-Z0-9.,!? ]', '', sentence) for sentence in sample_sentences]

else:
    # Get sample sentences from novel or chicago
    with open(f'data/{use_data}.txt', 'r') as f: # or open what we set as use_data
        txt = f.read()
    # clean the txt
    clean_txt = clean_whitespace(txt)
    # tokenize w spacy
    nlp = spacy.load("en_core_web_sm")
    text_tokens = nlp(clean_txt).sents
    sample_sentences = [sent.text for sent in text_tokens]
    sample_sentences

sample_texts = sample_sentences

# we note down some run-info for saving the results
run_info = {
    'date': datetime.datetime.now().strftime("%Y-%m-%d"),
    'model': model_name,
    'used_data': use_data,
    'sample_texts_len': len(sample_texts),
    'sample_texts_first': sample_texts[0],
    'sample_texts_last': sample_texts[-1]
}
print(run_info)
# %%
# get the dictionary scores
use_data_path = None

# see if we already saved measures
for path in os.listdir('data/measures'):
    if use_data in str(path):
        use_data_path = path
        print(f"Found data path: {use_data_path}")
        break  # Exit the loop once the file is found

if use_data_path:  # If a file was found, load it
    data = pd.read_json(f'data/measures/{use_data}.json', orient='records', lines=True)

else:  # If no file is found, generate and save the scores
    imageability_scores, norm_imageability_scores = get_dict_scores(sample_texts, 'resources/mrc_psychol_dict.json', 'imag', 'lemma_', normalize_by_tokens=False)
    visuality_scores, norm_visuality_scores = get_dict_scores(sample_texts, 'resources/sensorimotor_norms_dict.json', 'Visual.mean', 'lemma_', normalize_by_tokens=False)
    concreteness_scores, norm_concreteness_scores = get_dict_scores(sample_texts, 'resources/concreteness_brysbaert.json', None, 'lemma_', normalize_by_tokens=False)
    reilly_scores, norm_reilly_scores = get_dict_scores(sample_texts, 'resources/reilly_dict.json', None, 'lemma_', normalize_by_tokens=False)

    print('imageability_scores:', len(imageability_scores))
    print('visuality_scores:', len(visuality_scores))
    print('concreteness_scores:', len(concreteness_scores))

    # Save the generated scores
    dict_scores = {
        'imageability': imageability_scores,
        'norm_imageability': norm_imageability_scores,
        'visuality': visuality_scores,
        'norm_visuality': norm_visuality_scores,
        'concreteness': concreteness_scores,
        'norm_concreteness': norm_concreteness_scores,
        'reilly': reilly_scores,
        'norm_reilly': norm_reilly_scores
    }

    # make df
    data_df = pd.DataFrame.from_dict(dict_scores)
    # dump it
    data_df.to_json(f'data/measures/{use_data}.json', orient='records', lines=True)

    # and reopen
    data = pd.read_json(f'data/measures/{use_data}.json', orient='records', lines=True)


# %%
# and now the embedding-based metrics...
model = "openai/clip-vit-base-patch32"

if model == "openai/clip-vit-base-patch32":

    # check if the embeddings are already saved
    for path in os.listdir('data/embeddings'):
        if use_data in str(path):
            use_data_path = path
            print(use_data_path)
            break
        else:
            use_data_path = None

    if use_data_path:
        print(use_data_path)
        with open(f'data/embeddings/{use_data_path}', 'r') as f:
            text_embeddings = json.load(f)
            print('len of embeddings:', len(text_embeddings))
            print('len of embedding vector:', len(text_embeddings[0]))
            print('len of sample_texts:', len(sample_texts))
    else:
        print('no embeddings found, making them now...')
        text_embeddings = generate_clip_embeddings(sample_texts, model_name, save_name=use_data)
        print('len of embeddings:', len(text_embeddings))
        print('len of embedding vector:', len(text_embeddings[0]))
        print('len of sample_texts:', len(sample_texts))


# %%
# embedding-based metrics

# L2 norm of the embeddings
norms = np.linalg.norm(text_embeddings, axis=1)

# entropies
def prob(old_embedding):
    abs_embedding = np.abs(old_embedding)  # Ensure non-negative values
    total = np.sum(abs_embedding)
    return abs_embedding / total if total != 0 else np.ones_like(old_embedding) / len(old_embedding)

def calculate_entropy(embedding):
    return entropy(prob(embedding))

entropies = [calculate_entropy(embedding) for embedding in text_embeddings]

# embedding variance
variances = np.var(text_embeddings, axis=1)

# Function to compute the ratio of L1 to L2 norms
def calculate_sparsity_ratio(embedding):
    embedding = np.array(embedding)
    l1_norm = np.sum(np.abs(embedding))  # L1 norm
    l2_norm = np.sqrt(np.sum(embedding**2))  # L2 norm
    sparsity_ratio = l1_norm / (np.sqrt(len(embedding)) * l2_norm)
    return sparsity_ratio

sparsity = [calculate_sparsity_ratio(embedding) for embedding in text_embeddings]

# %%
# add embedding-based metrics to the dataframe
data['norms'] = norms
data['entropies'] = entropies
data['variances'] = variances
data['sparsity'] = sparsity
# and text
data['text'] = sample_texts
data['embedding'] = text_embeddings

# scaled embeddings
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(np.array(text_embeddings))
data['scaled_embedding'] = scaled_embeddings.tolist()

data.head()

# %%
# describe the data (get means and SDs)
data.describe()

# %%

custom_palette = sns.diverging_palette(580, 580, s=60, as_cmap=True)

# visualize the distribution of the embedding with the highest norm and lowest norm
measures = ['norms', 'entropies']

# remove all points below 2 words length
data_filtered = data[data['text'].apply(lambda x: len(x.split()) > 5)]

for measure in measures:

    measure_embedding_dict = dict(zip(data_filtered[measure], data_filtered['embedding']))
    sorted_dict = sorted(measure_embedding_dict.items(), key=lambda x: x[0])

    # Extract least and most measure embeddings
    least_value, least_embedding = sorted_dict[0]
    most_value, most_embedding = sorted_dict[-1]

    print(f"Least {measure}: {least_value}, Most {measure}: {most_value}")
    # print the sentences that we are plotting
    print('plotted_least:', data_filtered[data_filtered[measure] == least_value]['text'].values[0])
    print('plotted_msot:', data_filtered[data_filtered[measure] == most_value]['text'].values[0])
    print('')
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(2, 1, figsize=(9, 4), gridspec_kw={"height_ratios": [1.7, 1]}, sharex=True)
    # Plot the heatmap
    sns.heatmap(
        [least_embedding, most_embedding],
        annot=False,
        cmap="coolwarm",# custom_palette,
        ax=ax[0],
        cbar=False,
        xticklabels=False,
        yticklabels=["Least", "Most"],
    )
    ax[0].tick_params(axis="x", rotation=45)  # Rotate x-ticks on the heatmap
    # add a line in the midlle
    ax[0].axhline(1, color="black", linewidth=0.5, linestyle="--", alpha=0.2)

    # Plot the line plot in the middle
    ax[1].plot(least_embedding, label="Least", color="red", alpha=0.5, linestyle="-", linewidth=2)
    ax[1].plot(most_embedding, label="Most", color="blue", alpha=0.4, linestyle="-", linewidth=2)
    ax[1].set_ylim(-4, 4)  # Adjust the range based on your data
    ax[1].set_xlabel("Embedding Dimension", fontsize=12)
    # set xticks to match heatmap and write them out
    ax[0].set_title(f"{measure}", fontsize=14)
    ax[1].set_xticks(range(0, len(least_embedding), 10))  # Add x-ticks every 10 dimensions
    ax[1].set_xticklabels(range(0, len(least_embedding), 10), fontsize=8, rotation=0.45)  # Add labels for ticks
    ax[1].tick_params(axis="x", rotation=90)  # Rotate x-ticks on the line plot
    ax[1].legend(loc='upper right')
    plt.tight_layout(pad=1.0)

    # save
    plt.savefig(f'figs/example_heatmap_lineplot_{use_data}_{measure}.png')

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # also plot the distribution of embeddings values across dimensions
    plt.figure(figsize=(8, 3))
    plt.title(f"{measure} distribution")
    sns.kdeplot(least_embedding, color='coral', alpha=0.5, lw=4, bw_adjust=0.4)#, bins=20, stat='density')
    sns.kdeplot(most_embedding, color='lightseagreen', alpha=0.5, lw=4, bw_adjust=0.4)#, bins=20, stat='density')
    plt.xlabel("Embedding values")
    plt.ylabel("Count")
    plt.legend(["Least", "Most"])
    plt.tight_layout(pad=1.0)
    plt.savefig(f'figs/example_histplot_{use_data}_{measure}.png')
    plt.show()


# %%
# heatmap of the correlations between the measures

df = data[['norms', 'entropies', 'sparsity',
           'imageability', 'visuality', 'concreteness',
           'text', 'embedding']].copy()

# rename the columns
df.columns = ['norm', 'entropy', 'sparsity',
              'imag', 'visual', 'concrete',
              'text', 'embedding']

# drop the text and id columns
dt = df.drop(columns=['text', 'embedding'])

# get the spearman correlation
corr_df = dt.corr(method='spearman')

plt.figure(figsize=(3, 3))
sns.heatmap(corr_df, annot=True, cbar=False, cmap='coolwarm')
# rotate the x-axis labels
plt.xticks(rotation=50)
# pad the x-axis labels
plt.yticks(rotation=0)
plt.tight_layout(pad=1.0)
plt.savefig(f'figs/corr_{use_data}_dict_scores.png')
plt.show()


# %%
# let's also visualize these correlations. 
# we make a a scatterplot arranged like a heatmap

plt.figure(figsize=(20, 15))
sns.set_style('whitegrid')
for i, col in enumerate(dt.columns):
    for j, col2 in enumerate(dt.columns):
        plt.subplot(len(dt.columns), len(dt.columns), i * len(dt.columns) + j + 1)
        if i == j:
            sns.histplot(df[col], kde=True, color='coral')
            plt.xlabel(col)
            plt.ylabel('count')
        else:
            sns.scatterplot(x=col, y=col2, data=dt, alpha=0.3)
            # find correlation
            corr = spearmanr(df[col], df[col2])[0]
            # annotate the correlation strength at the top of plot
            plt.text(0.5, 0.9, f'{corr:.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=25)
plt.tight_layout()
plt.show()


# %%

# visualize the embeddings themselves
feature_names = ['imageability', 'visuality', 'concreteness']

for feat in feature_names:
    # Create a dictionary mapping measure values to embeddings
    dict_feature_embeddings = dict(zip(data[feat], data['scaled_embedding']))
    
    # Sort by measure values
    sorted_dict_feat = sorted(dict_feature_embeddings.items(), key=lambda x: x[0])

    top_10_embeddings = [sorted_dict_feat[-i][1] for i in range(1, 11)]
    bottom_10_embeddings = [sorted_dict_feat[i][1] for i in range(10)]

    # Compute the mean embedding for top and bottom
    mean_top_embedding = np.mean(top_10_embeddings, axis=0)
    mean_bottom_embedding = np.mean(bottom_10_embeddings, axis=0)

    top = sorted_dict_feat[0][1]
    bottom = sorted_dict_feat[-1][1]

    # Create the figure for the heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))

    sns.heatmap(
        [mean_top_embedding, mean_bottom_embedding],
        annot=False,
        cmap="Spectral",#"vlag",
        ax=ax,
        cbar=True,
        xticklabels=False,
        yticklabels=['top',"bottom"],
    )
    
    # Rotate x-ticks for better visibility
    ax.tick_params(axis="x", rotation=45)
    
    # Add a line in the middle of the heatmap
    ax.axhline(1, color="white", linewidth=1, linestyle="-", alpha=0.7)
    
    # Set the title
    ax.set_title(f"{feat}", fontsize=14)
    
    # Adjust x-ticks
    ax.set_xticks(range(0, len(mean_top_embedding), 10))  # Add x-ticks every 10 dimensions
    ax.set_xticklabels(range(0, len(mean_top_embedding), 10), fontsize=8)  # Label ticks
    ax.tick_params(axis="x", rotation=90)  # Rotate x-ticks
    ax.set_xlabel("Embedding Dimension", fontsize=10)

    # Adjust layout
    plt.tight_layout()


# %%
# print the most and least imageable sentences
print('least imageable:', df[df['imag'] == df['imag'].min()]['text'].values[0])
print('most imageable:', df[df['imag'] == df['imag'].max()]['text'].values[0])

# print the most and least norm sentences
print('least norm:', df[df['norm'] == df['norm'].min()]['text'].values[0])
print('most norm:', df[df['norm'] == df['norm'].max()]['text'].values[0])

# %%
# print the 10 most and least imageable sentences
# filter out sentence more than 30 words
data_window = data_filtered[data_filtered['text'].apply(lambda x: len(x.split()) < 20)]
least_imageable = data_window.sort_values('imageability').head(40)
most_imageable = data_window.sort_values('imageability').tail(40)

print('least imageable:')
for text in least_imageable['text']:
    print(text)
print('---')
print('most imageable:')
for text in most_imageable['text']:
    print(text)

# %%
# Save things

# get the correlation matrix
corr_df = dt.corr(method='spearman').round(2)

# add main correlations to the run_info
run_info['features_used'] = list(df.columns)
run_info['correlation_matrix'] = corr_df.to_dict()
run_info['correlation_matrix'] = {key: {key2: value2 for key2, value2 in value.items()} for key, value in run_info['correlation_matrix'].items()}

# add the least and most imageable sentences
run_info['least_imageable'] = df[df['imag'] == df['imag'].min()]['text'].values[0]
run_info['most_imageable'] = df[df['imag'] == df['imag'].max()]['text'].values[0]

# add the least and most norm sentences
run_info['least_norm'] = df[df['norm'] == df['norm'].min()]['text'].values[0]
run_info['most_norm'] = df[df['norm'] == df['norm'].max()]['text'].values[0]

# save the run_info
with open(f'results/run_info_{use_data}_{run_info["date"]}.json', 'w') as f:
    json.dump(run_info, f, indent=4)



# %%
# Histograms with example sentences

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# let's just plot the norms and entropies to see if they make sense using an agreed-upon sentence
concrete_sentence = "The thin white surgical gloves he wore as he pumped the gas looked like pale skin." # from chicago
abstract_sentence = "Wishful thinking as the saying goes." # from chicago

concrete_inputs = processor(text=concrete_sentence, return_tensors="pt", padding=True, truncation=True)
abstract_inputs = processor(text=abstract_sentence, return_tensors="pt", padding=True, truncation=True)

concrete_sentence_imag = get_dict_scores([concrete_sentence], 'resources/mrc_psychol_dict.json', 'imag', 'lemma_', normalize_by_tokens=False)[0]
abstract_sentence_imag = get_dict_scores([abstract_sentence], 'resources/mrc_psychol_dict.json', 'imag', 'lemma_', normalize_by_tokens=False)[0]
print('concrete imageability:', concrete_sentence_imag)
print('abstract imageability:', abstract_sentence_imag)

with torch.no_grad():
        concrete_embedding = model.get_text_features(**concrete_inputs).cpu().numpy() # making numpy array
        abstract_embedding = model.get_text_features(**abstract_inputs).cpu().numpy()

concrete_norm = np.linalg.norm(concrete_embedding)
concrete_entropy = calculate_entropy(concrete_embedding.squeeze())

abstract_norm = np.linalg.norm(abstract_embedding)
abstract_entropy = calculate_entropy(abstract_embedding.squeeze())

print('concrete norm:', concrete_norm, 'concrete entropy:', concrete_entropy)
print('abstract norm:', abstract_norm, 'abstract entropy:', abstract_entropy)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(norms, bins=20, color='skyblue', alpha=0.7)
plt.axvline(concrete_norm, color='blue', linestyle='dashed', linewidth=2)
plt.axvline(abstract_norm, color='red', linestyle='dashed', linewidth=2)
plt.xlabel('norm')
plt.ylabel('count')

plt.subplot(1, 2, 2)
plt.hist(entropies, bins=20, color='skyblue', alpha=0.7)
plt.axvline(concrete_entropy, color='blue', linestyle='dashed', linewidth=2)
plt.axvline(abstract_entropy, color='red', linestyle='dashed', linewidth=2)
plt.xlabel('entropy')
plt.ylabel('count')
plt.legend(['imageable', 'abstract', f'{use_data}'])

plt.savefig(f'figs/histplot_comparison_{use_data}.png')
plt.show()

# %%
print('done:)')

# %%
