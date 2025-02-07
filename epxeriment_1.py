# %%
import numpy as np
import os
from scipy.stats import skew, kurtosis
from scipy.stats import entropy

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

from transformers import CLIPProcessor, CLIPModel
import torch

### first part of our study: dictionary entries

# Here we want to extract CLIP and BERT embeddings for the words in the dictionary
# we then correlate the embedding-based features with the imageability, concretenes, and visuality scores

# %%

model_name = "openai/clip-vit-base-patch32"

# # correlations on the dictionary entries
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

# %%

# make new dicts of the imageability and visuality keys with their norms/entropies

with open('resources/mrc_psychol_dict.json', 'r') as f:
    dict_mrc = json.load(f)

with open('resources/sensorimotor_norms_dict.json', 'r') as f:
    dict_sensori = json.load(f)

with open('resources/concreteness_brysbaert.json', 'r') as f:
    dict_conc = json.load(f)

with open('resources/reilly_dict.json', 'r') as f:
    dict_reilly = json.load(f)

with open('resources/cortese_dict.json', 'r') as f:
    dict_cortese = json.load(f)


# drop all columns but the Visual.mean
dict_sensori = {key: value['Visual.mean'] for key, value in dict_sensori.items()}
dict_mrc = {key: value for key, value in dict_mrc.items() if key in dict_sensori.keys()}
dict_conc = {key: value for key, value in dict_conc.items() if key in dict_sensori.keys()}
dict_reilly = {key: value for key, value in dict_reilly.items() if key in dict_sensori.keys()}
dict_cortese = {key: value for key, value in dict_cortese.items() if key in dict_sensori.keys()}
print('len of dictionaries:', 'len of MRC:', len(dict_mrc), 'len of sensorimotor:', len(dict_sensori), 'len of concreteness:', len(dict_conc), 'len of reilly:', len(dict_reilly), 'len of cortese:', len(dict_cortese))

# get words
words = list(dict_mrc.keys()) # we are using MRC because this is the smallest dictionary


# %%
# we need to extract embeddings for the dictionary keys
inputs = processor(text=words, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    entry_embeddings = model.get_text_features(**inputs)

# convert to numpy
entry_embeddings = entry_embeddings.cpu().numpy()

print(entry_embeddings[0])

# %%
# norm
entry_norm = np.linalg.norm(entry_embeddings, axis=1)

# entropies
def prob(old_embedding):
    abs_embedding = np.abs(old_embedding)  # Ensure non-negative values
    total = np.sum(abs_embedding)
    return abs_embedding / total if total != 0 else np.ones_like(old_embedding) / len(old_embedding)

def calculate_entropy(embedding):
    return entropy(prob(embedding))

entry_entropy = [calculate_entropy(embedding) for embedding in entry_embeddings]

# variance
entry_variance = np.var(entry_embeddings, axis=1)
# skewness
entry_skews = [skew(embedding) for embedding in entry_embeddings]
# kurtosis
entry_kurtoses = [kurtosis(embedding) for embedding in entry_embeddings]

# sparse ratio
# Function to compute the ratio of L1 to L2 norms
def calculate_sparsity_ratio(embedding):
    embedding = np.array(embedding)
    l1_norm = np.sum(np.abs(embedding))  # L1 norm
    l2_norm = np.sqrt(np.sum(embedding**2))  # L2 norm
    sparsity_ratio = l1_norm / (np.sqrt(len(embedding)) * l2_norm)
    return sparsity_ratio
entry_sparsity = [calculate_sparsity_ratio(embedding) for embedding in entry_embeddings]

# %%
### PART I

# plot the correlations of the norms and entropies with the imageability and visuality scores
# make df
df_entries = pd.DataFrame({'norm': entry_norm, 'entropy': entry_entropy, 'variance': entry_variance, 'sparsity': entry_sparsity})#, 'skew': entry_skews, 'kurtosis': entry_kurtoses})
df_entries['imag'] = [dict_mrc[word]['imag'] for word in words]
df_entries['imag R'] = [dict_reilly[word] if word in dict_reilly.keys() else np.nan for word in words]
df_entries['imag C'] = [dict_cortese[word] if word in dict_cortese.keys() else np.nan for word in words]
df_entries['visual'] = [dict_sensori[word] for word in words]
df_entries['concrete'] = [dict_conc[word] if word in dict_conc.keys() else np.nan for word in words]

# make a heatmap
corr_df = df_entries.corr(method='spearman')

plt.figure(figsize=(4.5, 4.5))
sns.heatmap(corr_df, annot=True, cbar=False, cmap='coolwarm')
# rotate the x-axis labels
plt.xticks(rotation=50)
# add space so we dont cut the labels
plt.tight_layout(pad=1.0)
plt.savefig('figs/corr_dictionary_entries.png')
plt.show()

# %%

### PART II
# and try w BERT
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism to avoid issues with transformers

def get_word_vector(word, tokenizer, model, layers=None):
    """Get a word vector by tokenizing the word, passing it through BERT, and averaging the hidden states."""
    encoded = tokenizer(word, return_tensors="pt", add_special_tokens=True)  # Add special tokens for BERT
    # Get all token IDs (the word might get split into subwords)
    token_ids = encoded["input_ids"]
    
    # Push input through the model to get hidden states
    with torch.no_grad():
        output = model(**encoded)

    # Get the hidden states from all layers
    states = output.hidden_states
    # Stack and sum all requested layers (default: last 4 layers)
    layers = [-4, -3, -2, -1] if layers is None else layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

    # For each word, we need to select the token(s) corresponding to it
    word_tokens_output = output[0] if token_ids[0].shape[0] == 1 else output[1:]  # Handle word split into subwords
    return word_tokens_output.mean(dim=0)  # Average the subword embeddings if split

# Check if we already have the embeddings
if os.path.exists("data/BERT/bert_embeddings.json"):
    with open("data/BERT/bert_embeddings.json", "r") as f:
        word_embeddings_array = np.array(json.load(f))
else:
    # Load BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModel.from_pretrained("bert-base-cased", output_hidden_states=True)
    word_embeddings = []
    for word in words:
        word_embeddings.append(get_word_vector(word, tokenizer, model))

    word_embeddings_array = np.stack(word_embeddings)

    # save the BERt embeddings
    with open('data/BERT/bert_embeddings.json', 'w') as f:
        json.dump(word_embeddings_array.tolist(), f)

# %%
# get the embedding-based features

# norm
word_norm = np.linalg.norm(word_embeddings_array, axis=1)
# entropy
probs_embeddings_w = [prob(embedding) for embedding in word_embeddings_array]
word_entropy = [entropy(embedding) for embedding in probs_embeddings_w]

# variance
word_variance = np.var(word_embeddings_array, axis=1)
# skewness
word_skews = [skew(embedding) for embedding in word_embeddings_array]
# kurtosis
word_kurtoses = [kurtosis(embedding) for embedding in word_embeddings_array]
# sparse ratio
word_sparsity = [calculate_sparsity_ratio(embedding) for embedding in word_embeddings_array]

# plot the correlations of the norms and entropies with the imageability and visuality scores
# make df
classic_df = pd.DataFrame({'norm': word_norm, 'entropy': word_entropy, 'variance': word_variance, 'sparsity': word_sparsity})#, 'skew': word_skews, 'kurtosis': word_kurtoses})
classic_df['imag'] = [dict_mrc[word]['imag'] for word in words]
classic_df['imag R'] = [dict_reilly[word] if word in dict_reilly.keys() else np.nan for word in words]
classic_df['imag C'] = [dict_cortese[word] if word in dict_cortese.keys() else np.nan for word in words]
classic_df['visual'] = [dict_sensori[word] for word in words]
classic_df['concrete'] = [dict_conc[word] if word in dict_conc.keys() else np.nan for word in words]

# %%
# make a heatmap
corr_df = classic_df.corr(method='spearman')
# round it
corr_df = corr_df.round(2)

plt.figure(figsize=(4.5, 4.5))
sns.heatmap(corr_df, annot=True, cbar=False, cmap='coolwarm')
# rotate the x-axis labels
plt.xticks(rotation=50)
# rotate the y-axis labels
plt.yticks(rotation=0)
# add space so we dont cut the labels
plt.tight_layout(pad=1.0)

plt.savefig('figs/corr_dictionary_entries_BERT.png')
plt.show()

# %%
print('All done!')
# %%