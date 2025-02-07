# %%
import json
import spacy
import numpy as np
from pathlib import Path
from loguru import logger
import datetime

# for embeddings
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoModel
from typing import List

import torch

# get embeddings
logger.add("logs/embeddings.log", format="{time} {level} {message}", level="INFO")

def ensure_dir_exists(path: Path) -> None:
    if not path.exists():
        logger.info(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)


# Generate embeddings using the CLIP model
def generate_clip_embeddings(
    texts: List[str],  # Corrected type hint
    model_name: str = "openai/clip-vit-base-patch32",
    save_name: Path = None
) -> List[List[float]]:  # Corrected type hint
    """
    Args:
        texts (List[str]): list of sentences
        model_name (str): name of the model
        save_name (Path | None): name to save embeddings under

    Returns:
        List[List[float]]: embeddings for each sentence
    """
    logger.info(f"Loading CLIP model and processor: {model_name}")

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Using device: {device}")

    # process
    logger.info("Processing input texts.")
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Generate embeddings
    logger.info("Generating embeddings.")
    with torch.no_grad():
        embeddings = model.get_text_features(**inputs).cpu().numpy()  # making numpy array

    # save embeddings if save path
    if save_name:
        date = datetime.datetime.now().strftime("%Y-%m-%d")  # get current date
        save_path = Path(f"data/embeddings/{save_name}_{model_name}_{date}.json")

        try:
            ensure_dir_exists(save_path.parent)  # Ensure directory exists
            with open(save_path, "w") as f:
                json.dump(embeddings.tolist(), f)
            logger.info(f"Saved embeddings to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")

    return embeddings



def get_dict_scores(texts, dict_path, score_key, token_attr='lemma_', normalize_by_tokens=True):
    """
    Computes scores (e.g., imageability, visuality, concreteness) for a list of sentences using a given lexicon.

    Args:
        texts (list of str): List of sentences to process.
        dict_path (str): Path to the JSON lexicon file.
        score_key (str): Key in the lexicon for extracting scores (e.g., 'imag', 'Visual.mean').
        token_attr (str): Token attribute to match with the lexicon keys (default: 'lemma_').
        normalize_by_tokens (bool): Whether to normalize scores by the total number of tokens (default: True).
            otherwise, normalize by the number of valid tokens with scores.

    Returns:
        tuple: A tuple of two lists:
            - total_scores (list of float): Total scores for each sentence.
            - normalized_scores (list of float): Normalized scores for each sentence.
    """
    # Load the dictionary
    with open(dict_path, 'r') as f:
        lexicon = json.load(f)
    # lower case all keys
    lexicon = {k.lower(): v for k, v in lexicon.items()}
    print(f'Loaded lexicon for scoring from {dict_path}, len of lexicon:', len(lexicon))

    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Lists to store the scores
    total_scores = []
    normalized_scores = []

    # Process each text
    for text in texts:
        # List to store scores for tokens in the sentence
        token_scores = []

        # Process the sentence using spaCy
        doc = nlp(text)

        for token in doc:
            token_value = getattr(token, token_attr).lower()  # Get specified token attribute
            # Match token value with keys in the lexicon
            if token_value in lexicon:
                token_scores.append(lexicon[token_value][score_key] if isinstance(lexicon[token_value], dict) else lexicon[token_value])
            else:
                token_scores.append(np.nan)

        # Compute total and normalized scores
        if token_scores:
            total_score = np.nansum(token_scores)  # Sum of valid scores
            if normalize_by_tokens:
                normalized_score = total_score / len(doc)  # Normalize by sentence length
            else:
                valid_scores_count = np.count_nonzero(~np.isnan(token_scores))  # Normalize by valid tokens
                normalized_score = total_score / valid_scores_count if valid_scores_count > 0 else np.nan
        else:
            total_score, normalized_score = np.nan, np.nan  # Handle empty cases

        # Append results
        total_scores.append(total_score)
        normalized_scores.append(normalized_score)

    return total_scores, normalized_scores



# if necessary (not used)
# get nominal verb ratio, ttr of nouns, and noun count
def get_nominal_verb_ratio(texts):
    # load model
    nlp = spacy.load("en_core_web_sm")

    # Lists to store results
    nominal_verb_ratios = []
    noun_counts = []
    noun_ttrs = []

    # Process each text
    for text in texts:
        doc = nlp(text)

        num_nominal = sum(1 for token in doc if token.pos_ in ['PROPN', 'ADJ'])
        num_verb = sum(1 for token in doc if token.pos_ == 'VERB')
        nouns = [token.text for token in doc if token.pos_ == 'NOUN']

        # Calc nominal/verb ratio
        nominal_verb_ratios.append(
            (num_nominal + len(nouns)) / num_verb if num_verb > 0 else 0)

        # Count number of nouns
        noun_counts.append(len(nouns))

        # Calculate TTR of nouns
        noun_ttrs.append(len(set(nouns)) / len(nouns) if len(nouns) > 0 else 0)

    return nominal_verb_ratios, noun_counts, noun_ttrs