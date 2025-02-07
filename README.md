This is the repository accompanying the study: "Encoding Imagism? Measuring Literary Imageability, Visuality and Concreteness via Multimodal Word Embeddings"

We perform 3 experiments which each have their own ```.py``` file in the main folder. 
- note that when running ```experiment_2```, you will have to set the file you want to treat in the very beginning (hemingway, woolf, or the chicago corpus)
- ```functions.py```defines some useful functions for making embeddings and dictionary scores
- the ```data``` folder contains both textual data, precomputed & saved embeddings (json)(```data/embeddings/```) and precomputed dictionary-scores (```data/mesures/```) for each text -- but you can also recompute them by deleting these files once having cloned the repo
- the ```resources```folder contains the dictionaries used
- the folders ```figs``` & ```results``` will contain outputs once analysis is run (for each file)