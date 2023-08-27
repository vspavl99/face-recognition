import umap

def resize_embedding(embeddings, dim):
    reducer = umap.UMAP(n_components=dim, random_state=2023)
    umap_emb = reducer.fit_transform(embeddings)
    return umap_emb
