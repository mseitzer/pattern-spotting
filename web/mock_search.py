import numpy as np

def search(search_model, query, top_n=0, localize=True, localize_n=50, 
           rerank=True, avg_qe=True):
    """A mock search implementation avoiding the slow computation time"""
    RNG = np.random.RandomState(1337)
    num_features = len(search_model.feature_store)

    n = min(max(5, top_n), num_features)

    indices = np.arange(n)
    similarities = np.linspace(1.0, 0.0, n)
    similarities[0] = np.nan

    if localize:
        bounding_boxes = []
        for idx in indices[:localize_n]:
            metadata = search_model.get_metadata(idx)
            x1 = RNG.randint(0, metadata['width'])
            x2 = RNG.randint(x1, metadata['width'])
            y1 = RNG.randint(0, metadata['height'])
            y2 = RNG.randint(y1, metadata['height'])
            bounding_boxes.append((x1, y1, x2, y2))
    else:
        bounding_boxes = None

    return indices, similarities, bounding_boxes
