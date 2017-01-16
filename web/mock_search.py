import numpy as np

def search_roi(search_model, image, bounding_box, top_n=0):
    """A mock search_roi implementation avoiding the slow computation time"""
    RNG = np.random.RandomState(1337)
    num_features = len(search_model.feature_store)

    n = min(max(5, top_n), num_features)

    indices = np.arange(n)
    similarities = np.linspace(1.0, 0.0, n)

    bounding_boxes = []
    for idx in indices:
        metadata = search_model.get_metadata(idx)
        x1 = RNG.randint(0, metadata['width'])
        x2 = RNG.randint(x1, metadata['width'])
        y1 = RNG.randint(0, metadata['height'])
        y2 = RNG.randint(y1, metadata['height'])
        bounding_boxes.append((x1, y1, x2, y2))

    return indices, similarities, bounding_boxes
