import faiss
import numpy as np


class IndexWrapper():

    def __init__(self, dataset, method='flatL2'):
        assert method in ['LSH', 'flatL2'], 'Invalid index method'
        self._dataset = dataset
        self._method = method
        self._index = self._create_index()

    def __len__(self):
        return self._index.ntotal

    def __getitem__(self, idx: int):
        # Return url and corresponding embedding
        return self._dataset.index[idx], self._dataset.embeddings[idx]

    @property
    def d(self):
        return self._index.d

    @property
    def ntotal(self):
        return self._index.ntotal

    def _create_index(self):
        """
        Returns FAISS index created from dataset.embeddings using the
        specified index method
        """
        # Initialize index
        index = None
        d = self._dataset.embeddings_dimension

        if self._method == 'flatL2':
            index = faiss.IndexFlatL2(d)

        elif self._method == 'LSH':
            n_bits = 2 * d
            index = faiss.IndexLSH(d, n_bits)
            index.train(self._dataset.embeddings)

        # Add embeddings
        index.add(self._dataset.embeddings)

        return index

    def search(self, q: np.array, k: int, attrs=None):
        """
        Returns tuple, where each element is an attribute array specified
        in attrs. Valid attributes include 'distances', 'urls', 'indices', and
        'embeddings'
        """
        # If no attributes selected, use default
        default_attrs = ('indices', 'distances')
        attrs = default_attrs if attrs is None else attrs

        # Initialize results dictionary
        results = {}

        # Convert query to 2D np array for FAISS index
        q = q if len(q.shape) == 2 else np.array([q])

        # Search k-NN of q using the index
        nn_distances, nn_indices = self._index.search(q, k)
        results['distances'], results['indices'] = nn_distances, nn_indices

        # Get other attributes
        other_attrs = [attr for attr in attrs if attr not in default_attrs]
        for attr in other_attrs:
            assert hasattr(self._dataset, attr), 'Invalid attribute'
            results[attr] = [
                getattr(self._dataset, attr)[idx] for idx in results['indices']]

        # Return results in order given by attrs
        return tuple([results[attr] for attr in attrs])
