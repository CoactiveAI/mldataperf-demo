from typing import List, Tuple, Optional, Callable
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from IPython.core.display import Image, display, clear_output
from sklearn.metrics import average_precision_score

from .datasets import OpenImagesDataset
from .index import IndexWrapper


def create_dataset(metadata_dir: str, embeddings_dir: str, split: str):
    return OpenImagesDataset(metadata_dir, embeddings_dir, split=split)


def create_index(dataset, method: str = 'LSH'):
    return IndexWrapper(dataset, method=method)


def set_random_seed(seed: Optional[int] = None):
    seed = np.random.randint(10000) if seed is None else seed
    np.random.seed(seed)
    return seed


def create_seed(labels: np.array,
                emb_dimension: int,
                npositive: int = 5,
                nnegative: int = 95,
                random_seed: int = 400):
    # Make sure there are enough negative examples
    nnegative = max(nnegative, emb_dimension - npositive)

    # Select indices for random samples
    pos_samples = np.random.choice(
        np.where(labels == 1)[0], npositive, replace=False)
    neg_samples = np.random.choice(
        np.where(labels == 0)[0], nnegative, replace=False)

    # Combine and sort
    seed = np.sort(np.concatenate((pos_samples, neg_samples)))

    return seed


def visualize_urls(urls: List[str], message: str = ''):
    print(message)
    [display(Image(url=url, width=400)) for url in urls]
    input('Press any key to continue')
    clear_output()


def train_model(embeddings: np.array, labels: np.array):
    return LogisticRegression().fit(embeddings, labels)


def eval_model(model, dataset):
    proba = model.predict_proba(dataset.embeddings)[:, 1]
    return average_precision_score(dataset.labels, proba)


def select_maxent(model, dataset, indices: List[int], budget: int):
    # Get prediction probability for elegible indices
    proba = model.predict_proba(dataset.embeddings[indices])

    # Calculate information entropy from probabilities
    entropy = -1.0 * (np.log(proba) * proba).sum(axis=1)

    # Select indices with highest entropy (i.e. MaxEnt)
    selected = entropy.argsort(axis=0)[::-1][:budget]

    # Convert to original index
    selected = np.array(indices)[selected]

    return selected


def collect_labels(dataset, indices: np.array, target_class: str):
    labels = []
    for url in dataset.urls[indices]:

        # Show link and image
        clear_output()
        print(f'Link: {url}')
        display(Image(url=url, width=400))
        time.sleep(0.25)

        # Get user label
        need_input = True
        while need_input:
            label = input(f'Is this an example of {target_class}? [Y/n] ')
            label = 'Y' if label is None or label == '' else label[0].upper()

            if label not in ['N', 'Y']:
                print('Invalid input')
                continue

            labels.append(1 if label == 'Y' else 0)
            need_input = False

    clear_output()
    print(f'Finished labeling {len(labels)} images')
    return labels


def visualize_scores(scores: List[float], ylabel: str = 'Average precision'):
    plt.title('Model score for test data')
    plt.xlabel('Active learning round (#)')
    plt.ylabel(ylabel)

    plt.plot(range(len(scores)), scores, 'o-', color='b')
    plt.xticks(range(len(scores)))
    plt.legend(loc="best")
    plt.show()


def seals(train,
          test,
          knn,
          concept: str,
          rounds: Tuple[int],
          npos: int = 5,
          nneg: int = 95,
          k: int = 100,
          select: Callable = select_maxent):

    # Set initial values
    labeled = create_seed(train.labels, train.embeddings_dimension, npos, nneg)
    new_labeled = labeled
    labels = train.labels[labeled]
    candidates = set()
    scores = []
    model = None

    # Visualize seed
    visualize_urls(train.urls[labeled[np.where(labels == 1)]],
                   message=f'Initial positive samples for {concept}')

    # Main active learning loop
    for budget in rounds:

        # Update candidate pool
        neighbors = knn.search(train.embeddings[new_labeled],
                               k=k, attrs=['indices'])[0][1:].flatten()
        candidates.update(neighbors)

        # Train model using labeled samples and score
        model = train_model(train.embeddings[labeled], labels)
        scores.append(eval_model(model, test))

        # Select points to label (MaxEnt for now)
        new_labeled = select(model, train, list(candidates), budget)

        # User labels selected points
        new_labels = collect_labels(train, new_labeled, concept)

        # Update arrays
        labeled = np.concatenate((labeled, new_labeled))
        labels = np.concatenate((labels, new_labels))
        candidates -= set(new_labeled)

    # Train and score final iteration of model
    model = train_model(train.embeddings[labeled], labels)
    scores.append(eval_model(model, test))
    return model, scores
