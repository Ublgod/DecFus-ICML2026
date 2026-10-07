import pandas as pd
from torchvision import datasets
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def build_non_iid_by_dirichlet(random_state, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers):
    """Create a non-IID data partition with a Dirichlet distribution."""
    n_auxi_workers = 10

    # Shuffle the target indices
    random_state.shuffle(indices2targets)

    # Split the indices
    from_index = 0
    splitted_targets = []
    num_splits = math.ceil(n_workers / n_auxi_workers)
    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]

    for idx, _ in enumerate(split_n_workers):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
            from_index: (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    idx_batch = []
    for _targets in splitted_targets:
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        _n_workers = min(n_auxi_workers, n_workers)
        n_workers = n_workers - n_auxi_workers

        min_size = 0
        # Make sure every client has at least a minimum number of samples
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # Balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < 1.1 * _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[:-1]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    min_size = min([len(idx_j) for idx_j in _idx_batch])
                except ZeroDivisionError:
                    pass
        idx_batch += _idx_batch
    return idx_batch


def prepare_data(client_indices, targets, num_class, client_ids):
    """Prepare the data for visualization."""
    num_clients = len(client_indices)
    targets_clients = [targets[cd] for cd in client_indices]
    class_num_clients = [Counter(tc) for tc in targets_clients]

    total_class_counts = Counter(np.concatenate(targets_clients))
    data = np.zeros((num_clients, num_class), dtype=float)

    for i in range(num_clients):
        for j in range(num_class):
            if total_class_counts[j] > 0:
                data[i, j] = class_num_clients[i][j] / total_class_counts[j]

    print('Class distribution: ')
    for i in range(num_clients):
        print(f'Client {client_ids[i]}: {sorted(class_num_clients[i].items())}')
        print(f'Client {client_ids[i]} total samples: {sum(class_num_clients[i].values())}')

    df = pd.DataFrame(data)
    df = df.reset_index().melt(id_vars='index')
    df.columns = ['Client IDs', 'Class labels', 'Bubble Size']
    return df


def visualize_data_distribution():
    """Main function: visualize CIFAR10 data partitions under different Dirichlet distributions."""

    # Parameters
    non_iid_alphas = [0.1, 0.5, 1.0, 100.0]  # can be changed to [0.1, 0.5, 1.0, 100.0]
    num_clients = 10
    client_ids = list(range(num_clients))
    classes_n = 10

    # Load the CIFAR10 dataset
    print("Downloading the CIFAR10 dataset...")
    dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=True)
    targets = np.array(dataset_train.targets)

    # Set the random seed
    rs = np.random.RandomState(1)

    # Indices of all samples
    data_length = len(targets)
    all_indices = np.arange(data_length)
    np.random.shuffle(all_indices)

    # Plot settings
    sns.set(style="darkgrid")
    color = 'red'
    max_bsize = 500

    # Create subplots
    fig, axes = plt.subplots(1, len(non_iid_alphas),
                             figsize=(5 * len(non_iid_alphas), 5),
                             sharey='row')

    # One visualization per alpha value
    for idx, alpha in enumerate(non_iid_alphas):
        print(f'\n============ Alpha: {alpha} =============')

        # Build (index, label) pairs
        indices_w_labels = np.array([(idx, target) for idx, target in enumerate(targets)])

        # Partition the data with a Dirichlet distribution
        list_of_indices = build_non_iid_by_dirichlet(
            rs, indices_w_labels, alpha, classes_n, data_length, num_clients
        )

        # Prepare the data for visualization
        df = prepare_data(list_of_indices, targets, classes_n, client_ids)

        # Bubble size range
        min_ratio, max_ratio = df['Bubble Size'].min(), df['Bubble Size'].max()
        cur_bmin = max_bsize * np.power(min_ratio, 0.6)
        cur_bmax = max_bsize * np.power(max_ratio, 0.6)

        # Select the current subplot
        ax = axes[idx] if len(non_iid_alphas) > 1 else axes

        # Scatter plot
        class_labels = list(range(classes_n))
        sns.scatterplot(data=df, x='Client IDs', y='Class labels',
                        size='Bubble Size', sizes=(cur_bmin, cur_bmax),
                        legend=False, ax=ax, color=color)

        # Axes
        ax.set_xticks(range(len(client_ids)))
        ax.set_yticks(class_labels)
        ax.set_xticklabels(client_ids, fontsize=16)
        ax.set_yticklabels(class_labels, fontsize=16)

        # Title - shown as IID when alpha=100
        title = 'IID' if alpha >= 100.0 else f'alpha={alpha}'
        ax.set_title(title, fontsize=18)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)

    # Legend
    handles = [
        plt.scatter([], [], s=max_bsize / 10, color=color, label='10%'),
        plt.scatter([], [], s=max_bsize / 4, color=color, label='25%'),
        plt.scatter([], [], s=max_bsize / 2, color=color, label='50%'),
        plt.scatter([], [], s=max_bsize, color=color, alpha=1, label='100%')
    ]
    fig.legend(handles=handles, loc='upper right', title='Bubble size',
               labelspacing=1.5, borderpad=1.5, bbox_to_anchor=(1.05, 0.935))

    # Adjust the layout and save - leave more room for the legend
    plt.tight_layout(rect=[0, 0, 0.98, 1])
    plt.savefig('cifar10_dirichlet_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nVisualization finished. Figure saved as 'cifar10_dirichlet_distribution.png'")


if __name__ == "__main__":
    # Run
    visualize_data_distribution()
