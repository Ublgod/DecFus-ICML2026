import numpy as np
import json


def read_record(file):
    with open(file,"r") as f:
        dataJson = json.load(f)
        users_train = dataJson["train_data"]
    dict_users_train = {}
    for key,value in users_train.items():
        newKey = int(key)
        dict_users_train[newKey] = value
    '''
    for key,value in users_test.items():
        newKey = int(key)
        dict_users_test[newKey] = value
    '''
    return dict_users_train #, dict_users_test

def separate_data(train_data, num_clients, num_classes, beta=0.4):

    if hasattr(train_data, 'targets'):
        y_train = np.array(train_data.targets)
    elif hasattr(train_data, 'labels'):
        y_train = np.array(train_data.labels)

    min_size_train = 0
    min_require_size = 10
    K = num_classes

    N_train = len(y_train)
    dict_users_train = {}

    while min_size_train < min_require_size:
        idx_batch_train = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k_train = np.where(y_train == k)[0]
            np.random.shuffle(idx_k_train)
            proportions = np.random.dirichlet(np.repeat(beta, num_clients))
            proportions_train = np.array([p * (len(idx_j) < N_train / num_clients) for p, idx_j in zip(proportions, idx_batch_train)])
            proportions_train = proportions_train / proportions_train.sum()
            proportions_train = (np.cumsum(proportions_train) * len(idx_k_train)).astype(int)[:-1]
            idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, np.split(idx_k_train, proportions_train))]
            min_size_train = min([len(idx_j) for idx_j in idx_batch_train])

    for j in range(num_clients):
        np.random.shuffle(idx_batch_train[j])
        dict_users_train[j] = idx_batch_train[j]

    record_net_data_stats(y_train,dict_users_train)

    return dict_users_train


def separate_data_hierarchical(train_data, num_clients, num_classes, beta=0.1, num_groups=10):
    """
    Two-level non-IID partition (noniid_case=6):
      Step 1: split the data across num_groups parent clients with Dirichlet(beta)
      Step 2: split each parent client's data evenly by class into sub_per_group child clients
    Returns the data indices of num_clients clients
    """
    assert num_clients % num_groups == 0, \
        f"num_clients({num_clients}) must be divisible by num_groups({num_groups})"
    sub_per_group = num_clients // num_groups

    # Step 1: Dirichlet split across num_groups parent clients
    parent_dict = separate_data(train_data, num_groups, num_classes, beta)

    # Get the labels
    if hasattr(train_data, 'targets'):
        y_train = np.array(train_data.targets)
    elif hasattr(train_data, 'labels'):
        y_train = np.array(train_data.labels)

    # Step 2: split each parent client again
    dict_users = {}
    for parent_id in range(num_groups):
        parent_indices = np.array(parent_dict[parent_id])
        parent_labels = y_train[parent_indices]

        # Empty list for each child client
        sub_indices = [[] for _ in range(sub_per_group)]

        # Split by class
        unique_classes = np.unique(parent_labels)
        for cls in unique_classes:
            cls_mask = parent_labels == cls
            cls_indices = parent_indices[cls_mask]
            np.random.shuffle(cls_indices)

            if len(cls_indices) < sub_per_group:
                # Not enough data: assign randomly to different child clients
                for idx in cls_indices:
                    target_sub = np.random.randint(0, sub_per_group)
                    sub_indices[target_sub].append(idx)
            else:
                # Split evenly
                splits = np.array_split(cls_indices, sub_per_group)
                for sub_id, split in enumerate(splits):
                    sub_indices[sub_id].extend(split.tolist())

        # Write into the final dict, keys from 0 to num_clients-1
        for sub_id in range(sub_per_group):
            global_id = parent_id * sub_per_group + sub_id
            np.random.shuffle(sub_indices[sub_id])  # Shuffle within each child client
            dict_users[global_id] = sub_indices[sub_id]

    # Print statistics
    record_net_data_stats(y_train, dict_users)

    for key in dict_users:
        dict_users[key] = [int(x) for x in dict_users[key]]

    return dict_users

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():

        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp


    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))

    return net_cls_counts
