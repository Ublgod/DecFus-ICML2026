import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import copy
import numpy as np
import random
import swanlab
from models.Update import DatasetSplit
from utils.utils import save_result
from models.test import test_img
import math
from collections import deque


class DecFusClient(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs),
                                    batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, current_lr):
        net.to(self.args.device)
        net.train()

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=current_lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=current_lr)

        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                model_output = net(images)
                loss = self.loss_func(model_output['output'], labels)
                loss.backward()
                optimizer.step()

        return net.state_dict()


class VarianceStabilityDetector:
    """Variance stability detector based on stability relative to a reference value."""

    def __init__(self, window_size=8, stability_threshold=0.1, protection_rounds=30, ref_start_round=0.1, ref_end_round=0.2):
        self.window_size = window_size
        self.stability_threshold = stability_threshold  # stability threshold
        self.protection_rounds = protection_rounds
        self.variance_history = deque(maxlen=window_size)
        self.is_stable = False
        self.stable_round = None

        self.ref_start_round = ref_start_round
        self.ref_end_round = ref_end_round

        self.reference_variances = []
        self.reference_mean = None



    def update(self, current_variance, current_round):
        """Update the variance history and check for stability."""
        self.variance_history.append(current_variance)

        # Build the reference baseline (mean variance over the reference rounds)
        if self.ref_start_round <= current_round <= self.ref_end_round:
            self.reference_variances.append(current_variance)
        if self.reference_mean is None and current_round > self.ref_end_round:
            # Make sure data was collected during the reference period
            if self.reference_variances:
                self.reference_mean = np.mean(self.reference_variances)
            else:
                # Fallback: if no data was collected for some reason (e.g. the window is too small),
                # use a variance from the history as the reference so that the algorithm can continue
                if self.variance_history:
                    self.reference_mean = self.variance_history[5]

            # Once the reference is set, clamp it from below so that it cannot be too small
            if self.reference_mean is not None:
                self.reference_mean = max(self.reference_mean, 1e-6)

        # Never declare stability during the protection period
        if current_round < self.protection_rounds:
            return False

        # Once stable, stay stable
        if self.is_stable:
            return True

        # Check for stability
        if len(self.variance_history) >= self.window_size and self.reference_mean is not None:
            variances = list(self.variance_history)

            # Standard deviation over the current window
            current_std = np.std(variances)

            # Relative stability: ratio of the standard deviation to the reference mean
            relative_stability = current_std / self.reference_mean

            # Stable once the relative stability drops below the threshold
            if relative_stability < self.stability_threshold:
                self.is_stable = True
                self.stable_round = current_round
                return True

        return False

    def get_relative_stability(self):
        """Return the current relative stability value."""
        if len(self.variance_history) >= self.window_size and self.reference_mean is not None:
            variances = list(self.variance_history)
            current_std = np.std(variances)
            return current_std / self.reference_mean
        return float('inf')


def adaptive_sigmoid_threshold_decay(current_round, args, stability_detector):
    """
    Adaptive cutoff decay driven by variance stability.

    Args:
    current_round: current training round
    args: argument object
    stability_detector: variance stability detector
    """
    initial_threshold = args.initial_threshold
    final_threshold = args.final_threshold
    total_epochs = args.epochs

    if stability_detector.is_stable:
        # After stability is detected, the decay progress is (t - t_s) / (kappa * T), as in Eq. 6 of the paper
        stable_round = stability_detector.stable_round * args.test_round
        rounds_since_stable = current_round - stable_round
        descent_rounds = total_epochs * args.variance_descent_ratio
        descent_progress = rounds_since_stable / descent_rounds

        # Apply the sigmoid function
        steepness = args.sigmoid_steepness
        midpoint = args.sigmoid_midpoint

        sigmoid_value = 1 / (1 + math.exp(-steepness * (descent_progress - midpoint)))
        current_threshold = initial_threshold - sigmoid_value * (initial_threshold - final_threshold)
    else:
        # Before stability is detected, keep the initial cutoff (the detector never fires during the protection period)
        descent_progress = 0.0
        current_threshold = initial_threshold

    # Record for logging
    args.current_threshold = current_threshold
    args.virtual_progress = descent_progress

    return current_threshold


def compute_parameter_similarity(param1, param2):
    """Cosine similarity between two parameter tensors."""
    if torch.is_tensor(param1) and torch.is_tensor(param2):
        param1_flat = param1.view(-1)
        param2_flat = param2.view(-1)
        similarity = F.cosine_similarity(
            param1_flat.unsqueeze(0),
            param2_flat.unsqueeze(0)
        )
        return similarity.item()
    return 0.0


def DecFus(args, net_glob, dataset_train, dataset_test, dict_users):
    """DecFus: Decentralized Layer-wise Fusion with dynamic exploration and exploitation."""
    net_glob.train()

    # Initialize records
    acc = []
    loss = []
    train_loss = []

    args.current_threshold = args.initial_threshold

    # One independent variance stability detector per client
    # Note: detectors are only updated every test_round rounds, so protection_rounds is scaled accordingly
    variance_detectors = []
    for i in range(args.num_users):
        variance_detectors.append(VarianceStabilityDetector(
            window_size=args.variance_window_size,
            stability_threshold=args.variance_stability_threshold,  # stability threshold argument
            protection_rounds=int(args.epochs * args.variance_protection_ratio / args.test_round),  # converted to test rounds
            ref_start_round=int(args.epochs * args.variance_reference_start_ratio / args.test_round),
            ref_end_round=int(args.epochs * args.variance_reference_end_ratio / args.test_round)
        ))

    # Network topology
    network_topology = {
        1: [2, 8, 9, 5],
        2: [1, 6, 10, 3],
        3: [2, 8, 10, 4],
        4: [3, 10, 5, 7],
        5: [4, 10, 1, 6],
        6: [5, 2, 9, 7],
        7: [8, 9, 4, 6],
        8: [1, 3, 9, 7],
        9: [8, 7, 1, 6],
        10: [2, 3, 4, 5]
    }

    # Initialize client models and sample counts
    client_models = []
    sample_nums = []
    for i in range(args.num_users):
        client_models.append(copy.deepcopy(net_glob.state_dict()))
        sample_nums.append(len(dict_users[i]))

    param_keys = list(client_models[0].keys())
    current_lr = args.lr

    # Per-client cutoff values
    client_thresholds = [args.initial_threshold] * args.num_users

    # Main training loop
    for iter in range(args.epochs):
        print('*' * 80)
        print('Round {:3d}'.format(iter))

        # Local training
        for idx in range(args.num_users):
            net_local = copy.deepcopy(net_glob)
            net_local.load_state_dict(client_models[idx])
            local = DecFusClient(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(net=net_local, current_lr=current_lr)
            client_models[idx] = copy.deepcopy(w)

        # Evaluate before aggregation (performance right after local training)
        if iter % args.test_round == 0:
            avg_acc, avg_loss, avg_train_loss = 0, 0, 0

            for idx in range(args.num_users):
                net_local = copy.deepcopy(net_glob)
                net_local.load_state_dict(client_models[idx])
                item_acc, item_loss = test_with_loss(net_local, dataset_test, args)
                _, train_item_loss = test_with_loss(net_local, dataset_train, args)

                avg_acc += item_acc
                avg_loss += item_loss
                avg_train_loss += train_item_loss

            avg_acc /= args.num_users
            avg_loss /= args.num_users
            avg_train_loss /= args.num_users

            acc.append(avg_acc)
            loss.append(avg_loss)
            train_loss.append(avg_train_loss)

        # Similarity-based adaptive aggregation
        new_client_models = []
        for idx in range(args.num_users):
            neighbors = network_topology[idx + 1]
            current_model = client_models[idx]
            new_model = copy.deepcopy(current_model)

            # Use the stored cutoff (only updated every test_round rounds)
            individual_threshold = client_thresholds[idx]

            # For each parameter tensor, find the least similar neighbor
            lowest_similarities = []

            for param_name in param_keys:
                if 'num_batches_tracked' in param_name:
                    continue

                # Similarity between this tensor and the same tensor of every neighbor
                param_similarities = []
                for neighbor in neighbors:
                    neighbor_model = client_models[neighbor - 1]
                    similarity = compute_parameter_similarity(
                        current_model[param_name],
                        neighbor_model[param_name]
                    )
                    param_similarities.append({
                        'neighbor': neighbor,
                        'param_name': param_name,
                        'similarity': similarity
                    })

                # Neighbor with the lowest similarity
                lowest_sim_item = min(param_similarities, key=lambda x: x['similarity'])
                lowest_similarities.append(lowest_sim_item)

            if lowest_similarities:
                # Similarity cutoff value from this client's own cutoff percentile
                similarity_values = [item['similarity'] for item in lowest_similarities]
                threshold = np.percentile(similarity_values, individual_threshold * 100)

                # Update parameters
                for sim_item in lowest_similarities:
                    neighbor = sim_item['neighbor']
                    param_name = sim_item['param_name']
                    similarity = sim_item['similarity']

                    if similarity > threshold:
                        # Similarity above the cutoff: average with all neighbors, weighted by sample count
                        weighted_param = current_model[param_name].clone() * sample_nums[idx]
                        total_samples = sample_nums[idx]

                        for n in neighbors:
                            weighted_param += client_models[n - 1][param_name].clone() * sample_nums[n - 1]
                            total_samples += sample_nums[n - 1]

                        new_model[param_name] = weighted_param / total_samples
                    else:
                        if random.random() < args.exchange_prob:  # with probability exchange_prob (default 0.4)
                            # take the tensor from the least similar neighbor
                            new_model[param_name] = client_models[neighbor - 1][param_name].clone()
                        else:  # otherwise (default 0.6)
                            # pick uniformly among itself and the remaining neighbors
                            possible_choices = [idx]
                            for n in neighbors:
                                if n != neighbor:
                                    possible_choices.append(n - 1)

                            random_idx = random.choice(possible_choices)
                            if random_idx == idx:
                                new_model[param_name] = current_model[param_name].clone()
                            else:
                                new_model[param_name] = client_models[random_idx][param_name].clone()

            new_client_models.append(new_model)

        # Update client models
        client_models = new_client_models

        # After aggregation, compute the variance and update the cutoff
        if iter % args.test_round == 0:
            # Training loss after aggregation (used for variance detection)
            train_losses_after_exchange = []
            for idx in range(args.num_users):
                net_local = copy.deepcopy(net_glob)
                net_local.load_state_dict(client_models[idx])
                local_dataset = DatasetSplit(dataset_train, dict_users[idx])
                _, train_item_loss = test_with_loss(net_local, local_dataset, args)
                train_losses_after_exchange.append(
                    train_item_loss.item() if torch.is_tensor(train_item_loss) else train_item_loss
                )

            # Each client computes the variance over its neighborhood, then updates its detector and cutoff
            neighbor_train_variances = []
            for idx in range(args.num_users):
                # Neighborhood (including the client itself)
                neighbors = network_topology[idx + 1]
                neighbor_indices = [idx] + [n - 1 for n in neighbors]

                # Neighborhood variance of the post-aggregation training loss
                neighbor_train_losses = [train_losses_after_exchange[i] for i in neighbor_indices]
                neighbor_train_var = np.var(neighbor_train_losses)
                neighbor_train_variances.append(neighbor_train_var)

                # Update the variance detector (the round passed in is iter // args.test_round)
                # update() sets the detector's internal is_stable flag
                variance_detectors[idx].update(neighbor_train_var, iter // args.test_round)

                # Update this client's cutoff
                # adaptive_sigmoid_threshold_decay reads detector.is_stable
                individual_threshold = adaptive_sigmoid_threshold_decay(
                    current_round=iter,
                    args=args,
                    stability_detector=variance_detectors[idx]
                )
                client_thresholds[idx] = individual_threshold

            avg_neighbor_train_variance = np.mean(neighbor_train_variances)

            # Mean cutoff and number of stable clients
            avg_threshold = np.mean(client_thresholds)
            num_stable_clients = sum(1 for detector in variance_detectors if detector.is_stable)

            # Collect all metrics into a single swanlab.log call
            log_dict = {
                # Test accuracy and losses
                "round": iter,
                "average_test_accuracy": avg_acc,
                "average_test_loss": avg_loss,
                "average_train_loss": avg_train_loss,

                # Variance-related metrics
                "avg_neighbor_train_loss_variance": avg_neighbor_train_variance,
                "avg_threshold": avg_threshold,
                "num_stable_clients": num_stable_clients,
            }

            # Per-client metrics
            for client_id in range(args.num_users):
                log_dict[f"client_{client_id}_neighbor_train_variance"] = neighbor_train_variances[client_id]
                log_dict[f"client_{client_id}_threshold"] = client_thresholds[client_id]
                log_dict[f"client_{client_id}_relative_stability"] = variance_detectors[
                    client_id].get_relative_stability()  # relative stability
                log_dict[f"client_{client_id}_is_stable"] = variance_detectors[client_id].is_stable
                if variance_detectors[client_id].stable_round is not None:
                    log_dict[f"client_{client_id}_stable_round"] = variance_detectors[
                                                                       client_id].stable_round * args.test_round  # convert back to actual rounds

            swanlab.log(log_dict)

    # Save results
    save_result(acc, 'test_acc', args)
    save_result(loss, 'test_loss', args)
    save_result(train_loss, 'test_train_loss', args)


def test_with_loss(net_glob, dataset_test, args):
    """Wrapper around the test function."""
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Testing accuracy: {:.2f}".format(acc_test))
    return acc_test, loss_test
