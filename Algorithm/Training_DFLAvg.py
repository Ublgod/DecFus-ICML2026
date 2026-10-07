import torch
from torch.utils.data import DataLoader
from torch import nn
import copy
import swanlab
from models.Update import DatasetSplit
from utils.utils import save_result
from models.test import test_img



class DecentralizedFedAvg(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs),
                                    batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, current_lr):
        """Local training of a single client."""
        net.to(self.args.device)
        net.train()

        # Set up the optimizer
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

def DFLAvg(args, net_glob, dataset_train, dataset_test, dict_users):
    """
    Decentralized Federated Averaging (DFLAvg)
    """
    net_glob.train()

    # Initialize records
    acc = []
    loss = []

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

    # Initialize each client's model
    client_models = []
    for i in range(args.num_users):
        client_models.append(copy.deepcopy(net_glob.state_dict()))

    current_lr = args.lr

    # Main training loop
    for iter in range(args.epochs):
        print('*' * 80)
        print('Round {:3d}'.format(iter))

        # Local training
        for idx in range(args.num_users):
            net_local = copy.deepcopy(net_glob)
            net_local.load_state_dict(client_models[idx])
            local = DecentralizedFedAvg(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(net=net_local, current_lr=current_lr)
            client_models[idx] = copy.deepcopy(w)

        # Evaluate every test_round rounds
        if iter % args.test_round == 0 or iter == 990:
            # Evaluate every client and average the results
            avg_acc = 0
            avg_loss = 0
            for idx in range(args.num_users):
                net_local = copy.deepcopy(net_glob)
                net_local.load_state_dict(client_models[idx])
                item_acc, item_loss = test_with_loss(net_local, dataset_test, args)
                # Use .item() to extract scalar values
                avg_acc += item_acc.item() if torch.is_tensor(item_acc) else item_acc
                avg_loss += item_loss.item() if torch.is_tensor(item_loss) else item_loss

            avg_acc /= args.num_users
            avg_loss /= args.num_users

            swanlab.log({
                "round": iter,
                "average_test_accuracy": avg_acc,
                "average_test_loss": avg_loss
            })

            acc.append(avg_acc)
            loss.append(avg_loss)

        # Neighborhood averaging
        new_client_models = []
        for idx in range(args.num_users):
            neighbors = network_topology[idx + 1]
            neighbor_models = [client_models[idx]]  # including itself
            for n in neighbors:
                neighbor_models.append(client_models[n - 1])

            # Compute the average
            avg_model = copy.deepcopy(client_models[idx])
            for k in avg_model.keys():
                if avg_model[k].dtype in (torch.int32, torch.int64, torch.bool):
                    continue  # skip: not aggregated
                avg_model[k] = torch.zeros_like(avg_model[k])
                for model in neighbor_models:
                    avg_model[k] += model[k]
                avg_model[k] = avg_model[k] / len(neighbor_models)
            new_client_models.append(avg_model)

        client_models = new_client_models
    # Save results
    save_result(acc, 'test_acc', args)
    save_result(loss, 'test_loss', args)


def test_with_loss(net_glob, dataset_test, args):
    """Test function wrapping test_model."""
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Testing accuracy: {:.2f}".format(acc_test))
    return acc_test, loss_test
