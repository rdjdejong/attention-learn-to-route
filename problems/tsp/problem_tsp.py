from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search


class TSP(object):

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        arranged_tensor = torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi)

        assert (arranged_tensor == pi.data.sort(1)[0]).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096,
                    dynamic=False, probability=0.8):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8,
            dynamic=dynamic, prob=probability
        )

        return beam_search(state, beam_size, propose_expansions)


class TSPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0,
                 distribution=None, dynamic=False, probability=0.2):
        super(TSPDataset, self).__init__()

        self.dynamic = dynamic

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                if self.dynamic:
                    data, revealed = data
                    self.revealed = [torch.IntTensor(row) for row in (revealed[offset:offset+num_samples])]

                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            if self.dynamic:
                dyn_size = int(size * 1.5)

                # Sample points randomly in [0, 1] square
                self.data = [torch.FloatTensor(dyn_size, 2).uniform_(0, 1) for i in range(num_samples)]

                # Sample random numbers
                rand = torch.rand((num_samples, dyn_size))

                # remove any zeros
                while (rand == 0).any().item():
                    idx = ((rand == 0).nonzero(as_tuple=True))
                    rand[idx] = torch.rand(idx[0].shape)

                # Calculate the number of nodes revealed using log of the probability
                # rand < prob**nodes_revealed
                log_base = torch.log(torch.tensor([probability]))
                nodes_revealed = torch.floor(torch.log(rand) / log_base).int()

                # Mask all nodes added above the max
                max_reached = torch.cumsum(nodes_revealed, dim=1) + size
                max_reached[max_reached > dyn_size] = dyn_size

                # Mask all nodes that won't be reached
                n_nodes = torch.full((num_samples, size), size)
                dyn_nodes = torch.range(size + 1, dyn_size).repeat(num_samples, 1)
                n_nodes = torch.cat((n_nodes, dyn_nodes), dim=1)

                # Mask the nodes that wouldn't be revealed
                cutoff = torch.argmax((max_reached < n_nodes).double(), axis=1)
                for i, idx in enumerate(cutoff):
                    if idx > 0:
                        max_reached[i][idx:] = idx

                self.revealed = [x for x in max_reached]
            else:
                # Sample points randomly in [0, 1] square
                self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.dynamic:
            return {"loc": self.data[idx], "revealed": self.revealed[idx]}
        return self.data[idx]
