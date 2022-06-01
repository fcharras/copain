import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from skorch import NeuralNet
from skorch.utils import TeeGenerator
from skorch.dataset import unpack_data

from copain.utils import WeightInitializer


class CopainANN(nn.Module):
    def __init__(
        self,
        n_actions,
        input_dim,
        nb_values_per_dim,
        starting_nb_embeddings,
        nb_embeddings_step,
        depth,
        embedding_size,
        hidden_dim,
        p_dropout,
        initialize_fn=None,
        initialize_fn_kwargs=None,
    ):
        super().__init__()

        self.n_actions = n_actions

        self._embedding_bag = _DynamicEmbeddingBag(
            input_dim,
            nb_values_per_dim,
            embedding_size,
            "sum",
            starting_nb_embeddings,
            nb_embeddings_step,
            initialize_fn,
            initialize_fn_kwargs,
        )

        per_value_feed_forward_steps = []
        for d in range(depth):
            in_dim = hidden_dim if (d > 0) else (embedding_size)
            out_dim = hidden_dim if (d < (depth - 1)) else n_actions
            per_value_feed_forward_steps.extend(
                [nn.Dropout(p_dropout), nn.ReLU(), nn.Linear(in_dim, out_dim)]
            )

        per_slot_feed_forward_steps = []
        for d in range(depth):
            in_dim = hidden_dim if (d > 0) else (input_dim)
            out_dim = hidden_dim if (d < (depth - 1)) else n_actions
            per_slot_feed_forward_steps.extend(
                [nn.Dropout(p_dropout), nn.ReLU(), nn.Linear(in_dim, out_dim)]
            )

        self.nb_values_per_dim = nb_values_per_dim

        self._per_value_feed_forward = nn.Sequential(*per_value_feed_forward_steps)
        self._per_slot_feed_forward = nn.Sequential(*per_slot_feed_forward_steps)

        self.apply(WeightInitializer(initialize_fn, initialize_fn_kwargs))

    def forward(self, X):
        return (self._per_value_feed_forward(self._embedding_bag(X))
                + self._per_slot_feed_forward(X/self.nb_values_per_dim))/2


class _DynamicEmbeddingBag(nn.Module):

    # TODO: improve so that are registered only memory entries for which
    # at least 2 distincts values have been observed
    def __init__(
        self,
        input_dim,
        nb_values_per_dim,
        embedding_dim,
        mode,
        starting_nb_embeddings,
        nb_embeddings_step,
        initialize_fn,
        initialize_fn_kwargs,
    ):
        super().__init__()
        # 0 is a value for "unknown"
        self._embedding_dim = embedding_dim
        self._initializer = WeightInitializer(initialize_fn, initialize_fn_kwargs)
        self._embedding_bag = nn.EmbeddingBag(
            starting_nb_embeddings, embedding_dim, mode=mode, padding_idx=0
        )
        self._mapping = torch.nn.Parameter(
            torch.zeros((input_dim, nb_values_per_dim), dtype=torch.int32),
            requires_grad=False,
        )
        self._mapping_filtered = torch.nn.Parameter(
            torch.zeros((input_dim, nb_values_per_dim), dtype=torch.int32),
            requires_grad=False,
        )
        self._max_mapping_ = 0
        self._max_mapping = 0
        self.nb_embeddings_step = nb_embeddings_step

    def update_embeddings(self):
        if self._max_mapping > self._max_mapping_:
            # ignore the data for which only one value has been mapped, since this value alone is not
            # significant.
            self._mapping_filtered[:] = self._mapping[:]
            self._mapping_filtered[(self._mapping_filtered != 0).sum(1) <= 1] = 0
            self._max_mapping_ = self._max_mapping

        needed_nb_embeddings = self._max_mapping + 1
        current_nb_embeddings = self._embedding_bag.weight.shape[0]

        if current_nb_embeddings >= needed_nb_embeddings:
            return False

        needed_nb_embeddings += self.nb_embeddings_step

        current_device = self._embedding_bag.weight.device
        self._embedding_bag.to("cpu")

        prev_embedding_bag_weight = self._embedding_bag.state_dict()["weight"]
        del self._embedding_bag

        self._embedding_bag = nn.EmbeddingBag(
            needed_nb_embeddings, self._embedding_dim, mode="sum", padding_idx=0
        )

        self._embedding_bag.apply(self._initializer)

        new_state = self._embedding_bag.state_dict()
        new_state["weight"] = torch.vstack(
            (prev_embedding_bag_weight, new_state["weight"][current_nb_embeddings:])
        )
        self._embedding_bag.load_state_dict(new_state)

        self._embedding_bag.to(current_device)

        print("%s embeddings" % str(self._embedding_bag.weight.shape[0]))

        return True

    def _detect_unindexed_data(self, X, X_remapped, row_ix):
        if (X_remapped > 0).all():
            return

        # detect the indexes who are not mapped yet and schedule their registration
        X_remapped = self._mapping[row_ix, X]
        unmapped_row, unmapped_col = torch.nonzero(X_remapped == 0, as_tuple=True)

        if not unmapped_row.shape[0]:
            return

        unmapped_value = X[unmapped_row, unmapped_col]

        unmapped_value, unmapped_col = torch.unique(
            torch.vstack((unmapped_value.long(), unmapped_col.long())), dim=1
        )

        nb_unmapped_values = unmapped_value.shape[0]
        unmapped_value_index = torch.arange(
            nb_unmapped_values, dtype=torch.int32, device=X.device
        )
        self._mapping[unmapped_col, unmapped_value] = (
            self._max_mapping + unmapped_value_index + 1
        )
        self._max_mapping += nb_unmapped_values + 1

    def forward(self, X):
        """X of size batch_size x ram_size and dtype int in (0-255)"""
        # first step: transform X to the new mapping
        row_ix = (
            torch.arange(X.shape[1], dtype=torch.int32).repeat((X.shape[0], 1)).long()
        )
        X = X.long()
        X_remapped = self._mapping_filtered[row_ix, X]
        if self.training:
            self._detect_unindexed_data(X, X_remapped, row_ix)

        return self._embedding_bag(X_remapped)

class NeuralNet(NeuralNet):

    def __init__(
            self,
            module, criterion,
            amp_enabled=True, amp_init_scale=19,
            *args, **kwargs
    ):

        super(NeuralNet, self).__init__(module, criterion, *args, **kwargs)
        self.amp_enabled = amp_enabled
        self.amp_init_scale = amp_init_scale

    def fit(self, X, y=None, **fit_params):
        if self.amp_enabled:
            self.scaler = GradScaler(init_scale=2**self.amp_init_scale)
        super().fit(X, y, **fit_params)
        return self

    def train_step(self, batch, **fit_params):
        step_accumulator = self.get_train_step_accumulator()

        step = self.train_step_single(batch, **fit_params)
        step_accumulator.store_step(step)
        if self.amp_enabled:
            self.scaler.step(self.optimizer_)
            self.scaler.update()
        else:
            self.optimizer_.step()
        self.optimizer_.zero_grad()

        return step_accumulator.get_step()

    def train_step_single(self, batch, **fit_params):
        self._set_training(True)
        Xi, yi = unpack_data(batch)
        y_pred = self.infer(Xi, **fit_params)
        with autocast(enabled=self.amp_enabled):
            loss = self.get_loss(y_pred, yi, X=Xi, training=True)

        self.scaler.scale(loss).backward() if self.amp_enabled else loss.backward()

        self.notify(
            'on_grad_computed',
            named_parameters=TeeGenerator(self.get_all_learnable_params()),
            batch=batch
        )

        return {
            # get back the loss on the last batch (independent of gradient accumulation tricks)
            'loss': loss,
            'y_pred': y_pred,
            }

    def infer(self, *args, **kwargs):
        with autocast(enabled=self.amp_enabled):
            return super(NeuralNet, self).infer(*args, **kwargs)
