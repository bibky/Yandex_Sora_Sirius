"""Feedforward network, twin network and differential training in PyTorch (paper, section 1).

Moved without changes from DifferentialML_PyTorch.ipynb; the only difference is that
`device` and `dtype` are arguments instead of notebook-level globals.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..preprocessing import normalize_data

DEFAULT_LR_SCHEDULE = [(0.0, 1e-8), (0.2, 0.1), (0.6, 0.01), (0.9, 1e-6), (1.0, 1e-8)]


def vanilla_net(input_dim, hidden_units, hidden_layers, seed=None,
                device=None, dtype=torch.float32):
    """Fully connected net with Softplus activations and Xavier-uniform initialization."""
    if seed is not None:
        torch.manual_seed(seed)

    layers = [nn.Linear(input_dim, hidden_units), nn.Softplus()]
    for _ in range(hidden_layers - 1):
        layers += [nn.Linear(hidden_units, hidden_units), nn.Softplus()]
    layers += [nn.Linear(hidden_units, 1)]

    model = nn.Sequential(*layers).to(device=device, dtype=dtype)

    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    return model


def twin_forward(model, x):
    """Forward + d(y)/d(x) per sample (diagonal of the Jacobian)."""
    x = x.detach().clone().requires_grad_(True)
    y = model(x)
    dydx = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
    )[0]
    return y, dydx


class Neural_Approximator:
    """Standard or differential neural approximator (name kept from the authors' notebook).

    Loss of the differential network:
    alpha * MSE(y_pred, y) + beta * MSE(lambda_j * dydx_pred, lambda_j * dydx),
    alpha = 1 / (1 + lam * n), beta = 1 - alpha.
    """

    def __init__(self, x_raw, y_raw, dydx_raw=None, device=None, dtype=torch.float32):
        self.x_raw, self.y_raw, self.dydx_raw = x_raw, y_raw, dydx_raw
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.dtype = dtype
        self.model = None; self.optimizer = None
        self.m = self.n = 0
        self.differential = False
        self.weight_seed = None
        self.hidden_units = None
        self.hidden_layers = None

    def _tensor(self, a):
        return torch.tensor(a, dtype=self.dtype, device=self.device)

    def _make_model(self):
        self.model = vanilla_net(self.n, self.hidden_units, self.hidden_layers,
                                 self.weight_seed, device=self.device, dtype=self.dtype)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-8)

    def prepare(self, m, differential, lam=1.0,
                hidden_units=20, hidden_layers=4, weight_seed=None):

        (self.x_mean, self.x_std, self.x,
         self.y_mean, self.y_std, self.y,
         self.dy_dx, self.lambda_j) = normalize_data(
            self.x_raw, self.y_raw, self.dydx_raw, m)

        self.differential = differential
        self.m, self.n = self.x.shape
        self.hidden_units, self.hidden_layers = hidden_units, hidden_layers
        self.weight_seed = weight_seed

        self._make_model()

        if differential:
            if self.dy_dx is None:
                raise Exception("No differential labels for differential training")
            self.alpha = 1.0 / (1.0 + lam * self.n)
            self.beta  = 1.0 - self.alpha
            self.lambda_t = self._tensor(self.lambda_j)

        self.x_t = self._tensor(self.x)
        self.y_t = self._tensor(self.y)
        if differential:
            self.dy_dx_t = self._tensor(self.dy_dx)

    def _vanilla_epoch(self, lr, batch_size):
        for g in self.optimizer.param_groups: g['lr'] = lr
        first, last = 0, min(batch_size, self.m)
        self.model.train()
        while first < self.m:
            xb, yb = self.x_t[first:last], self.y_t[first:last]
            self.optimizer.zero_grad()
            loss = F.mse_loss(self.model(xb), yb)
            loss.backward(); self.optimizer.step()
            first, last = last, min(last + batch_size, self.m)

    def _diff_epoch(self, lr, batch_size):
        for g in self.optimizer.param_groups: g['lr'] = lr
        first, last = 0, min(batch_size, self.m)
        self.model.train()
        while first < self.m:
            xb = self.x_t[first:last]
            yb = self.y_t[first:last]
            db = self.dy_dx_t[first:last]

            self.optimizer.zero_grad()
            y_pred, dydx_pred = twin_forward(self.model, xb)

            loss_val  = F.mse_loss(y_pred, yb)
            loss_diff = F.mse_loss(dydx_pred * self.lambda_t, db * self.lambda_t)
            loss = self.alpha * loss_val + self.beta * loss_diff

            loss.backward(); self.optimizer.step()
            first, last = last, min(last + batch_size, self.m)

    def train(self, description="training", reinit=True, epochs=100,
              learning_rate_schedule=None,
              batches_per_epoch=16, min_batch_size=256,
              callback=None, callback_epochs=None):

        if learning_rate_schedule is None:
            learning_rate_schedule = DEFAULT_LR_SCHEDULE
        if callback_epochs is None:
            callback_epochs = []
        batch_size = max(min_batch_size, self.m // batches_per_epoch)
        lr_epochs, lr_rates = zip(*learning_rate_schedule)

        if reinit:
            self._make_model()

        if callback and 0 in callback_epochs:
            callback(self, 0)

        for epoch in tqdm(range(epochs), desc=description):
            lr = float(np.interp(epoch / epochs, lr_epochs, lr_rates))
            if not self.differential:
                self._vanilla_epoch(lr, batch_size)
            else:
                self._diff_epoch(lr, batch_size)
            if callback and epoch in callback_epochs:
                callback(self, epoch)

        if callback and epochs in callback_epochs:
            callback(self, epochs)

    @torch.no_grad()
    def predict_values(self, x):
        xs = (x - self.x_mean) / self.x_std
        xt = torch.as_tensor(xs, dtype=self.dtype, device=self.device)
        self.model.eval()
        ys = self.model(xt).cpu().numpy()
        return self.y_mean + self.y_std * ys

    def predict_values_and_derivs(self, x):
        xs = (x - self.x_mean) / self.x_std
        xt = torch.as_tensor(xs, dtype=self.dtype, device=self.device)
        self.model.eval()
        ys, ds = twin_forward(self.model, xt)
        ys = ys.detach().cpu().numpy(); ds = ds.detach().cpu().numpy()
        y    = self.y_mean + self.y_std * ys
        dydx = self.y_std / self.x_std * ds
        return y, dydx
