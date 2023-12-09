from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Iterable, Optional, cast

import torch
from torch import Size, nn


class GenericLSTMCell(nn.Module):
    class H0ShapeFn(nn.Module, ABC):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.hidden_size = hidden_size

        @abstractmethod
        def __call__(self, input_shape: Size) -> Iterable[int]:
            ...

    def init_h_or_c(self, input: torch.Tensor) -> torch.Tensor:
        assert (
            self.h0_shape_fn is not None
        ), "h0_shape_fn can be None only when hx is provided"
        hcx = torch.zeros(
            *self.h0_shape_fn(input.shape), device=input.device, dtype=input.dtype
        )
        return hcx

    def __init__(
        self,
        h0_shape_fn: Optional[H0ShapeFn],
        layer_xh: nn.Module,
        layer_hh: nn.Module,
        dim: int = 1,
    ):
        super().__init__()
        self.h0_shape_fn = h0_shape_fn
        self.layer_xh = layer_xh
        self.layer_hh = layer_hh
        self.dim = dim

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            input (torch.Tensor): input vector
            hx (Optional[Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]], optional): hidden vector. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: hidden and context vectors
        """

        if hx is None:
            h_t = self.init_h_or_c(input)
            c_t = self.init_h_or_c(input)

        else:
            h_t, c_t = hx

        gates = self.layer_xh(input) + self.layer_hh(h_t)

        # Compute the gates (i_t, f_t, g_t, o_t)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=self.dim)

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        cy = c_t * f_t + i_t * g_t

        hy = o_t * torch.tanh(cy)

        return hy, cy


class LinearLSTMCell(GenericLSTMCell):
    class LinearH0ShapeFn(GenericLSTMCell.H0ShapeFn):
        def __init__(self, hidden_size: int) -> None:
            super().__init__(hidden_size)

        def __call__(self, input_shape: Size) -> Iterable[int]:
            # B x C
            return input_shape[0], self.hidden_size

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__(
            LinearLSTMCell.LinearH0ShapeFn(hidden_size),
            nn.Linear(input_size, hidden_size * 4, bias=True),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            dim=1,
        )


class Conv2dLSTMCell(GenericLSTMCell):
    class Conv2dH0ShapeFn(GenericLSTMCell.H0ShapeFn):
        def __init__(self, hidden_size: int) -> None:
            super().__init__(hidden_size)

        def __call__(self, input_shape: Size) -> Iterable[int]:
            # B x C x H x W
            return (input_shape[0], self.hidden_size, input_shape[2], input_shape[3])

    def __init__(self, input_size: int, hidden_size: int, kernel_size: int):
        super().__init__(
            Conv2dLSTMCell.Conv2dH0ShapeFn(hidden_size),
            nn.Conv2d(input_size, hidden_size * 4, bias=True, kernel_size=kernel_size),
            nn.Conv2d(hidden_size, hidden_size * 4, bias=True, kernel_size=kernel_size),
            dim=1,
        )


class GenericLSTM(nn.Module):
    def init_h_or_c(self, input: torch.Tensor) -> torch.Tensor:
        # B x hidden_size x ...
        input_fix: list[int] = []
        for i, x in enumerate(input.shape):
            if i != self.timestep_dim:
                input_fix.append(x)

        cell_h0_size = self.h0_shape_fn(Size(input_fix))

        # num_layers x B x hidden_size x ...
        hcx = torch.zeros(
            (self.num_layers,) + cell_h0_size, device=input.device, dtype=input.dtype
        )

        return hcx

    def __init__(
        self,
        h0_shape_fn: GenericLSTMCell.H0ShapeFn,
        layer_xh: nn.Module,
        layer_hh: nn.Module,
        num_layers: int = 2,
        timestep_dim: int = 1,
        channel_dim: int = 2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.h0_shape_fn = h0_shape_fn
        self.timestep_dim = timestep_dim

        self.layers: nn.ModuleList[GenericLSTMCell] = nn.ModuleList()
        self.layers.append(
            GenericLSTMCell(
                None,
                layer_xh,
                layer_hh,
                dim=channel_dim if channel_dim < timestep_dim else channel_dim - 1,
            )
        )

        for _ in range(1, num_layers):
            self.layers.append(
                GenericLSTMCell(
                    None,
                    layer_hh,
                    layer_hh,
                    dim=channel_dim if channel_dim < timestep_dim else channel_dim - 1,
                )
            )

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if hx is None:
            h_0 = self.init_h_or_c(input)
            c_0 = self.init_h_or_c(input)

        else:
            h_0, c_0 = hx

        outputs: list[torch.Tensor] = []
        h_x: list[torch.Tensor] = [None for _ in range(len(self.layers))]
        c_x: list[torch.Tensor] = [None for _ in range(len(self.layers))]
        for timestep in range(input.shape[self.timestep_dim]):
            input_layer = input.index_select(
                dim=self.timestep_dim,
                index=torch.tensor([timestep], requires_grad=False).to(input.device),
            ).squeeze(self.timestep_dim)

            for layer_idx, layer in enumerate(self.layers):
                # obtain features from cell.
                # B x hidden_size
                if timestep == 0:
                    hx_layer = h_0[layer_idx]
                    cx_layer = c_0[layer_idx]

                else:
                    hx_layer = h_x[layer_idx]
                    cx_layer = c_x[layer_idx]

                hx_cell, cx_cell = layer(input_layer, (hx_layer, cx_layer))

                # set hx as the new input for the next layer cell
                input_layer = hx_cell

                # change the hx for the current hx cx
                h_x[layer_idx] = hx_cell
                c_x[layer_idx] = cx_cell

            output_timestep = input_layer
            outputs.append(output_timestep)

        # stack all the timesteps
        # B x seq.length x hidden_size x ...
        return torch.stack(outputs, dim=1), (
            torch.stack(h_x, axis=0),
            torch.stack(c_x, axis=0),
        )


class LinearLSTM(GenericLSTM):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        timestep_dim: int = 1,
        channel_dim: int = 2,
    ):
        super().__init__(
            h0_shape_fn=LinearLSTMCell.LinearH0ShapeFn(hidden_size),
            layer_xh=nn.Linear(input_size, hidden_size * 4, bias=True),
            layer_hh=nn.Linear(hidden_size, hidden_size * 4, bias=True),
            num_layers=num_layers,
            timestep_dim=timestep_dim,
            channel_dim=channel_dim,
        )


class Conv2dLSTM(GenericLSTM):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        num_layers: int = 2,
        timestep_dim: int = 1,
        channel_dim: int = 2,
    ):
        super().__init__(
            h0_shape_fn=Conv2dLSTMCell.Conv2dH0ShapeFn(hidden_size),
            layer_xh=nn.Conv2d(
                input_size,
                hidden_size * 4,
                bias=True,
                kernel_size=kernel_size,
                padding=1,
            ),
            layer_hh=nn.Conv2d(
                hidden_size,
                hidden_size * 4,
                bias=True,
                kernel_size=kernel_size,
                padding=1,
            ),
            num_layers=num_layers,
            timestep_dim=timestep_dim,
            channel_dim=channel_dim,
        )
