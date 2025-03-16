from torch import Tensor, concat, nn, stack


class Column2NodeLayer(nn.Module):

    def __init__(
        self,
        f_n_hidden: int = 3,
        f_hidden_size: int = 32,
        g_n_hidden: int = 3,
        g_hidden_size: int = 32,
        output_size: int = 32,
        activation_function: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        self.f_n_hidden = f_n_hidden
        self.f_hidden_size = f_hidden_size
        self.g_n_hidden = g_n_hidden
        self.g_hidden_size = g_hidden_size
        self.output_size = output_size
        self.activation_function = activation_function

        self.f = self.__generate_mlp(
            2, f_n_hidden, f_hidden_size, g_hidden_size, activation_function
        )
        self.skip_connection = nn.Sequential(
            nn.Linear(g_hidden_size, output_size)
        )
        self.g = self.__generate_mlp(
            g_hidden_size,
            g_n_hidden,
            g_hidden_size,
            output_size,
            activation_function,
        )

    def __generate_mlp(
        self,
        input_size: int,
        n_hidden: int,
        hidden_size: int,
        output_size: int,
        activation_function: nn.Module,
    ) -> nn.Module:
        sizes = [input_size] + [hidden_size] * n_hidden + [output_size]
        layers = []
        for in_size, out_size in zip(sizes[0:-1], sizes[1:]):
            layers += [nn.Linear(in_size, out_size), activation_function]
        return nn.Sequential(*layers)

    def forward(self, X: Tensor, y: Tensor) -> Tensor:
        assert (
            X.shape[0] == y.shape[0]
        ), "Lengths of X and y should be the same"
        feature_target_pairs = stack(
            [X.reshape(-1), y.repeat(X.shape[1])], dim=1
        )
        f_out = (
            self.f(feature_target_pairs)
            .reshape(X.shape[1], -1, self.g_hidden_size)
            .mean(dim=1)
        )
        return self.g(f_out) + self.skip_connection(f_out)


class MomentumLayer(nn.Module):

    def __init__(
        self,
        hidden_size: int = 32,
        n_hidden: int = 3,
        output_size: int = 32,
        activation_function: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.output_size = output_size
        self.activation_function = activation_function

        self.initial_linear = nn.Linear(2, hidden_size)
        self.hidden_linears = nn.ModuleList(
            [nn.Linear(hidden_size + 1, hidden_size) for _ in range(n_hidden)]
        )
        self.output_linear = nn.Linear(hidden_size + 1, output_size)

    def forward(self, X: Tensor, y: Tensor) -> Tensor:
        assert (
            X.shape[0] == y.shape[0]
        ), "Lengths of X and y should be the same"
        feature_target_pairs = stack(
            [X.reshape(-1), y.repeat(X.shape[1])], dim=1
        )
        output = self.activation_function(
            self.initial_linear(feature_target_pairs).reshape(
                X.shape[1], -1, self.hidden_size
            )
        ).mean(dim=1)
        for i, layer in enumerate(self.hidden_linears):
            output = self.__process_momentum_iteration(layer, i, X, output)
        output = self.__process_momentum_iteration(
            self.output_linear, len(self.hidden_linears), X, output
        )
        return output

    def __process_momentum_iteration(
        self, layer: nn.Module, power: int, X: Tensor, output: Tensor
    ) -> Tensor:
        output = (
            concat(
                [
                    output.repeat_interleave(X.shape[0], dim=0),
                    X.T.reshape(-1, 1) ** power,
                ],
                dim=1,
            )
            .reshape(X.shape[1], X.shape[0], -1)
            .mean(dim=1)
        )
        return self.activation_function(layer(output))
