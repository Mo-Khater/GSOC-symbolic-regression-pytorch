import torch
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
from EquationSearch import equation_search
# ---- minimal dataset ----

class Dataset:
    def __init__(self, X: torch.Tensor, y: torch.Tensor, weights: torch.Tensor | None = None):
        self.X = X
        self.y = y
        self.weights = weights


def main() -> None:
    # ---- build a simple regression dataset ----
    torch.manual_seed(0)
    X = torch.linspace(1, 5.0, 200).unsqueeze(1)
    # target: y = 0.5 * x^2 + 2*x + 1
    true_y = 0.5 * torch.sin(X[:, 0]) * X[:, 0] + 2.0 * torch.log(X[:, 0]) + 1.0

    # small noise
    noise = 0.05 * torch.randn_like(true_y)
    y = true_y + noise
    true_loss = ((true_y - y) ** 2).mean().item()
    print("True function loss (MSE vs noisy y):", true_loss)
    state = equation_search(X, y, niterations=3, nout=1)

    best_overall = min(state.hof[0].values(), key=lambda c: c.cost)
    print("Best evolved candidate loss:", best_overall.loss)
    print("Best evolved candidate:", best_overall.tree.to_string())

    # ---- plot real vs predicted ----
    with torch.no_grad():
        y_pred = best_overall.tree.forward(X)

    plt.figure(figsize=(8, 5))
    plt.plot(X[:, 0].cpu(), true_y.cpu(), label="True equation", linewidth=2)
    plt.plot(X[:, 0].cpu(), y_pred.cpu(), label="Predicted equation", linewidth=2)
    plt.scatter(X[:, 0].cpu(), y.cpu(), s=12, alpha=0.4, label="Noisy samples")
    plt.xlabel("x0")
    plt.ylabel("y")
    plt.title("True vs Predicted Equation")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    freeze_support()
    main()
