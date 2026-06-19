import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Callable, List, Dict, Optional


class DynamicalNNTrainer:
    """
    Клас за обучение на невронна мрежа чрез спускане по градиента,
    разглеждано като дискретна динамична система.
    
    Поддържа два режима:
        - 'fixed'   : постоянна скорост на обучение η
        - 'adaptive': адаптивно обновяване на η при нарастване на загубата
    """

    def __init__(self, model: nn.Module, loss_fn: Callable,
                 x: torch.Tensor, y: torch.Tensor,
                 eta: float = 0.01, max_iter: int = 1000, tol: float = 1e-6):
        self.model = model
        self.loss_fn = loss_fn
        self.x = x
        self.y = y
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.history: List[Dict] = []

    def run(self, mode: str = "fixed", adaptive_rule: str = "loss") -> nn.Module:
        """
        Изпълнява обучението.

        Parameters
        ----------
        mode : str
            "fixed" или "adaptive"
        adaptive_rule : str
            Правило за адаптация ("loss" – намалява η при нарастване на загубата)
        """
        self.history = []
        eta = self.eta

        for k in range(self.max_iter):
            outputs = self.model(self.x)
            loss = self.loss_fn(outputs, self.y)

            self.model.zero_grad()
            loss.backward()

            # Запис на историята
            self.history.append({
                'iteration': k,
                'loss': loss.item(),
                'eta': eta,
                'grad_norm': sum(
                    p.grad.norm().item() 
                    for p in self.model.parameters() 
                    if p.grad is not None
                )
            })

            # Критерий за ранно спиране
            if k > 0 and abs(self.history[-1]['loss'] - self.history[-2]['loss']) < self.tol:
                break

            # Актуализация на параметрите
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        param -= eta * param.grad

            # Адаптивно обновяване на η
            if mode == "adaptive":
                eta = self._update_eta(eta, loss.item(), adaptive_rule)

        return self.model

    def _update_eta(self, eta: float, current_loss: float, rule: str) -> float:
        """Адаптивно правило за намаляване на скоростта на обучение."""
        if rule == "loss" and len(self.history) > 1:
            if current_loss > self.history[-2]['loss'] * 1.05:
                return eta * 0.6
        return eta

    # ===================== ВИЗУАЛИЗАЦИИ =====================

    def plot_loss(self, title: str = "Загуба по време на обучението", 
                  log_scale: bool = True) -> None:
        losses = [h['loss'] for h in self.history]
        plt.figure(figsize=(9, 5))
        plt.plot(losses, label='Loss', color='blue', linewidth=1.8)
        if log_scale:
            plt.yscale('log')
            plt.ylabel("Загуба (log)")
        else:
            plt.ylabel("Загуба (Loss)")
        plt.xlabel("Итерация")
        plt.title(title)
        plt.grid(False)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_eta(self, title: str = "Промяна на скоростта на обучение η") -> None:
        etas = [h['eta'] for h in self.history]
        plt.figure(figsize=(8, 5))
        plt.plot(etas, label='Learning Rate (η)', color='green', linewidth=2)
        plt.xlabel("Итерация")
        plt.ylabel("Скорост на обучение η")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_combined(self, log_scale: bool = True) -> None:
        """Комбинирана графика на загубата и скоростта на обучение."""
        iterations = [h['iteration'] for h in self.history]
        losses = [h['loss'] for h in self.history]
        etas = [h['eta'] for h in self.history]

        fig, ax1 = plt.subplots(figsize=(10, 5))

        color1 = 'tab:blue'
        ax1.set_xlabel('Итерация')
        ax1.set_ylabel('Загуба (Loss)', color=color1)
        ax1.plot(iterations, losses, color=color1, linewidth=1.8, label='Loss')
        ax1.tick_params(axis='y', labelcolor=color1)
        if log_scale:
            ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color2 = 'tab:green'
        ax2.set_ylabel('Скорост на обучение η', color=color2)
        ax2.plot(iterations, etas, color=color2, linestyle='--', linewidth=2.2, label='η')
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title("Загуба и скорост на обучение η по време на обучението")
        fig.tight_layout()
        plt.show()

    def get_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

#Функция за оценка на началната скорост на обучение
def find_suitable_eta(model: nn.Module,
                      loss_fn: Callable,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      safety_factor: float = 0.75,
                      max_iter: int = 30) -> float:
    """
    Оценява подходяща начална скорост на обучение η чрез
    приближена оценка на λ_max на Хесиана в началната точка
    (степенен метод + Hessian-vector product).
    """
    print("Изчисляване на подходяща η чрез Хесиана...")

    outputs = model(x)
    loss = loss_fn(outputs, y)
    params = list(model.parameters())

    # Градиент (създаваме граф за втори производни)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grads = torch.cat([g.view(-1) for g in grads])

    # Степенен метод за оценка на λ_max
    v = torch.randn_like(flat_grads)
    v = v / (torch.norm(v) + 1e-8)

    for _ in range(max_iter):
        hvp = torch.autograd.grad(flat_grads, params, grad_outputs=v, retain_graph=True)
        flat_hvp = torch.cat([h.view(-1) for h in hvp])
        v = flat_hvp / (torch.norm(flat_hvp) + 1e-8)

    # Финална оценка
    hvp_final = torch.autograd.grad(flat_grads, params, grad_outputs=v, retain_graph=False)
    flat_hvp_final = torch.cat([h.view(-1) for h in hvp_final])
    lambda_max = torch.dot(v, flat_hvp_final).item()

    if lambda_max <= 0:
        print(" Хесианът не е положително определен. Връщам η = 0.01")
        return 0.01

    eta_max = 2.0 / abs(lambda_max)
    good_eta = safety_factor * eta_max

    print(f" Оценена λ_max(H) ≈ {abs(lambda_max):.4f}")
    print(f" Максимална устойчива η < {eta_max:.4f}")
    print(f" Предложена начална η = {good_eta:.5f} (safety_factor = {safety_factor})")

    return good_eta

   # Примерни архитектури на невронни мрежи
class SimpleNN(nn.Module):
"""Проста двуслойна мрежа (за тестване)."""
def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(2, 8)
    self.layer2 = nn.Linear(8, 1)
    self.activation = nn.Tanh()

def forward(self, x):
    x = self.activation(self.layer1(x))
    return self.layer2(x)


class OscillatingNN(nn.Module):
"""По-дълбока мрежа (4 скрити слоя) – използвана в експериментите от Глава 4."""
def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(4, 32)
    self.layer2 = nn.Linear(32, 16)
    self.layer3 = nn.Linear(16, 8)
    self.layer4 = nn.Linear(8, 1)
    self.activation = nn.Tanh()

def forward(self, x):
    x = self.activation(self.layer1(x))
    x = self.activation(self.layer2(x))
    x = self.activation(self.layer3(x))
    return self.layer4(x)

  # Пример за изпълнение на експеримент (дълбока мрежа)
if __name__ == "__main__":
    torch.manual_seed(42)

    # === Данни ===
    X = torch.randn(300, 4) * 2.5
    y = (torch.sin(X[:, 0]) + 0.5 * X[:, 1]**2 - 0.3 * torch.tanh(X[:, 2]) +
         torch.randn(300) * 0.3).unsqueeze(1)

    model = OscillatingNN()
    loss_fn = nn.MSELoss()

    # === Оценка на начална η ===
    eta = find_suitable_eta(model, loss_fn, X, y, safety_factor=0.85)
    print(f"\nИзползвана начална η = {eta:.5f}\n")

    # === Обучение ===
    trainer = DynamicalNNTrainer(
        model=model,
        loss_fn=loss_fn,
        x=X,
        y=y,
        eta=eta,
        max_iter=500
    )

    # Режим: "fixed" или "adaptive"
    trainer.run(mode="fixed")

    # === Резултати ===
    df = trainer.get_history()
    print("Първи 15 итерации:")
    print(df[['iteration', 'loss', 'eta']].head(15))
    print("\nПоследни 10 итерации:")
    print(df[['iteration', 'loss', 'eta']].tail(10))

    # === Визуализации ===
    trainer.plot_loss(title="Загуба по време на обучението (дълбока мрежа)")
    trainer.plot_combined(log_scale=True)
