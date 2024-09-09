import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def compute_variance_online(gradients):
    n = 0
    mean = None
    M2 = None

    for grad in gradients:
        grad_np = grad.cpu().numpy()
        if mean is None:
            mean = np.zeros_like(grad_np)
            M2 = np.zeros_like(grad_np)

        n += 1
        delta = grad_np - mean
        mean += delta / n
        delta2 = grad_np - mean
        M2 += delta * delta2

    variance = M2 / (n - 1) if n > 1 else M2
    return torch.from_numpy(variance).float().to(gradients[0].device)

def run_experiment(use_noise_injection, num_epochs=1000, gradient_accumulation_steps=4, eta=0.01):
    model = SimpleModel()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    gradient_dict = {p: [] for p in model.parameters()}
    effective_step_sizes = []

    for epoch in range(num_epochs):
        # Generate random input and target
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)

        # Forward pass
        output = model(x)
        loss = criterion(output, y)

        # Backward pass
        loss = loss / gradient_accumulation_steps
        loss.backward()

        # Collect gradients
        for p in model.parameters():
            if p.grad is not None:
                gradient_dict[p].append(p.grad.view(-1).detach())

        if (epoch + 1) % gradient_accumulation_steps == 0:
            if use_noise_injection:
                # Compute gradient variance and inject noise
                for p in model.parameters():
                    if p.grad is not None:
                        gradient_variance = compute_variance_online(gradient_dict[p])
                        gradient_dict[p] = []

                        # Compute gradient magnitude
                        grad_magnitude = torch.norm(p.grad)

                        # Scale noise based on gradient magnitude
                        noise_scale = torch.sqrt(eta * gradient_variance.view(p.grad.shape)) * grad_magnitude
                        noise = torch.randn_like(p.grad) * noise_scale

                        # Inject noise
                        p.grad += noise

            # Compute effective step size
            effective_step = 0
            for p in model.parameters():
                if p.grad is not None:
                    effective_step += (optimizer.param_groups[0]['lr'] * p.grad.norm()).item()
            effective_step_sizes.append(effective_step)

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

    return effective_step_sizes

# Run experiments
adamw_steps = run_experiment(use_noise_injection=False)
noise_injection_steps = run_experiment(use_noise_injection=True)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(adamw_steps, label='AdamW')
plt.plot(noise_injection_steps, label='AdamW with Noise Injection')
plt.xlabel('Gradient Updates')
plt.ylabel('Effective Step Size')
plt.title('Comparison of Effective Step Sizes')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.savefig('effective_step_sizes.png')
plt.close()

# Compute statistics
adamw_mean = np.mean(adamw_steps)
adamw_std = np.std(adamw_steps)
noise_mean = np.mean(noise_injection_steps)
noise_std = np.std(noise_injection_steps)

print(f"AdamW - Mean: {adamw_mean:.4f}, Std: {adamw_std:.4f}")
print(f"Noise Injection - Mean: {noise_mean:.4f}, Std: {noise_std:.4f}")
