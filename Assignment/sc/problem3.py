"""
Problem 3: Recurrence f_{j,k+1} = f_{j,k} - f_{j+1,k}
Computes e_k = f_hat_{0,k} - f_{0,k} for k=1..60, x0=1.
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_errors(x0=1.0, max_k=60):
    errors = []
    for k in range(1, max_k + 1):
        f_hat = np.array([np.sin(x0 + j * np.pi / 3.0) for j in range(k + 1)])
        for level in range(k):
            f_hat = f_hat[:-1] - f_hat[1:]
        f_exact = np.sin(x0 - k * np.pi / 3.0)
        errors.append(f_hat[0] - f_exact)
    return errors

errors = compute_errors()
ks = np.arange(1, 61)
eps = np.finfo(np.float64).eps

# Print errors
print(f"{'k':>4s}  {'e_k':>22s}  {'|e_k|':>22s}")
print("-" * 54)
for k, e in zip(ks, errors):
    print(f"{k:4d}  {e:22.15e}  {abs(e):22.15e}")

is_mono = all(abs(errors[i]) <= abs(errors[i+1]) for i in range(len(errors)-1))
print(f"\nMonotonically increasing? {is_mono}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(ks, errors, 'b.-', markersize=3, linewidth=0.8)
axes[0].set_xlabel('k'); axes[0].set_ylabel(r'$e_k$')
axes[0].set_title('Error on Linear Scale')
axes[0].axhline(y=0, color='k', linewidth=0.5)
axes[0].grid(True, alpha=0.3)

abs_err = np.array([abs(e) if abs(e) > 0 else 1e-300 for e in errors])
axes[1].semilogy(ks, abs_err, 'b.-', markersize=3, linewidth=0.8, label=r'$|e_k|$')
axes[1].semilogy(ks, 2.0**ks * eps, 'r--', linewidth=1.5, label=r'$2^k \varepsilon_{mach}$')
axes[1].set_xlabel('k'); axes[1].set_ylabel(r'$|e_k|$')
axes[1].set_title('Error on Log Scale')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(1e-18, 1e2)

plt.tight_layout()
plt.savefig('problem3_plots.png', dpi=150, bbox_inches='tight')
plt.show()
