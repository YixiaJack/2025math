"""
Problem 9: Binomial Coefficients
(a) Compute a_{n,k} via recurrence, check a_{n,n}=1
(b) Compute E(k) = (1/2^n) sum(k * a_{n,k}) = n/2
(c) Plot bell-shaped curves a_{n,k}/M_n
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def binom_coeffs(n, dtype=np.float64):
    a = np.zeros(n + 1, dtype=dtype)
    a[0] = dtype(1.0)
    for k in range(n):
        a[k+1] = a[k] * dtype(n - k) / dtype(k + 1)
    return a

# === Part (a) ===
print("=" * 80)
print("Part (a): Binomial coefficients via recurrence")
print("=" * 80)
print(f"{'n':>6s}  {'max (double)':>15s}  {'a_nn (double)':>18s}  "
      f"{'max (single)':>15s}  {'a_nn (single)':>18s}")
print("-" * 78)
for n in [10, 20, 30, 50, 100, 130, 150, 200, 500, 1000]:
    ad = binom_coeffs(n, np.float64)
    as_ = binom_coeffs(n, np.float32)
    print(f"{n:6d}  {np.max(ad):15.6e}  {ad[n]:18.10e}  "
          f"{np.max(as_):15.6e}  {as_[n]:18.10e}")

# === Part (b) ===
print("\n" + "=" * 80)
print("Part (b): E(k) = (1/2^n) sum(k * a_{n,k}) = n/2")
print("=" * 80)
print(f"{'n':>6s}  {'E(k)':>20s}  {'n/2':>10s}  {'|error|':>15s}  {'M_n':>20s}")
print("-" * 76)
for n in [10, 20, 50, 100, 200, 500, 1000, 1020, 1025, 1030]:
    a = binom_coeffs(n)
    Mn = np.max(a)
    ks = np.arange(n + 1, dtype=np.float64)
    two_n = np.float64(2.0) ** np.float64(n)
    Ek = np.sum(ks * a) / two_n
    exact = n / 2.0
    err = abs(Ek - exact) if np.isfinite(Ek) else float('inf')
    print(f"{n:6d}  {Ek:20.10e}  {exact:10.1f}  {err:15.6e}  {Mn:20.6e}")

# === Part (c) ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for n, style in [(10, 'b-'), (20, 'r--'), (50, 'g-.')]:
    a = binom_coeffs(n)
    Mn = np.max(a)
    ks = np.arange(n + 1)
    axes[0].plot(ks, a / Mn, style, linewidth=1.5, label=f'n = {n}')
    z = (ks - n / 2.0) / np.sqrt(n / 4.0)
    axes[1].plot(z, a / Mn, style, linewidth=1.5, label=f'n = {n}')

z_c = np.linspace(-4, 4, 200)
axes[1].plot(z_c, np.exp(-z_c**2 / 2), 'k:', lw=1, alpha=0.5, label='Gaussian')

axes[0].set_xlabel('k'); axes[0].set_ylabel(r'$a_{n,k}/M_n$')
axes[0].set_title('Binomial Coefficients (raw k)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel(r'$(k - n/2)/\sqrt{n/4}$'); axes[1].set_ylabel(r'$a_{n,k}/M_n$')
axes[1].set_title('Standardized (approaching Gaussian)')
axes[1].legend(); axes[1].grid(True, alpha=0.3); axes[1].set_xlim(-4, 4)

plt.tight_layout()
plt.savefig('problem9c_plot.png', dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for n, style in [(10, 'b-'), (20, 'r--'), (50, 'g-.')]:
    a = binom_coeffs(n)
    Mn = np.max(a)
    ks = np.arange(n+1)
    # Left: plot over chosen k range near n/2
    mu = n / 2.0
    sigma = np.sqrt(n / 4.0)
    k_lo = max(0, int(mu - 4*sigma))
    k_hi = min(n, int(mu + 4*sigma))
    mask = (ks >= k_lo) & (ks <= k_hi)
    axes[0].plot(ks[mask] - mu, a[mask]/Mn, style, linewidth=1.5, label=f'n = {n}')
    # Right: standardized variable
    z = (ks - mu) / sigma
    axes[1].plot(z, a/Mn, style, linewidth=1.5, label=f'n = {n}')

z_c = np.linspace(-4, 4, 200)
axes[1].plot(z_c, np.exp(-z_c**2/2), 'k:', lw=1, alpha=0.5, label='Gaussian')

axes[0].set_xlabel('$k - n/2$'); axes[0].set_ylabel(r'$a_{n,k}/M_n$')
axes[0].set_title('Centered at $n/2$')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel(r'$(k - n/2)/\sqrt{n/4}$'); axes[1].set_ylabel(r'$a_{n,k}/M_n$')
axes[1].set_title('Standardized (approaching Gaussian)')
axes[1].legend(); axes[1].grid(True, alpha=0.3); axes[1].set_xlim(-4, 4)

plt.tight_layout()
plt.savefig('problem9c_plot.png', dpi=150, bbox_inches='tight')
