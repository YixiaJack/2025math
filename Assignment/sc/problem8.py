"""
Problem 8: Fibonacci and Pib numbers
(a) Log scale plot with 1/eps markers
(b) Forward-backward Fibonacci
(c) Forward-backward Pib
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

c = 1.0 + np.sqrt(3.0) / 100.0
n_max = 100
eps_s = np.finfo(np.float32).eps
eps_d = np.finfo(np.float64).eps

# === Part (a): Forward computation & log-scale plot ===
f = np.zeros(n_max + 1); f[0] = f[1] = 1.0
p = np.zeros(n_max + 1); p[0] = p[1] = 1.0
for k in range(2, n_max + 1):
    f[k] = f[k-1] + f[k-2]
    p[k] = c * p[k-1] + p[k-2]

ns = np.arange(n_max + 1)
plt.figure(figsize=(10, 6))
plt.semilogy(ns, f, 'b-', linewidth=1.5, label='Fibonacci $f_n$')
plt.semilogy(ns, p, 'r-', linewidth=1.5, label=f'Pib $p_n$')
plt.axhline(1/eps_s, color='green', ls='--', lw=1,
            label=f'$1/\\varepsilon_{{mach}}$ single $\\approx$ {1/eps_s:.2e}')
plt.axhline(1/eps_d, color='purple', ls='--', lw=1,
            label=f'$1/\\varepsilon_{{mach}}$ double $\\approx$ {1/eps_d:.2e}')
plt.xlabel('n'); plt.ylabel('Value')
plt.title('Fibonacci and Pib Numbers (Log Scale)')
plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('problem8a_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# === Part (b): Forward-backward Fibonacci ===
def backward_fib_double(n):
    f = np.zeros(n+1, dtype=np.float64); f[0]=f[1]=1.0
    for k in range(2, n+1): f[k] = f[k-1] + f[k-2]
    fk1, fk = f[n], f[n-1]
    for k in range(n-2, -1, -1):
        fk1, fk = fk, fk1 - fk
    return abs(fk - 1.0)

def backward_fib_single(n):
    f = np.zeros(n+1, dtype=np.float32); f[0]=f[1]=np.float32(1.0)
    for k in range(2, n+1): f[k] = f[k-1] + f[k-2]
    fk1, fk = np.float32(f[n]), np.float32(f[n-1])
    for k in range(n-2, -1, -1):
        fk1, fk = fk, np.float32(fk1 - fk)
    return abs(float(fk) - 1.0)

n_d = range(2, 95); n_s = range(2, 50)
err_fib_d = [backward_fib_double(n) for n in n_d]
err_fib_s = [backward_fib_single(n) for n in n_s]

plt.figure(figsize=(10, 6))
plt.semilogy(list(n_d), err_fib_d, 'b.-', ms=3, lw=0.8, label='Double')
plt.semilogy(list(n_s), err_fib_s, 'r.-', ms=3, lw=0.8, label='Single')
plt.axhline(1.0, color='k', ls='--', lw=0.5, label='Complete loss')
plt.xlabel('n'); plt.ylabel('|recomputed $f_0$ - 1|')
plt.title('Fibonacci: Backward Recomputation Error')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('problem8b_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# === Part (c): Forward-backward Pib ===
def backward_pib_double(n):
    p = np.zeros(n+1); p[0]=p[1]=1.0
    for k in range(2, n+1): p[k] = c*p[k-1] + p[k-2]
    pk1, pk = p[n], p[n-1]
    for k in range(n-2, -1, -1):
        pk1, pk = pk, pk1 - c*pk
    return abs(pk - 1.0)

def backward_pib_single(n):
    c32 = np.float32(c)
    p = np.zeros(n+1, dtype=np.float32); p[0]=p[1]=np.float32(1.0)
    for k in range(2, n+1): p[k] = c32*p[k-1] + p[k-2]
    pk1, pk = np.float32(p[n]), np.float32(p[n-1])
    for k in range(n-2, -1, -1):
        pk1, pk = pk, np.float32(pk1 - c32*pk)
    return abs(float(pk) - 1.0)

err_pib_d = [backward_pib_double(n) for n in n_d]
err_pib_s = [backward_pib_single(n) for n in n_s]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].semilogy(list(n_d), err_fib_d, 'b.-', ms=3, lw=0.8, label='Double')
axes[0].semilogy(list(n_s), err_fib_s, 'r.-', ms=3, lw=0.8, label='Single')
axes[0].axhline(1.0, color='k', ls='--', lw=0.5)
axes[0].set_xlabel('n'); axes[0].set_ylabel('|recomputed - exact|')
axes[0].set_title('Fibonacci: Backward Error'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].semilogy(list(n_d), err_pib_d, 'b.-', ms=3, lw=0.8, label='Double')
axes[1].semilogy(list(n_s), err_pib_s, 'r.-', ms=3, lw=0.8, label='Single')
axes[1].axhline(1.0, color='k', ls='--', lw=0.5)
axes[1].set_xlabel('n'); axes[1].set_ylabel('|recomputed - exact|')
axes[1].set_title('Pib: Backward Error'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('problem8c_plot.png', dpi=150, bbox_inches='tight')
plt.show()
