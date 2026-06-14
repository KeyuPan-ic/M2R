import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# -----------------------------------------------------------------
# 1. Helper functions
# -----------------------------------------------------------------

def s(t, p):
    return 2*(p-1) - t

def F_n(t, p, n):
    st = s(t, p)
    return (np.exp(-t/2) * t**(p/2 + n - 1)
            - np.exp(-st/2) * st**(p/2 + n - 1))

def find_cj(p, j, n_points=500):
    """
    Find c_j in (p-1, p) such that integral of F_j from p-1 to c_j
    equals integral of F_j from c_j to p.
    """
    t_full = np.linspace(p-1+1e-9, p-1e-9, n_points)
    F_vals = np.array([F_n(t, p, j) for t in t_full])
    total  = np.trapezoid(F_vals, t_full)

    def residual(c):
        t_part = np.linspace(p-1+1e-9, c, n_points)
        F_part = np.array([F_n(t, p, j) for t in t_part])
        return np.trapezoid(F_part, t_part) - total/2

    return brentq(residual, p-1+1e-6, p-1e-6, xtol=1e-8) 
    # essentially we search for a root in (p−1, p) 
    # but there are small numerical offsets to avoid the endpoints where F_n can be zero

def find_j_p(p):
    """j(p) = smallest integer > (5 + sqrt(1+6p))/2"""
    import math
    return math.floor((5 + np.sqrt(1 + 6*p)) / 2) + 1

def find_pstar(p):
    """
    Find valid p* in (max(c_0, p-1+sqrt(2)/4), min(c_1,...,c_{j(p)-1}, p-1+sqrt(2)/2)).
    Choose midpoint of valid range.
    """
    upper_limit = p - 1 + np.sqrt(2)/2
    jp = find_j_p(p)
    c0 = find_cj(p, 0)

    cjs_upper = [upper_limit]
    for j in range(1, jp):
        cjs_upper.append(find_cj(p, j))

    upper_bound = min(cjs_upper)
    lower_bound = max(c0, p - 1 + np.sqrt(2)/4)

    if lower_bound >= upper_bound:
        raise ValueError(f"Empty p* range for p={p}")

    return (lower_bound + upper_bound) / 2

# -----------------------------------------------------------------
# 2. g function
# -----------------------------------------------------------------

def g(t, p, p_star):
    """
    Piecewise linear g, symmetric about t=p-1, with g(p-2)=g(p)=0, minimum at p*.
    """
    if p-2 < t < p - 1:
        t = 2*(p-1) - t   # reflect into [p-1, p]
    if p_star <= t <= p:
        return t - p
    else:
        return 2*p_star - p - t

def g_prime(t, p, p_star):
    if t < p - 1:
        return -g_prime(2*(p-1) - t, p, p_star)
    return 1.0 if t >= p_star else -1.0

# -----------------------------------------------------------------
# 3. Parameter computation
# -----------------------------------------------------------------

def compute_params(p, n_points=10000):
    """
    Compute p*, b, A, B, a for the Shao-Strawderman estimator.
    """
    j_p = find_j_p(p)
    p_star = find_pstar(p)
    b = 1 - np.sqrt(2) * (p_star - (p-1))
    if not b < 1/2:
        raise ValueError(f"Must have b < 1/2")

    # A computed directly
    exponent = p/2 + j_p - 1
    A = 1 - np.exp(1 - b) * ((p - 2 + b) / (p - b))**exponent

    # B: min over j of specified ratio of integrals
    t = np.linspace(p-2+1e-8, p-1e-8, n_points)
    g_vals  = np.array([g(tv, p, p_star)  for tv in t])
    gp_vals = np.array([g_prime(tv, p, p_star) for tv in t])

    B_min = np.inf
    for j in range(j_p):
        num = 4 * np.trapezoid(gp_vals * np.exp(-t/2) * t**(p/2+j-1), t)
        den =     np.trapezoid(g_vals**2 * np.exp(-t/2) * t**(p/2+j-2), t)
        if abs(den) > 1e-14 and num > 0:
            B_min = min(B_min, num/den)

    B = B_min
    a_max = min(B, 2*(p-2)*b*A)
    a = 0.5 * a_max   # safely inside constraint

    return dict(p_star=p_star, b=b, A=A, B=B, a=a)

# -----------------------------------------------------------------
# 4. Estimators
# -----------------------------------------------------------------

def mle(X, **_):
    return X.copy()

def james_stein(X, **_):
    p = len(X)
    norm_sq = float(np.dot(X, X))
    if norm_sq == 0:
        return X.copy()
    return (1 - (p-2)/norm_sq) * X

def james_stein_plus(X, **_):
    p = len(X)
    norm_sq = float(np.dot(X, X))
    if norm_sq == 0:
        return X.copy()
    return max(0.0, 1 - (p-2)/norm_sq) * X

def shao_strawderman(X, p_star, a, g_func, **_):
    p = len(X)
    norm_sq = float(np.dot(X, X))
    base = james_stein_plus(X)
    if (p-2) <= norm_sq <= p and norm_sq > 1e-12:
        correction = a * g_func(norm_sq) / norm_sq * X
        return base - correction
    return base

# -----------------------------------------------------------------
# 5. Monte Carlo risk estimation
# -----------------------------------------------------------------

def empirical_risk(estimator, theta, n_sim=50000, **kwargs):
    p = len(theta)
    losses = np.zeros(n_sim)
    for i in range(n_sim):
        X = np.random.normal(theta, 1.0)
        est = estimator(X, **kwargs)
        losses[i] = np.sum((est - theta)**2)
    return losses.mean()

# -----------------------------------------------------------------
# 6. Simulation and plotting
# -----------------------------------------------------------------

def run_and_plot(p, n_sim=50000, n_theta=25):
    params = compute_params(p)
    g_func = lambda t: g(t, p, params['p_star'])
    ses = {name: [] for name in 
       ['MLE','James-Stein','James-Stein+','Shao-Strawderman']}

    print(f"\np={p}: p*={params['p_star']:.4f}, b={params['b']:.4f}, "
          f"A={params['A']:.4f}, B={params['B']:.4f}, a={params['a']:.6f}")

    theta_norms = np.linspace(0, 6, n_theta)
    risks = {name: [] for name in ['MLE','James-Stein','James-Stein+','Shao-Strawderman']}
    prop_differ_ss_jsp = []
    prop_differ_jsp_js = []

    for norm in theta_norms:
        print(f"  p={p}, ||theta||={norm:.2f}...", flush=True)
        theta = np.zeros(p)
        theta[0] = norm

        # Vectorised: sample all n_sim draws at once
        X_all = np.random.normal(theta, 1.0, size=(n_sim, p))
        norm_sqs = np.sum(X_all**2, axis=1)  # shape (n_sim,)

        # MLE losses
        losses_mle = np.sum((X_all - theta)**2, axis=1)

        # JS losses
        shrink_js = 1 - (p-2)/norm_sqs
        est_js_all = shrink_js[:,None] * X_all
        losses_js = np.sum((est_js_all - theta)**2, axis=1)

        # JS+ losses
        shrink_jsp = np.maximum(0, shrink_js)
        est_jsp_all = shrink_jsp[:,None] * X_all
        losses_jsp = np.sum((est_jsp_all - theta)**2, axis=1)

        # SS losses
        in_band = (norm_sqs >= p-2) & (norm_sqs <= p)
        g_vals = np.array([g_func(ns) for ns in norm_sqs])
        correction = np.where(in_band,
                            params['a'] * g_vals / norm_sqs,
                            0.0)
        est_ss_all = est_jsp_all - correction[:,None] * X_all
        losses_ss = np.sum((est_ss_all - theta)**2, axis=1)

        # Difference counts
        n_differ_ss  = np.sum(~np.all(np.isclose(est_ss_all,  est_jsp_all), axis=1))
        n_differ_jsp = np.sum(~np.all(np.isclose(est_jsp_all, est_js_all),  axis=1))

        risks['MLE'].append(losses_mle.mean())
        risks['James-Stein'].append(losses_js.mean())
        risks['James-Stein+'].append(losses_jsp.mean())
        risks['Shao-Strawderman'].append(losses_ss.mean())
        prop_differ_ss_jsp.append(n_differ_ss / n_sim)
        prop_differ_jsp_js.append(n_differ_jsp / n_sim)

        # Standard errors for ribbon plot
        ses['MLE'].append(losses_mle.std() / np.sqrt(n_sim))
        ses['James-Stein'].append(losses_js.std() / np.sqrt(n_sim))
        ses['James-Stein+'].append(losses_jsp.std() / np.sqrt(n_sim))
        ses['Shao-Strawderman'].append(losses_ss.std() / np.sqrt(n_sim))

    # Summary statistics (see Table 1 for the SS vs JS+ results)
    print(f"Shao-Strawderman differs from JS+ on average {100*np.mean(prop_differ_ss_jsp):.1f}% of draws")
    print(f"JS+ differs from JS on average {100*np.mean(prop_differ_jsp_js):.1f}% of draws")

    # Plot of risk curves with ribbons
    plt.figure(figsize=(10, 6))
    styles = {'MLE': ('steelblue','-',2), 'James-Stein': ('darkorange','-',2),
              'James-Stein+': ('green','-',2), 'Shao-Strawderman': ('purple','--',2)}
    for name, (color, ls, lw) in styles.items():
        r = np.array(risks[name])
        se = np.array(ses[name])
        plt.plot(theta_norms, r, label=name,
                color=color, linestyle=ls, linewidth=lw)
        plt.fill_between(theta_norms, r - 2*se, r + 2*se,
                        alpha=0.12, color=color)
    plt.axhline(y=p, color='steelblue', linestyle=':', alpha=0.4,
                label=f'MLE risk = p = {p}')
    plt.xlabel('||θ||')
    plt.ylabel('Empirical Risk')
    plt.title(f'Risk comparison of estimators  (p = {p})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'Risk_p{p}.png', dpi=150)
    plt.show()

    # Plots of the proportion of draws differing
    plt.figure(figsize=(10, 4))
    plt.plot(theta_norms, prop_differ_ss_jsp, label='Shao-Stawderman vs James-Stein+', color='purple')
    plt.plot(theta_norms, prop_differ_jsp_js, label='James-Stein+ vs James-Stein', color='green')
    plt.xlabel('||θ||')
    plt.ylabel('Proportion of draws differing')
    plt.title(f'How often do estimators differ?  (p = {p})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'Prop_differ_p{p}.png', dpi=150)
    plt.show()

    # Plots of risk differences
    plt.figure(figsize=(10, 4))
    diff_ss_jsp = np.array(risks['Shao-Strawderman']) - np.array(risks['James-Stein+'])
    diff_js_jsp = np.array(risks['James-Stein']) - np.array(risks['James-Stein+'])
    plt.plot(theta_norms, diff_ss_jsp, label='Shao-Strawderman - James-Stein+', color='purple', linestyle='--')
    plt.plot(theta_norms, diff_js_jsp, label='James-Stein - James-Stein+', color='darkorange')
    plt.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    plt.xlabel('||θ||')
    plt.ylabel('Risk difference relative to JS+')
    plt.title(f'Risk differences relative to JS+  (p = {p})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'Risk_diff_p{p}.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    np.random.seed(3)
    for p in [3, 5, 10, 50]:
        run_and_plot(p, n_sim=50000, n_theta=25)