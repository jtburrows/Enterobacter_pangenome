#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
optNMF — OptICA-style selection for standard NMF with
           conservation on L (columns), dominance metrics,
           robust mass fractions, and (optional) dual K-fold cross-validation.

"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot as plotly_plot

# ---------------------------------------------------------------------
# Utilities & constants
# ---------------------------------------------------------------------

EPS = 1e-12


def parse_list_or_slice(spec: str) -> List[int]:
    s = spec.strip()
    if ":" in s:
        parts = [p for p in s.split(":") if p != ""]
        if len(parts) not in (2, 3):
            raise ValueError(f"Bad slice spec: {spec!r}")
        start = int(parts[0]); end = int(parts[1]); step = int(parts[2]) if len(parts) == 3 else 1
        if step == 0:
            raise ValueError("Slice step cannot be 0")
        return list(range(start, end, step))
    return [int(x) for x in s.split(",") if x != ""]


@dataclass
class FitResult:
    # identity
    rank: int
    seed: int
    # factorization
    L: np.ndarray               # (n_rows, rank)
    A: np.ndarray               # (rank, n_cols)
    L_norm: np.ndarray
    A_norm: np.ndarray
    p99_scales: np.ndarray
    # reconstruction & losses
    rss: float
    beta_loss_mean: float       # mean β-div per entry (ref only)
    # likelihoods (train/held-out based on mask)
    loglike_train: float
    loglike_test: Optional[float]
    aic_train: float
    bic_train: float
    aic_test: Optional[float]
    bic_test: Optional[float]
    # per-rank observed counts for correct perplexity averaging
    n_obs_train: int = 0
    n_obs_test: Optional[int] = None
    # binarization & confusion (held-out-aware elsewhere, but we use FULL data for metrics)
    thresholds_L: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    thresholds_A: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    L_bin: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int8))
    A_bin: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int8))
    conf: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=int))
    # GMM diagnostics
    gmm_k_L: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=int))
    gmm_k_A: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=int))
    gmm_bic_L: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    gmm_bic_A: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    gmm_bic_L_k2k3: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=float))  # [bic@2,bic@3]
    gmm_bic_A_k2k3: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=float))
    # DensMAP-inspired mass fractions
    mass_fracs: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    mass_fracs_robust: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    # Coherence (full matrix)
    coherence_full: float = 0.0
    f1_by_component_full: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    # Dominance (separate)
    dominance_L: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    dominance_A: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    # JAIC-h / Jaccard
    jaic_h: float = np.nan
    jaccard_index: float = np.nan


# ---------------------------------------------------------------------
# β-divergence and model log-likelihoods
# ---------------------------------------------------------------------

def beta_from_loss(beta_loss: str) -> float:
    if beta_loss == "frobenius":
        return 2.0
    if beta_loss in {"kullback-leibler", "kullback_leibler", "kl"}:
        return 1.0
    if beta_loss in {"itakura-saito", "itakura_saito", "is"}:
        return 0.0
    raise ValueError(f"Unsupported beta_loss: {beta_loss!r}")


def beta_divergence(X: np.ndarray, Y: np.ndarray, beta: float) -> float:
    """Sum β-divergence D_beta(X|Y) for β ∈ {2,1,0} (Frobenius, KL, IS)."""
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    if beta == 2.0:  # Frobenius
        return 0.5 * np.square(X - Y).sum()
    if beta == 1.0:  # KL
        Xsafe = np.maximum(X, EPS); Ysafe = np.maximum(Y, EPS)
        return (Xsafe * np.log(Xsafe / Ysafe) - Xsafe + Ysafe).sum()
    if beta == 0.0:  # IS
        Xsafe = np.maximum(X, EPS); Ysafe = np.maximum(Y, EPS)
        return (Xsafe / Ysafe - np.log(Xsafe / Ysafe) - 1.0).sum()
    raise ValueError("Only β∈{2,1,0} supported.")


def _log_factorial(x: np.ndarray) -> np.ndarray:
    """Stable log(x!) with optional SciPy; falls back to Stirling."""
    try:
        import scipy.special as sps  # local import avoids joblib pickling issues
        return sps.gammaln(x + 1.0)
    except Exception:
        x = np.maximum(x, 1e-8)
        return x * np.log(x) - x + 0.5 * np.log(2.0 * np.pi * x) + (1.0 / (12.0 * x)) - (1.0 / (360.0 * x**3))


def gaussian_loglike(X: np.ndarray, MU: np.ndarray, mask: Optional[np.ndarray]) -> float:
    """Profiled Gaussian log-likelihood (σ²_hat from observed entries)."""
    if mask is None:
        diff = X - MU; N = diff.size; rss = np.square(diff).sum()
    else:
        diff = (X - MU) * mask; N = int(mask.sum()); rss = np.square(diff).sum()
    sigma2 = max(rss / max(N, 1), EPS)
    return -0.5 * N * (math.log(2.0 * math.pi * sigma2) + 1.0)


def poisson_loglike(X: np.ndarray, MU: np.ndarray, mask: Optional[np.ndarray]) -> float:
    MU = np.maximum(MU, EPS); X = np.maximum(X, 0.0)
    if mask is None:
        return (X * np.log(MU) - MU - _log_factorial(X)).sum()
    else:
        return ((X * np.log(MU) - MU - _log_factorial(X)) * mask).sum()


def gamma_loglike_shape1(X: np.ndarray, MU: np.ndarray, mask: Optional[np.ndarray]) -> float:
    """Gamma(shape=1) (Exponential) log-likelihood on observed entries."""
    MU = np.maximum(MU, EPS); X = np.maximum(X, EPS)
    if mask is None:
        return (-np.log(MU) - (X / MU)).sum()
    else:
        return ((-np.log(MU) - (X / MU)) * mask).sum()


def model_loglike_for_beta_loss(X: np.ndarray, Y: np.ndarray, beta_loss: str, mask: Optional[np.ndarray]) -> float:
    if beta_loss == "frobenius":
        return gaussian_loglike(X, Y, mask)
    if beta_loss in {"kullback-leibler", "kullback_leibler", "kl"}:
        return poisson_loglike(X, Y, mask)
    if beta_loss in {"itakura-saito", "itakura_saito", "is"}:
        return gamma_loglike_shape1(X, Y, mask)
    raise ValueError(f"Unsupported beta_loss: {beta_loss!r}")


def aic_bic(loglike: float, n_params: int, n_obs: int, addl_params: int = 0) -> Tuple[float, float]:
    k = n_params + addl_params
    aic = -2.0 * loglike + 2.0 * k
    bic = -2.0 * loglike + k * math.log(max(n_obs, 1))
    return aic, bic


# ---------------------------------------------------------------------
# Normalization, binarization, dominance, mass, confusion
# ---------------------------------------------------------------------

def normalize_LA_by_p99(L: np.ndarray, A: np.ndarray, p: float = 99.0
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentile scaling of columns of L with compensating row scaling of A so L@A is invariant."""
    L = np.array(L, dtype=float, copy=True); A = np.array(A, dtype=float, copy=True)
    r = L.shape[1]; scales = np.empty(r, dtype=float)
    for k in range(r):
        s = max(float(np.percentile(L[:, k], p)), EPS)
        scales[k] = s
        L[:, k] /= s; A[k, :] *= s
    return L, A, scales


def _fit_gmm_1d(x: np.ndarray, n_components_options=(2, 3), reg_covar: float = 1e-6, random_state: int = 0
               ) -> Tuple[np.ndarray, np.ndarray, int, float, Dict[int, float]]:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    best_model = None; best_bic = np.inf; chosen_k = None; bic_by_k: Dict[int, float] = {}
    for q in n_components_options:
        gm = GaussianMixture(n_components=q, covariance_type='full', reg_covar=reg_covar,
                             random_state=random_state).fit(x)
        bic = gm.bic(x); bic_by_k[q] = float(bic)
        if bic < best_bic:
            best_bic = bic; best_model = gm; chosen_k = q
    labels = best_model.predict(x)
    return labels, best_model.means_.ravel(), int(chosen_k), float(best_bic), bic_by_k


def _fit_beta_mixture_1d(x: np.ndarray, n_components_options=(2, 3), random_state: int = 0,
                         max_iter: int = 200, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, int, float, Dict[int, float]]:
    """
    Lightweight EM for 1D Beta mixture on values in (0,1). Returns:
      labels, component_means (alpha/(alpha+beta)), chosen_k, chosen_BIC, BIC_by_k
    We use a moments-based M-step for (alpha,beta) for speed & robustness.
    """
    rng = np.random.RandomState(random_state)
    v = np.asarray(x, dtype=float).ravel()
    eps = 1e-9
    # Map to (0,1) if needed
    v_min, v_max = float(np.min(v)), float(np.max(v))
    if v_max <= 0:
        v_scaled = np.full_like(v, fill_value=eps)
    else:
        # scale to [0,1] using min-max; keep a copy of ordering for thresholding on original scale
        if v_max - v_min > eps:
            v_scaled = (v - v_min) / (v_max - v_min + eps)
        else:
            v_scaled = np.clip(v / (v_max + eps), eps, 1.0 - eps)
    v_scaled = np.clip(v_scaled, eps, 1.0 - eps)

    def _log_beta_pdf(xx, a, b):
        # log Beta(x|a,b) using gammaln
        return (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + (a - 1.0) * np.log(xx) + (b - 1.0) * np.log(1.0 - xx))

    def _em_for_k(k: int):
        # init via kmeans on v_scaled
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=5, random_state=random_state)
        cl = km.fit_predict(v_scaled.reshape(-1, 1))
        resp = np.eye(k, dtype=float)[cl] + 1e-3
        resp = resp / resp.sum(axis=1, keepdims=True)

        N = v_scaled.size
        ll_prev = -np.inf
        for _ in range(max_iter):
            # M-step: update weights, alpha,beta via weighted moments
            w = resp.sum(axis=0) + eps   # (k,)
            pi = w / (N + eps)
            means = (resp.T @ v_scaled) / w
            var = (resp.T @ ((v_scaled - means[np.newaxis, :]) ** 2)) / w
            # clamp variances away from 0
            var = np.maximum(var, 1e-6)
            nu = (means * (1.0 - means) / var) - 1.0  # alpha+beta-2+? -> here for Beta variance formula
            # numerical safety: if nu <= 0, push to a modest value
            nu = np.maximum(nu, 1.0)
            alpha = np.maximum(means * nu, 1e-3)
            beta = np.maximum((1.0 - means) * nu, 1e-3)

            # E-step: responsibilities
            logpdf = np.vstack([_log_beta_pdf(v_scaled, alpha[j], beta[j]) for j in range(k)]).T  # (N,k)
            logw = np.log(pi + eps)
            logpost = logpdf + logw
            # log-sum-exp
            m = np.max(logpost, axis=1, keepdims=True)
            resp = np.exp(logpost - m)
            resp = resp / (resp.sum(axis=1, keepdims=True) + eps)

            # log-likelihood
            ll = float(np.sum(m + np.log(resp.sum(axis=1, keepdims=True) + eps)))
            if abs(ll - ll_prev) <= tol * (abs(ll_prev) + eps):
                break
            ll_prev = ll

        # Final params
        pi = resp.sum(axis=0) / (N + eps)
        means = (resp.T @ v_scaled) / (resp.sum(axis=0) + eps)
        # derive alpha,beta again for reporting means (not strictly needed)
        var = (resp.T @ ((v_scaled - means[np.newaxis, :]) ** 2)) / (resp.sum(axis=0) + eps)
        var = np.maximum(var, 1e-6)
        nu = np.maximum((means * (1.0 - means) / var) - 1.0, 1.0)
        alpha = np.maximum(means * nu, 1e-3)
        beta = np.maximum((1.0 - means) * nu, 1e-3)

        # Compute LL and BIC
        logpdf = np.vstack([_log_beta_pdf(v_scaled, alpha[j], beta[j]) for j in range(k)]).T
        ll = float(np.sum(np.log(np.sum(pi * np.exp(logpdf), axis=1) + eps)))
        # Parameters: k*(alpha,beta) + (k-1) mixture weights
        p = 2 * k + (k - 1)
        bic = -2.0 * ll + p * math.log(max(N, 1))
        labels = np.argmax(resp, axis=1)
        return labels, means, bic, ll

    best = None; best_bic = np.inf; best_k = None; bic_by_k = {}
    
    # Default fallback - simple median split
    default_med = np.median(v_scaled)
    default_labels = (v_scaled >= default_med).astype(int)
    default_means = np.array([
        np.mean(v_scaled[default_labels==0]) if np.any(default_labels==0) else 0.5,
        np.mean(v_scaled[default_labels==1]) if np.any(default_labels==1) else 0.5
    ])
    
    for q in n_components_options:
        try:
            labels, means, bic, ll = _em_for_k(q)
            bic_by_k[q] = float(bic)
            if bic < best_bic:
                best = (labels, means, bic)
                best_bic = bic; best_k = q
        except Exception:
            # fall back to trivial two groups by median for this k
            bic_by_k[q] = np.inf
            # Still update best if this is our first iteration and best is None
            if best is None:
                best = (default_labels, default_means, np.inf)
                best_k = 2

    # If still None (shouldn't happen with above logic, but be safe)
    if best is None:
        best = (default_labels, default_means, np.inf)
        best_k = 2
        
    labels, means, bic = best
    return labels, means, int(best_k), float(bic), bic_by_k


def _otsu_threshold(values: np.ndarray, nbins: int = 128) -> float:
    """Simple, dependency-free Otsu threshold on 1D data."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    # Guard: constant input
    if np.all(v == v[0]):
        return float(v[0])
    hist, bin_edges = np.histogram(v, bins=nbins)
    hist = hist.astype(float)
    # Bin centers
    mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]
    mu1 = np.cumsum(hist * mids) / np.maximum(w1, 1e-12)
    mu2 = (np.cumsum((hist * mids)[::-1]) / np.maximum(w2[::-1], 1e-12))[::-1]
    # Maximize between-class variance
    sigma_b2 = w1[:-1] * w2[1:] * (mu1[:-1] - mu2[1:]) ** 2
    idx = int(np.argmax(sigma_b2))
    return float(mids[idx])


def _binarize_1d(
    x: np.ndarray,
    method: str,
    random_state: int,
    reg_covar: float,
    threshold: Optional[float]
) -> Tuple[np.ndarray, float, int, float, Dict[int, float]]:
    """
    Binarize a single vector x according to 'method' and return:
    labels (0/1), reported_threshold, chosen_k (for mixture-like methods), chosen_BIC, BICs_by_k
    For kmeans/otsu we return k and BIC-like values as placeholders (k=-1, bic=np.nan).
    """
    rng = np.random.RandomState(random_state)
    x = np.asarray(x, dtype=float).reshape(-1, 1)

    if method in {"beta", "beta-mixture"}:
        # Fit a 2–3 component Beta mixture on min–max scaled values
        vx = x.ravel()
        labels, means, k, bic, bic_by_k = _fit_beta_mixture_1d(
            vx, n_components_options=(2, 3), random_state=random_state
        )
        hi = int(np.argmax(means))
        thr = float(np.min(vx[labels == hi])) if np.any(labels == hi) else float(np.max(vx))
        mask = (labels == hi).astype(np.int8).ravel()
        return mask, thr, int(k), float(bic), bic_by_k


    if method in {"gmm", "log-gmm"}:
        vx = np.log1p(x) if (method == "log-gmm") else x
        labels, means, k, bic, bic_by_k = _fit_gmm_1d(
            vx.ravel(), n_components_options=(2, 3),
            reg_covar=reg_covar, random_state=random_state
        )
        hi = int(np.argmax(means))
        thr = float(np.min(vx[labels == hi])) if np.any(labels == hi) else float(np.max(vx))
        # Report threshold in linear scale if we log-transformed
        if method == "log-gmm":
            thr = float(np.expm1(thr))
        # Build the 0/1 mask from labels against the high-mean cluster
        mask = (labels == hi).astype(np.int8).ravel()
        return mask, thr, int(k), float(bic), bic_by_k

    if method == "kmeans":
        # k=3; "on" = cluster with highest mean
        km = KMeans(n_clusters=3, n_init=10, random_state=random_state)
        lab = km.fit_predict(x)
        means = np.array([x[lab == c].mean() if np.any(lab == c) else -np.inf for c in range(3)], dtype=float)
        hi = int(np.argmax(means))
        thr = float(np.min(x[lab == hi])) if np.any(lab == hi) else float(np.max(x))
        return (lab == hi).astype(np.int8).ravel(), thr, -1, float("nan"), {}

    if method == "otsu":
        thr = _otsu_threshold(x.ravel())
        return (x.ravel() >= thr).astype(np.int8), float(thr), -1, float("nan"), {}

    if method == "percentile":
        if threshold is None:
            raise ValueError("percentile binarization requires --x-threshold (or side-specific).")
        thr = float(np.quantile(x.ravel(), threshold)) if threshold <= 1.0 else float(threshold)
        return (x.ravel() >= thr).astype(np.int8), thr, -1, float("nan"), {}

    raise ValueError(f"Unknown binarization method: {method!r}")


def binarize_LA(
    L_norm: np.ndarray,
    A_norm: np.ndarray,
    method_L: str,
    method_A: str,
    random_state: int,
    reg_covar: float = 1e-6,
    threshold_L: Optional[float] = None,
    threshold_A: Optional[float] = None,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Binarize L_norm (by columns) and A_norm (by rows), allowing different methods per side.
    Returns:
      L_bin, A_bin, thr_L, thr_A, k_L, k_A, bicL_chosen, bicA_chosen, bicL_k2k3, bicA_k2k3
    """
    n_rows, r = L_norm.shape
    r2, n_cols = A_norm.shape
    assert r == r2

    L_bin = np.zeros_like(L_norm, dtype=np.int8)
    A_bin = np.zeros_like(A_norm, dtype=np.int8)

    thr_L = np.zeros(r, dtype=float)
    thr_A = np.zeros(r, dtype=float)

    k_L = np.full(r, -1, dtype=int)
    k_A = np.full(r, -1, dtype=int)

    bicL_chosen = np.full(r, np.nan, dtype=float)
    bicA_chosen = np.full(r, np.nan, dtype=float)

    bicL_k2k3 = np.full((r, 2), np.nan, dtype=float)
    bicA_k2k3 = np.full((r, 2), np.nan, dtype=float)

    for k in range(r):
        # ---- L side (column k)
        lab_x, thrx, kx, bicx, dxb = _binarize_1d(
            L_norm[:, k], method_L, random_state, reg_covar, threshold_L
        )
        L_bin[:, k] = lab_x
        thr_L[k] = float(thrx)
        k_L[k] = int(kx)
        bicL_chosen[k] = float(bicx)
        if dxb:
            bicL_k2k3[k, :] = [dxb.get(2, np.nan), dxb.get(3, np.nan)]

        # ---- A side (row k)
        lab_y, thry, ky, bici, dyi = _binarize_1d(
            A_norm[k, :], method_A, random_state, reg_covar, threshold_A
        )
        A_bin[k, :] = lab_y
        thr_A[k] = float(thry)
        k_A[k] = int(ky)
        bicA_chosen[k] = float(bici)
        if dyi:
            bicA_k2k3[k, :] = [dyi.get(2, np.nan), dyi.get(3, np.nan)]

    return (L_bin, A_bin, thr_L, thr_A, k_L, k_A,
            bicL_chosen, bicA_chosen, bicL_k2k3, bicA_k2k3)


# ---- Dominance & mass robustness ------------------------------------

def _hhi(v: np.ndarray) -> float:
    s = float(np.sum(v))
    if s <= 0:
        return 0.0
    p = np.asarray(v, dtype=float) / s
    return float(np.sum(p * p))


def _gini(v: np.ndarray) -> float:
    x = np.asarray(v, dtype=float)
    s = float(np.sum(x))
    if s <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    # 1 + 1/n - 2 * (sum_i ((n+1-i) x_i)) / (n * sum_i x_i)
    cum = (np.arange(1, n + 1) * x).sum()
    return float(1.0 + 1.0 / n - 2.0 * cum / (n * s))


def dominance_scores_pair(L_norm: np.ndarray, A_norm: np.ndarray, metric: str = "hhi"
) -> Tuple[np.ndarray, np.ndarray]:
    """Return separate dominance scores for L columns and A rows."""
    r = L_norm.shape[1]
    f = _hhi if metric == "hhi" else _gini
    dom_L = np.zeros(r, dtype=float)
    dom_A = np.zeros(r, dtype=float)
    for k in range(r):
        dom_L[k] = f(L_norm[:, k])
        dom_A[k] = f(A_norm[k, :])
    return dom_L, dom_A


def _winsor_sum(v: np.ndarray, q: float) -> float:
    if v.size == 0:
        return 0.0
    thr = float(np.quantile(v, q))
    return float(np.minimum(v, thr).sum())


def component_mass_fractions(L_norm: np.ndarray, A_norm: np.ndarray) -> np.ndarray:
    sL = L_norm.sum(axis=0); sA = A_norm.sum(axis=1)
    masses = np.maximum(sL * sA, 0.0)
    Z = float(np.sum(masses))
    return masses / Z if Z > 0 else np.zeros_like(masses)


def component_mass_fractions_trimmed(L_norm: np.ndarray, A_norm: np.ndarray, q: float = 0.99) -> np.ndarray:
    r = L_norm.shape[1]
    sL = np.array([_winsor_sum(L_norm[:, k], q) for k in range(r)], dtype=float)
    sA = np.array([_winsor_sum(A_norm[k, :], q) for k in range(r)], dtype=float)
    masses = np.maximum(sL * sA, 0.0)
    Z = float(np.sum(masses))
    return masses / Z if Z > 0 else np.zeros_like(masses)


# ---------------------------------------------------------------------
# Confusion, coherence, JAIC-h
# ---------------------------------------------------------------------

def compute_confusion_from_binarized(
    P_true: np.ndarray,
    L_bin: np.ndarray,
    A_bin: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute P_hat, error, and confusion (mask-aware if mask provided).
    IMPORTANT: P_hat is **clipped to {0,1}** so that the confusion counts
    are exactly TP, FP, TN, FN for binary comparison.
    """
    P_hat_counts = (L_bin @ A_bin)
    P_hat = (P_hat_counts > 0).astype(np.int8)  # clip to {0,1}  ### NEW
    P_err = P_true - P_hat
    if mask is not None:
        sel = (mask > 0)
        true_bin = (P_true[sel] > 0).astype(np.int8).ravel()
        pred_bin = (P_hat[sel] > 0).astype(np.int8).ravel()
    else:
        true_bin = (P_true > 0).astype(np.int8).ravel()
        pred_bin = (P_hat > 0).astype(np.int8).ravel()
    cm = confusion_matrix(true_bin, pred_bin, labels=[1, 0])
    return P_hat, P_err, cm


def component_coherence_f1(
    X: np.ndarray,
    L_bin: np.ndarray,
    A_bin: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Per-component F1 on the outer-product support of each component (optionally masked)."""
    r = L_bin.shape[1]
    f1s = np.zeros(r, dtype=float)
    for k in range(r):
        pred_k = np.outer(L_bin[:, k] > 0, A_bin[k, :] > 0)
        if mask is not None:
            sel = (mask > 0)
            y_true = (X[sel] > 0).ravel(); y_pred = (pred_k[sel] > 0).ravel()
        else:
            y_true = (X > 0).ravel(); y_pred = (pred_k > 0).ravel()
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1s[k] = (2.0 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return f1s


def jaic_h_from_confusion(cm: np.ndarray, rank: int, m: int, n: int) -> Tuple[float, float]:
    """Return (JAIC-h, Jaccard index) computed from a 2×2 confusion matrix."""
    TP = float(cm[0, 0]); FN = float(cm[0, 1])
    FP = float(cm[1, 0]); TN = float(cm[1, 1])
    denom = TP + FP + FN
    jaccard = (TP / denom) if denom > 0 else 0.0
    E = 1.0 - jaccard
    N = TP + TN + FP + FN
    k = 2.0 * rank * (m + n)  # parameters in L and A
    jaic_h = 2.0 * k + 2.0 * E * N
    return float(jaic_h), float(jaccard)


# ---------------------------------------------------------------------
# NMF fitting utilities (single run) + CV helpers
#   (CV utilities kept, but reconstruction metrics use FULL matrix.)
# ---------------------------------------------------------------------

def _build_nmf_estimator(
    rank: int,
    seed: int,
    beta_loss: str,
    init: str,
    max_iter: int,
    tol: float,
    alpha_W: float,
    alpha_H: float,
):
    """Instantiate sklearn NMF with our conventions."""
    if beta_loss == "frobenius":
        nmf = NMF(
            n_components=rank, init=init, random_state=seed, solver="cd",
            beta_loss=beta_loss, max_iter=max_iter, tol=tol,
            l1_ratio=0.0, alpha_W=alpha_W, alpha_H=alpha_H
        )
    else:
        nmf = NMF(
            n_components=rank, init=init, random_state=seed, solver="mu",
            beta_loss=beta_loss, max_iter=max_iter, tol=tol,
            alpha_W=alpha_W, alpha_H=alpha_H
        )
    return nmf


def fit_nmf_once(
    X: np.ndarray,
    rank: int,
    seed: int,
    beta_loss: str,
    init: str,
    max_iter: int,
    tol: float,
    alpha_W: float,
    alpha_H: float,
    threads_per_job: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run one NMF fit and return (L, A)."""
    with threadpool_limits(threads_per_job):
        nmf = _build_nmf_estimator(rank, seed, beta_loss, init, max_iter, tol, alpha_W, alpha_H)
        W = nmf.fit_transform(X)
        H = nmf.components_
    return W, H


def n_params_for_aic(r: int, m: int, n: int, dof_mode: str) -> int:
    base = r * (m + n)
    if dof_mode == "minus_r":
        return base - r
    if dof_mode == "minus_r2":
        return base - (r * r)
    raise ValueError(f"Unknown aic-dof mode: {dof_mode}")


def evaluate_single_config(
    X: np.ndarray,
    data_index: Iterable,
    data_columns: Iterable,
    rank: int,
    seed: int,
    beta_loss: str,
    init: str,
    max_iter: int,
    tol: float,
    alpha_W: float,
    alpha_H: float,
    aic_dof: str,
    bin_method_L: str,
    bin_method_A: str,
    x_threshold_L: Optional[float],
    x_threshold_A: Optional[float],
    gmm_reg_covar: float,
    holdout_mask: Optional[np.ndarray],
    threads_per_job: int,
    mass_trim_q: float,
    dominance_metric: str,
) -> FitResult:


    n_rows, n_cols = X.shape
    n_obs_full = n_rows * n_cols
    beta = beta_from_loss(beta_loss)

    # Fit
    L, A = fit_nmf_once(
        X, rank, seed, beta_loss=beta_loss, init=init,
        max_iter=max_iter, tol=tol, alpha_W=alpha_W, alpha_H=alpha_H,
        threads_per_job=threads_per_job
    )

    # Normalize, cache reconstruction
    L_norm, A_norm, scales = normalize_LA_by_p99(L, A, p=99.0)
    Y = L_norm @ A_norm

    # Losses
    bdiv = beta_divergence(X, Y, beta)
    beta_mean = bdiv / n_obs_full
    rss = np.square(X - Y).sum()

    # Likelihoods & ICs (allow held-out for likelihoods only)
    if holdout_mask is None:
        mask_train = None; mask_test = None
        n_obs_train = n_obs_full; n_obs_test = None
    else:
        mask_train = (1.0 - holdout_mask); mask_test = holdout_mask
        n_obs_train = int(mask_train.sum()); n_obs_test = int(mask_test.sum())

    ll_train = model_loglike_for_beta_loss(X, Y, beta_loss=beta_loss, mask=mask_train)
    addl = 1 if beta_loss == "frobenius" else 0
    k_params = n_params_for_aic(rank, n_rows, n_cols, aic_dof)
    aic_train, bic_train = aic_bic(ll_train, k_params, n_obs_train, addl_params=addl)
    if mask_test is not None:
        ll_test = model_loglike_for_beta_loss(X, Y, beta_loss=beta_loss, mask=mask_test)
        aic_test, bic_test = aic_bic(ll_test, k_params, n_obs_test, addl_params=addl)
    else:
        ll_test = None; aic_test = None; bic_test = None

    # Binarize
    L_bin, A_bin, thr_L, thr_A, k_L, k_A, bicL_sel, bicA_sel, bicL_k2k3, bicA_k2k3 = binarize_LA(
        L_norm, A_norm,
        method_L=bin_method_L, method_A=bin_method_A,
        random_state=seed, reg_covar=gmm_reg_covar,
        threshold_L=x_threshold_L, threshold_A=x_threshold_A
    )
    
    # Confusion & coherence — FULL MATRIX ONLY for reconstruction metrics
    _, _, cm_full = compute_confusion_from_binarized(X, L_bin, A_bin, mask=None)  # force full
    f1_full = component_coherence_f1(X, L_bin, A_bin, mask=None)
    coh_full = float(np.mean(f1_full)) if f1_full.size else 0.0

    # Mass & dominance
    mass_fracs = component_mass_fractions(L_norm, A_norm)
    mass_fracs_robust = component_mass_fractions_trimmed(L_norm, A_norm, q=mass_trim_q)
    dom_L, dom_A = dominance_scores_pair(L_norm, A_norm, metric=dominance_metric)

    # JAIC-h (from full confusion)
    jaic_h, jacc = jaic_h_from_confusion(cm_full, rank=rank, m=n_rows, n=n_cols)

    fr = FitResult(
        rank=rank, seed=seed,
        L=L, A=A, L_norm=L_norm, A_norm=A_norm, p99_scales=scales,
        rss=rss, beta_loss_mean=beta_mean,
        loglike_train=float(ll_train), loglike_test=None if ll_test is None else float(ll_test),
        aic_train=float(aic_train), bic_train=float(bic_train),
        aic_test=None if aic_test is None else float(aic_test),
        bic_test=None if bic_test is None else float(bic_test),
        n_obs_train=int(n_obs_train), n_obs_test=None if n_obs_test is None else int(n_obs_test),
        thresholds_L=thr_L, thresholds_A=thr_A, L_bin=L_bin, A_bin=A_bin, conf=cm_full,
        gmm_k_L=k_L, gmm_k_A=k_A, gmm_bic_L=bicL_sel, gmm_bic_A=bicA_sel,
        gmm_bic_L_k2k3=bicL_k2k3, gmm_bic_A_k2k3=bicA_k2k3,
        mass_fracs=mass_fracs, mass_fracs_robust=mass_fracs_robust,
        coherence_full=coh_full, f1_by_component_full=f1_full,
        dominance_L=dom_L, dominance_A=dom_A,
        jaic_h=jaic_h, jaccard_index=jacc,
    )
    return fr


# ---------------------------- Cross-validation (unchanged) ------------------------

def _cv_on_rows(
    X: np.ndarray,
    ranks: List[int],
    seeds: List[int],
    kfold: int,
    beta_loss: str,
    init: str,
    max_iter: int,
    tol: float,
    alpha_W: float,
    alpha_H: float,
    threads_per_job: int,
) -> Dict[int, Tuple[float, int]]:
    """
    K-fold CV where rows are split (e.g., genes).
    Returns dict rank -> (sum_loglik_test, total_entries_test).
    """
    m, n = X.shape
    if kfold <= 1 or kfold > m:
        return {r: (0.0, 0) for r in ranks}

    kf = KFold(n_splits=kfold, shuffle=True, random_state=0)
    out = {r: (0.0, 0) for r in ranks}

    for r in ranks:
        sum_ll = 0.0; total = 0
        for train_idx, test_idx in kf.split(np.arange(m)):
            X_tr = X[train_idx, :]
            X_te = X[test_idx, :]

            # Choose best seed on training by (highest) train log-likelihood
            best_ll_tr = -np.inf; best_model = None
            with threadpool_limits(threads_per_job):
                for seed in seeds:
                    nmf = _build_nmf_estimator(r, seed, beta_loss, init, max_iter, tol, alpha_W, alpha_H)
                    W_tr = nmf.fit_transform(X_tr)
                    H = nmf.components_
                    Y_tr = W_tr @ H
                    ll_tr = model_loglike_for_beta_loss(X_tr, Y_tr, beta_loss, mask=None)
                    if ll_tr > best_ll_tr:
                        best_ll_tr = ll_tr; best_model = nmf

            # Evaluate on test rows via transform
            with threadpool_limits(threads_per_job):
                W_te = best_model.transform(X_te)
                Y_te = W_te @ best_model.components_
            ll_te = model_loglike_for_beta_loss(X_te, Y_te, beta_loss, mask=None)
            sum_ll += float(ll_te)
            total += X_te.size

        out[r] = (sum_ll, total)
    return out


def _cv_on_cols_via_transpose(
    X: np.ndarray,
    ranks: List[int],
    seeds: List[int],
    kfold: int,
    beta_loss: str,
    init: str,
    max_iter: int,
    tol: float,
    alpha_W: float,
    alpha_H: float,
    threads_per_job: int,
) -> Dict[int, Tuple[float, int]]:
    """
    K-fold CV for columns (e.g., strains) by operating on X.T:
    split rows of X.T (strains), fit on training strains, predict held-out strains.
    Returns dict rank -> (sum_loglik_test, total_entries_test).
    """
    XT = X.T
    return _cv_on_rows(
        XT, ranks, seeds, kfold, beta_loss, init, max_iter, tol, alpha_W, alpha_H, threads_per_job
    )


# ---------------------------------------------------------------------
# Dimensionality trees (computed on L)
# ---------------------------------------------------------------------

def jaccard_sets(a: np.ndarray, b: np.ndarray) -> float:
    a_idx = set(np.where(a > 0)[0]); b_idx = set(np.where(b > 0)[0])
    if not a_idx and not b_idx:
        return 1.0
    if not a_idx or not b_idx:
        return 0.0
    inter = len(a_idx & b_idx); union = len(a_idx | b_idx)
    return inter / max(union, 1)


def build_optica_tree(results_by_rank: Dict[int, FitResult], similarity_threshold: float = 0.3) -> Dict[str, object]:
    ranks = sorted(results_by_rank.keys())
    edges: List[Tuple[int,int,int,int,float]] = []
    splits: Dict[int,int] = {}; merges: Dict[int,int] = {}
    for r_from, r_to in zip(ranks[:-1], ranks[1:]):
        L_from = results_by_rank[r_from].L_bin
        L_to   = results_by_rank[r_to].L_bin
        J = np.zeros((L_from.shape[1], L_to.shape[1]), dtype=float)
        for i in range(L_from.shape[1]):
            for j in range(L_to.shape[1]):
                J[i, j] = jaccard_sets(L_from[:, i], L_to[:, j])
        parents = np.argmax(J, axis=0)
        for j in range(L_to.shape[1]):
            i = int(parents[j]); s = float(J[i, j])
            if s >= similarity_threshold:
                edges.append((r_from, i, r_to, j, s))
        parent_counts: Dict[int,int] = {}; child_counts: Dict[int,int] = {}
        for (rf, i, rt, j, s) in edges:
            if rf == r_from and rt == r_to:
                parent_counts[i] = parent_counts.get(i, 0) + 1
                child_counts[j] = child_counts.get(j, 0) + 1
        splits[r_to] = int(sum(1 for c in parent_counts.values() if c >= 2))
        merges[r_to] = int(sum(1 for c in child_counts.values() if c >= 2))
    return {"edges": edges, "splits": splits, "merges": merges}


def build_optica_tree_cosine(results_by_rank: Dict[int, FitResult], corr_threshold: float = 0.3) -> Dict[str, object]:
    ranks = sorted(results_by_rank.keys())
    edges: List[Tuple[int,int,int,int,float]] = []
    splits: Dict[int,int] = {}; merges: Dict[int,int] = {}

    for r_from, r_to in zip(ranks[:-1], ranks[1:]):
        Lf = results_by_rank[r_from].L_norm
        Lt = results_by_rank[r_to].L_norm
        Af = Lf / (np.linalg.norm(Lf, axis=0, keepdims=True) + 1e-12)
        At = Lt / (np.linalg.norm(Lt, axis=0, keepdims=True) + 1e-12)
        S = Af.T @ At
        parents = np.argmax(S, axis=0)
        for j in range(At.shape[1]):
            i = int(parents[j]); s = float(S[i, j])
            if s >= corr_threshold:
                edges.append((r_from, i, r_to, j, s))
        parent_counts: Dict[int,int] = {}; child_counts: Dict[int,int] = {}
        for (rf, i, rt, j, s) in edges:
            if rf == r_from and rt == r_to:
                parent_counts[i] = parent_counts.get(i, 0) + 1
                child_counts[j] = child_counts.get(j, 0) + 1
        splits[r_to] = int(sum(1 for c in parent_counts.values() if c >= 2))
        merges[r_to] = int(sum(1 for c in child_counts.values() if c >= 2))
    return {"edges": edges, "splits": splits, "merges": merges}


def compute_forward_conservation(results_by_rank: Dict[int, FitResult], 
                                similarity_threshold: float = 0.3,
                                method: str = "jaccard") -> Dict[int, float]:
    """
    Compute forward-looking conservation: what fraction of components at each rank
    existed at the previous rank (are "conserved" from lower dimensions).
    
    This matches OptICA's forward conservation concept where we track whether 
    components discovered at lower dimensions persist at higher dimensions.
    
    Returns:
        Dictionary mapping rank -> conservation fraction
        First rank has conservation = 0.0 (all components are "new")
    """
    ranks = sorted(results_by_rank.keys())
    conservation = {}
    
    # First rank: all components are new (0% conserved)
    conservation[ranks[0]] = 0.0
    
    # For each subsequent rank, calculate what fraction existed before
    for idx in range(1, len(ranks)):
        r_prev = ranks[idx - 1]
        r_curr = ranks[idx]
        
        if method == "jaccard":
            # Use binarized L columns
            L_prev = results_by_rank[r_prev].L_bin
            L_curr = results_by_rank[r_curr].L_bin
            
            # Compute similarity matrix
            sim_matrix = np.zeros((L_prev.shape[1], L_curr.shape[1]), dtype=float)
            for i in range(L_prev.shape[1]):
                for j in range(L_curr.shape[1]):
                    sim_matrix[i, j] = jaccard_sets(L_prev[:, i], L_curr[:, j])
        
        elif method == "cosine":
            # Use normalized continuous L
            L_prev = results_by_rank[r_prev].L_norm
            L_curr = results_by_rank[r_curr].L_norm
            
            # Normalize columns to unit vectors
            L_prev_norm = L_prev / (np.linalg.norm(L_prev, axis=0, keepdims=True) + 1e-12)
            L_curr_norm = L_curr / (np.linalg.norm(L_curr, axis=0, keepdims=True) + 1e-12)
            
            # Compute cosine similarity
            sim_matrix = L_prev_norm.T @ L_curr_norm
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # For each component in current rank, find best match in previous rank
        best_matches = np.max(sim_matrix, axis=0)
        
        # Count how many components are conserved (have parent above threshold)
        n_conserved = np.sum(best_matches >= similarity_threshold)
        n_total = L_curr.shape[1] if method == "jaccard" else L_curr.shape[1]
        
        conservation[r_curr] = float(n_conserved) / float(n_total)
    
    return conservation


def compute_cumulative_conservation(results_by_rank: Dict[int, FitResult],
                                   similarity_threshold: float = 0.3,
                                   method: str = "jaccard") -> Dict[int, float]:
    """
    Compute cumulative forward conservation: what fraction of components at each rank
    can be traced back to ANY previous rank (not just the immediately previous one).
    
    This gives a measure of truly "new" components appearing at each dimension.
    """
    ranks = sorted(results_by_rank.keys())
    cumulative_conservation = {}
    
    # First rank: all components are new
    cumulative_conservation[ranks[0]] = 0.0
    
    for curr_idx in range(1, len(ranks)):
        r_curr = ranks[curr_idx]
        
        if method == "jaccard":
            L_curr = results_by_rank[r_curr].L_bin
            n_curr = L_curr.shape[1]
            
            # Check against ALL previous ranks
            max_similarities = np.zeros(n_curr)
            for prev_idx in range(curr_idx):
                r_prev = ranks[prev_idx]
                L_prev = results_by_rank[r_prev].L_bin
                
                for j in range(n_curr):
                    for i in range(L_prev.shape[1]):
                        sim = jaccard_sets(L_prev[:, i], L_curr[:, j])
                        max_similarities[j] = max(max_similarities[j], sim)
        
        elif method == "cosine":
            L_curr = results_by_rank[r_curr].L_norm
            L_curr_norm = L_curr / (np.linalg.norm(L_curr, axis=0, keepdims=True) + 1e-12)
            n_curr = L_curr.shape[1]
            
            max_similarities = np.zeros(n_curr)
            for prev_idx in range(curr_idx):
                r_prev = ranks[prev_idx]
                L_prev = results_by_rank[r_prev].L_norm
                L_prev_norm = L_prev / (np.linalg.norm(L_prev, axis=0, keepdims=True) + 1e-12)
                
                sim_matrix = L_prev_norm.T @ L_curr_norm
                max_similarities = np.maximum(max_similarities, np.max(sim_matrix, axis=0))
        
        n_conserved = np.sum(max_similarities >= similarity_threshold)
        cumulative_conservation[r_curr] = float(n_conserved) / float(n_curr)
    
    return cumulative_conservation


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def save_df(df: pd.DataFrame, path: str, fmt: str) -> None:
    if fmt == "csv":
        df.to_csv(path)
    elif fmt == "parquet":
        df.to_parquet(path, index=True)
    else:
        raise ValueError(f"Unknown output format: {fmt}")


def run_grid(
    data: pd.DataFrame,
    ranks: List[int],
    seeds: List[int],
    beta_loss: str,
    init: str,
    outdir: str,
    prefix: str,
    n_jobs: int,
    max_iter: int,
    tol: float,
    alpha_W: float,
    alpha_H: float,
    aic_dof: str,
    bin_method_L: str,
    bin_method_A: str,
    x_threshold_L: Optional[float],
    x_threshold_A: Optional[float],
    gmm_reg_covar: float,
    holdout_mask: Optional[np.ndarray],
    threads_per_job: int,
    out_format: str,
    mass_trim_q: float,
    dominance_metric: str,
    corr_threshold: float,
    make_sankey: bool,
    sankey_min_weight: float,
) -> Dict[int, FitResult]:

    os.makedirs(outdir, exist_ok=True)
    X = data.values.astype(float)

    # Evaluate all (rank, seed) in parallel
    jobs = [(r, s) for r in ranks for s in seeds]
    results_all: List[FitResult] = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(evaluate_single_config)(
            X, data.index, data.columns, r, s, beta_loss, init, max_iter, tol,
            alpha_W, alpha_H, aic_dof,
            bin_method_L, bin_method_A,
            x_threshold_L, x_threshold_A,
            gmm_reg_covar,
            holdout_mask, threads_per_job, mass_trim_q, dominance_metric
        )
        for (r, s) in jobs
    )


    # Keep best seed per rank by BIC (prefer held-out BIC if available; tie-break with coherence_full)
    by_rank: Dict[int, FitResult] = {}
    for r in ranks:
        candidates = [fr for fr in results_all if fr.rank == r]
        def rank_key(fr: FitResult):
            bic = fr.bic_test if fr.bic_test is not None else fr.bic_train
            return (bic, -fr.coherence_full)
        best = min(candidates, key=rank_key)
        by_rank[r] = best

        # Save matrices/diagnostics for the best seed
        if out_format == "csv":
            save_df(pd.DataFrame(best.L_norm, index=data.index, columns=[f"k{k}" for k in range(r)]),
                    os.path.join(outdir, f"{prefix}_L_norm_rank{r}_seed{best.seed}.csv"), "csv")
            save_df(pd.DataFrame(best.A_norm, index=[f"k{k}" for k in range(r)], columns=data.columns),
                    os.path.join(outdir, f"{prefix}_A_norm_rank{r}_seed{best.seed}.csv"), "csv")
            save_df(pd.DataFrame(best.L_bin.astype(int), index=data.index, columns=[f"k{k}" for k in range(r)]),
                    os.path.join(outdir, f"{prefix}_L_bin_rank{r}_seed{best.seed}.csv"), "csv")
            save_df(pd.DataFrame(best.A_bin.astype(int), index=[f"k{k}" for k in range(r)], columns=data.columns),
                    os.path.join(outdir, f"{prefix}_A_bin_rank{r}_seed{best.seed}.csv"), "csv")

        # Rank summary JSON
        perp_train = math.exp(-best.loglike_train / max(1, best.n_obs_train))
        perp_test = (None if best.loglike_test is None or best.n_obs_test is None
                     else math.exp(-best.loglike_test / max(1, best.n_obs_test)))
        summary = {
            "rank": r, "seed": best.seed,
            "loglike_train": best.loglike_train, "loglike_test": best.loglike_test,
            "perplexity_train": perp_train, "perplexity_test": perp_test,
            "aic_train": best.aic_train, "bic_train": best.bic_train,
            "aic_test": best.aic_test,   "bic_test": best.bic_test,
            "n_obs_train": int(best.n_obs_train), "n_obs_test": (None if best.n_obs_test is None else int(best.n_obs_test)),
            "coherence_full": best.coherence_full,
            "jaic_h": best.jaic_h, "jaccard_index": best.jaccard_index,
            "aic_dof": aic_dof,
            "x_binarize_L": bin_method_L, "x_binarize_A": bin_method_A,
            "x_threshold_L": x_threshold_L, "x_threshold_A": x_threshold_A,
            "gmm_reg_covar": gmm_reg_covar,
        }
        with open(os.path.join(outdir, f"{prefix}_summary_rank{r}.json"), "w") as fh:
            json.dump(summary, fh, indent=2)

    # Build and save OptICA-like tree artifacts (now based on **L**)
    tree_j = build_optica_tree(by_rank, similarity_threshold=0.3)
    pd.DataFrame(tree_j["edges"], columns=["rank_from", "k_from", "rank_to", "k_to", "jaccard"])\
      .to_csv(os.path.join(outdir, f"{prefix}_optica_tree_edges.csv"), index=False)
    with open(os.path.join(outdir, f"{prefix}_optica_tree_summary.json"), "w") as fh:
        json.dump({"splits": tree_j["splits"], "merges": tree_j["merges"]}, fh, indent=2)

    tree_c = build_optica_tree_cosine(by_rank, corr_threshold=corr_threshold)
    pd.DataFrame(tree_c["edges"], columns=["rank_from", "k_from", "rank_to", "k_to", "cosine"])\
      .to_csv(os.path.join(outdir, f"{prefix}_optica_tree_cosine_edges.csv"), index=False)
    with open(os.path.join(outdir, f"{prefix}_optica_tree_cosine_summary.json"), "w") as fh:
        json.dump({"splits": tree_c["splits"], "merges": tree_c["merges"]}, fh, indent=2)

    # Export node-level metrics to annotate trees
    node_metrics_path = os.path.join(outdir, f"{prefix}_node_metrics.csv")
    node_metrics_df = export_node_metrics(by_rank, node_metrics_path)

    # Optional interactive Sankey diagrams (Plotly)
    if make_sankey:
        jaccard_html = os.path.join(outdir, f"{prefix}_sankey_jaccard.html")
        cosine_html  = os.path.join(outdir, f"{prefix}_sankey_cosine.html")
        _build_sankey(tree_j["edges"], node_metrics_df, jaccard_html,
                      min_weight=sankey_min_weight,
                      title=f"{prefix} — OptICA tree (Jaccard≥{0.3:g}, on L)")
        _build_sankey(tree_c["edges"], node_metrics_df, cosine_html,
                      min_weight=sankey_min_weight,
                      title=f"{prefix} — Correlation tree (Cosine≥{corr_threshold:g}, on L)")
    return by_rank


# ---------------------------------------------------------------------
# Plotting (panels)
# ---------------------------------------------------------------------

def minmax01(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    lo, hi = float(np.min(y)), float(np.max(y))
    if hi <= lo + 1e-15:
        return np.zeros_like(y)
    return (y - lo) / (hi - lo)


def _kneedle_decreasing(x: np.ndarray, y: np.ndarray) -> int:
    """
    Simple Kneedle-style knee for decreasing curves:
    1) normalize x,y to [0,1]; 2) reflect y -> 1-y; 3) pick argmax of (y_reflected - x).
    Returns index into the input arrays.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)
    y_ref = 1.0 - y_n
    diff = y_ref - x_n
    return int(np.argmax(diff))



def plot_panels(
    data: pd.DataFrame,
    results: Dict[int, FitResult],
    beta_loss: str,
    outdir: str,
    prefix: str,
    title: str,
    dens_vanish_mult: float = 0.5,
    used_holdout: bool = False,
    dominance_threshold: float = 0.3,
    corr_threshold: float = 0.3,
    make_sankey: bool = False,
    sankey_min_weight: float = 0.0,
    cv_summary: Optional[Dict[int, Dict[str, float]]] = None,
    recon_early_eps: float = 0.05,
):

    ranks = sorted(results.keys())

    # Fit metrics
    aic_tr = np.array([results[r].aic_train for r in ranks])
    bic_tr = np.array([results[r].bic_train for r in ranks])
    aic_te = np.array([np.inf if results[r].aic_test is None else results[r].aic_test for r in ranks])
    bic_te = np.array([np.inf if results[r].bic_test is None else results[r].bic_test for r in ranks])
    rss = np.array([results[r].rss for r in ranks])
    jaic_h = np.array([results[r].jaic_h for r in ranks])

    # Perplexity with correct denominators
    perplexity = []
    for r in ranks:
        fr = results[r]
        if fr.loglike_test is not None and fr.n_obs_test:
            perplexity.append(math.exp(-fr.loglike_test / max(1, fr.n_obs_test)))
        else:
            # fall back to train
            perplexity.append(math.exp(-fr.loglike_train / max(1, fr.n_obs_train)))
    perplexity = np.asarray(perplexity, dtype=float)

    # Normalize fit metrics to [0,1] (lower is better)
    aic_base = (aic_te if used_holdout else aic_tr)
    bic_base = (bic_te if used_holdout else bic_tr)
    aic_n = minmax01(aic_base)
    bic_n = minmax01(bic_base)
    rss_n = minmax01(rss)
    jaic_n = minmax01(jaic_h)

    fig = plt.figure(figsize=(12, 16), constrained_layout=True)
    gs = fig.add_gridspec(4, 2)

    # 1) Fit vs rank
    ax1 = fig.add_subplot(gs[0, 0])
    model_label = {"frobenius": "Gaussian", "kullback-leibler": "Poisson", "itakura-saito": "Gamma"}[beta_loss]
    src = "(held-out)" if used_holdout else "(train)"
    ax1.plot(ranks, aic_n, "o-", label=f"AIC {src} ({model_label})")
    ax1.plot(ranks, bic_n, "o-", label=f"BIC {src} ({model_label})")
    ax1.plot(ranks, rss_n, "o-", label="RSS")
    ax1.plot(ranks, jaic_n, "o-", label="JAIC-h (Jaccard heuristic)")
    # Kneedle elbows (AIC/BIC/RSS)
    try:
        i_knee_aic = _kneedle_decreasing(np.asarray(ranks), aic_n)
        i_knee_bic = _kneedle_decreasing(np.asarray(ranks), bic_n)
        i_knee_rss = _kneedle_decreasing(np.asarray(ranks), rss_n)
        ax1.axvline(ranks[i_knee_aic], color="C0", linestyle=":", alpha=0.6, label=f"AIC knee @ r={ranks[i_knee_aic]}")
        ax1.axvline(ranks[i_knee_bic], color="C1", linestyle=":", alpha=0.6, label=f"BIC knee @ r={ranks[i_knee_bic]}")
        ax1.axvline(ranks[i_knee_rss], color="C2", linestyle=":", alpha=0.6, label=f"RSS knee @ r={ranks[i_knee_rss]}")
    except Exception:
        pass
    # BIC minimum (if not at endpoints)
    try:
        imin_bic = int(np.argmin(bic_base))
        if 0 < imin_bic < len(ranks) - 1:
            ax1.axvline(ranks[imin_bic], color="C1", linestyle="--", alpha=0.7, label=f"min BIC @ r={ranks[imin_bic]}")
    except Exception:
        pass
    # JAIC-h minimum
    try:
        jmin = int(np.argmin(jaic_h))
        ax1.axvline(ranks[jmin], color="C3", linestyle="--", alpha=0.7, label=f"min JAIC-h @ r={ranks[jmin]}")
    except Exception:
        pass
    ax1.set_title("Fit vs. rank (normalized)")
    ax1.set_xlabel("rank"); ax1.set_ylabel("normalized score"); ax1.legend(loc="best")

    # 2) Perplexity
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ranks, perplexity, "o-", label="Perplexity (likelihood-based)")
    if cv_summary is not None:
        ax2.plot(ranks, [cv_summary[r]["perplexity_rows"] for r in ranks], "o-", label="CV rows (genes)")
        ax2.plot(ranks, [cv_summary[r]["perplexity_cols"] for r in ranks], "o-", label="CV cols (strains)")
        ax2.plot(ranks, [cv_summary[r]["perplexity_geom_weighted"] for r in ranks], "o-", label="CV geo (weighted)")
    try:
        i_knee_ppx = _kneedle_decreasing(np.asarray(ranks), minmax01(perplexity))
        ax2.axvline(ranks[i_knee_ppx], color="C0", linestyle=":", alpha=0.6, label=f"Perplexity knee @ r={ranks[i_knee_ppx]}")
    except Exception:
        pass
    ax2.set_title("Model perplexity")
    ax2.set_xlabel("rank"); ax2.set_ylabel("perplexity (lower is better)"); ax2.legend(loc="best")

    # 3) Component mass profile: robust only (winsorized)
    q_levels = [0.10, 0.25, 0.50, 0.75, 0.90]
    mass_quant_robust = {q: [] for q in q_levels}
    vanish_frac = []
    domL_frac = []
    domA_frac = []
    for r in ranks:
        mfr = results[r].mass_fracs_robust
        scaled_r = mfr * r
        for q in q_levels:
            mass_quant_robust[q].append(float(np.quantile(scaled_r, q)))
        thr = dens_vanish_mult * 1.0
        vanish_frac.append(float(np.mean(scaled_r < thr)))
        domL_frac.append(float(np.mean(results[r].dominance_L >= dominance_threshold)))
        domA_frac.append(float(np.mean(results[r].dominance_A >= dominance_threshold)))
    # Mass inequality (Gini on robust scaled masses)
    mass_inequal = [ _gini(results[r].mass_fracs_robust * r) for r in ranks ]

    ax3 = fig.add_subplot(gs[1, 0])
    for q in q_levels:
        ax3.plot(ranks, mass_quant_robust[q], "o-", label=f"robust q={q:.2f}")
    ax3.axhline(1.0, linestyle="--", linewidth=1, alpha=0.5)
    ax3.set_title("Component mass profile (r × mass; uniform ≈ 1)")
    ax3.set_xlabel("rank"); ax3.set_ylabel("scaled mass quantiles")
    # RHS inequality axis
    ax3_rhs = ax3.twinx()
    ax3_rhs.plot(ranks, mass_inequal, "-", color="C5", alpha=0.7, label="mass inequality (Gini)")
    ax3_rhs.set_ylim(0, 1); ax3_rhs.set_ylabel("inequality (Gini)")
    # Merge legends
    l_l, lab_l = ax3.get_legend_handles_labels()
    l_r, lab_r = ax3_rhs.get_legend_handles_labels()
    ax3.legend(l_l + l_r, lab_l + lab_r, loc="best", ncol=2)

    # 3b) Vanishing vs dominant components
    ax3b = fig.add_subplot(gs[1, 1])
    ax3b.plot(ranks, vanish_frac, "o-", label=f"vanish (<{dens_vanish_mult}×uniform)")
    ax3b.plot(ranks, domL_frac, "o-", label=f"dominant L (≥{dominance_threshold:g})")
    ax3b.plot(ranks, domA_frac, "o-", label=f"dominant A (≥{dominance_threshold:g})")
    try:
        i_knee_vanish = _kneedle_decreasing(np.asarray(ranks), 1.0 - np.asarray(vanish_frac, dtype=float))
        ax3b.axvline(ranks[i_knee_vanish], color="C0", linestyle=":", alpha=0.6, label=f"vanish knee @ r={ranks[i_knee_vanish]}")
    except Exception:
        pass
    ax3b.set_title("Vanishing vs dominant components")
    ax3b.set_xlabel("rank"); ax3b.set_ylabel("fraction"); ax3b.set_ylim(0, 1); ax3b.legend(loc="best")

    # 4) Reconstruction metrics (full matrix)
    accs, precs, recalls, f1s, mccs, cohs = [], [], [], [], [], []
    for r in ranks:
        cm = results[r].conf  # full confusion
        # Convert to float first to avoid int overflow
        TP = float(cm[0, 0]); FN = float(cm[0, 1])
        FP = float(cm[1, 0]); TN = float(cm[1, 1])
        total = TP + TN + FP + FN
        acc = (TP + TN) / total if total else 0.0
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec  = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        # MCC (phi coefficient) calcs
        mcc_numer = (TP * TN) - (FP * FN)
        mcc_denom_sq = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
        mcc_denom = np.sqrt(mcc_denom_sq) if mcc_denom_sq >= 0 else 0.0
        mcc = (mcc_numer / mcc_denom) if (mcc_denom != 0) else 0.0
        accs.append(acc); precs.append(prec); recalls.append(rec); f1s.append(f1); mccs.append(mcc)
        cohs.append(results[r].coherence_full)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(ranks, accs, "o-", label="Accuracy")
    ax4.plot(ranks, cohs, "o-", label="Coherence (full)")
    ax4.plot(ranks, precs, "o-", label="Precision")
    ax4.plot(ranks, recalls, "o-", label="Recall")
    ax4.plot(ranks, f1s, "o-", label="F1 Score")
    ax4.plot(ranks, mccs, "o-", label="MCC")
    # earliest r where each metric is within (1−ε) of its max
    eps = float(recon_early_eps)
    def _first_r_at_thresh(y):
        y = np.asarray(y, dtype=float)
        if y.size == 0 or not np.isfinite(y).any():
            return None, None
        ymax = float(np.nanmax(y))
        target = (1.0 - eps) * ymax
        for i, v in enumerate(y):
            if v >= target:
                return i, float(v)
        return None, None
    idx_acc, val_acc = _first_r_at_thresh(accs)
    idx_pre, val_pre = _first_r_at_thresh(precs)
    idx_rec, val_rec = _first_r_at_thresh(recalls)
    idx_f1,  val_f1  = _first_r_at_thresh(f1s)
    idx_mcc, val_mcc = _first_r_at_thresh(mccs)
    if idx_acc is not None: ax4.plot(ranks[idx_acc], val_acc, marker="^", color="C0")
    if idx_pre is not None: ax4.plot(ranks[idx_pre], val_pre, marker="^", color="C2")
    if idx_rec is not None: ax4.plot(ranks[idx_rec], val_rec, marker="^", color="C3")
    if idx_f1  is not None: ax4.plot(ranks[idx_f1],  val_f1,  marker="^", color="C4")
    if idx_mcc is not None: ax4.plot(ranks[idx_mcc], val_mcc, marker="^", color="C5")
    picked = [ranks[i] for i in [idx_acc, idx_pre, idx_rec, idx_f1, idx_mcc] if i is not None]
    if picked:
        med_r = int(np.median(np.array(picked)))
        pct = int(round((1.0 - eps) * 100))
        ax4.axvline(
            med_r, color="k", linestyle=":", alpha=0.6,
            label=f"{pct}% median rank r={med_r} (ε={eps:g})"
        )
    ax4.set_title("Reconstruction metrics (P vs clip[L-bin * A-bin])")
    ax4.set_xlabel("rank"); ax4.set_ylabel("score"); ax4.set_ylim(0, 1); ax4.legend(loc="best")

    # 5) Sparsity (Hoyer) on binarized L and A
    def hoyer_sparsity(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float); n = x.size
        l1 = np.sum(np.abs(x)); l2 = math.sqrt(np.sum(x ** 2))
        return 0.0 if l2 == 0 else (math.sqrt(n) - l1 / l2) / (math.sqrt(n) - 1.0)
    spars_L = [hoyer_sparsity(results[r].L_bin) for r in ranks]
    spars_A = [hoyer_sparsity(results[r].A_bin) for r in ranks]
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(ranks, spars_L, "o-", label="Hoyer Sparsity (L_bin)")
    ax5.plot(ranks, spars_A, "o-", label="Hoyer Sparsity (A_bin)")
    ax5.axhline(0.6, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    try:
        idx_first = next(i for i, v in enumerate(spars_L) if v >= 0.6)
        r_cross = ranks[idx_first]
        ax5.axvline(r_cross, color="C0", linestyle=":", alpha=0.7)
        ax5.annotate(f"L_bin ≥ 0.6 @ r={r_cross}", xy=(r_cross, 0.6), xytext=(r_cross, 0.65),
                     arrowprops=dict(arrowstyle="-", color="C0"), fontsize=8, color="C0")
    except StopIteration:
        pass
    ax5.set_title("Sparsity")
    ax5.set_xlabel("rank"); ax5.set_ylabel("sparsity"); ax5.set_ylim(0, 1); ax5.legend(loc="best")

    # 6) Forward conservation on **L** (OptICA-style)
    cons_j = compute_forward_conservation(results, similarity_threshold=corr_threshold, method="jaccard")
    cons_c = compute_forward_conservation(results, similarity_threshold=corr_threshold, method="cosine")
    # Convert to lists aligned with ranks for plotting
    cons_j_list = [cons_j[r] for r in ranks]
    cons_c_list = [cons_c[r] for r in ranks]
    # Also compute cumulative conservation for additional insight
    cum_cons_j = compute_cumulative_conservation(results, similarity_threshold=corr_threshold, method="jaccard")
    cum_cons_c = compute_cumulative_conservation(results, similarity_threshold=corr_threshold, method="cosine")
    cum_cons_j_list = [cum_cons_j[r] for r in ranks]
    cum_cons_c_list = [cum_cons_c[r] for r in ranks]
    # Compute non-dominant fractions (opposite of dominance for OptICA criterion)
    non_dom_L = [1.0 - float(np.mean(results[r].dominance_L >= dominance_threshold)) for r in ranks]
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.plot(ranks, cons_j_list, "o-", label=f"Conserved (Jaccard≥{corr_threshold:g}, L)", color='C0')
    ax6.plot(ranks, cons_c_list, "o-", label=f"Conserved (Cosine≥{corr_threshold:g}, L)", color='C1')
    ax6.plot(ranks, cum_cons_j_list, "s--", label="Cumulative conserved (Jaccard)", color='C0', alpha=0.6)
    ax6.plot(ranks, cum_cons_c_list, "s--", label="Cumulative conserved (Cosine)", color='C1', alpha=0.6)
    ax6.plot(ranks, non_dom_L, "^-", label=f"Non-dominant L (<{dominance_threshold:g})", color='C2', alpha=0.7)
    # Find and mark OptICA selection point (where conserved ≈ non-dominant) using Jaccard vs L dominance
    differences = [abs(cons_j_list[i] - non_dom_L[i]) for i in range(len(ranks))]
    optima_idx = int(np.argmin(differences))
    optimal_rank = ranks[optima_idx]
    ax6.axvline(optimal_rank, color='red', linestyle=':', alpha=0.7, label=f'OptICA point (r={optimal_rank})')
    ax6.set_title("Forward conservation (OptICA-style)")
    ax6.set_xlabel("rank")
    ax6.set_ylabel("fraction")
    ax6.set_ylim(0, 1)
    ax6.legend(loc="best", fontsize=8)
    ax6.grid(True, alpha=0.3)

    # 7) Notes
    ax7 = fig.add_subplot(gs[3, 1]); ax7.axis("off")
    note = ("Notes:\n"
            "• Perplexity uses held-out if present, else train (per observed entries).\n"
            "• Mass panel shows robust (winsorized) quantiles only.\n"
            "• Conservation is computed on L (Jaccard on L_bin columns; Cosine on L_norm columns).\n"
            "• AIC / BIC are computed using L_norm and A_norm.\n"
            "• JAIC-h is the Jaccard-based AIC heuristic computed using L_bin & A_bin.")
    ax7.text(0.02, 0.5, note, fontsize=10, va="center")

    fig.suptitle(f"optNMF selection panel — {title}", fontsize=14)
    png = os.path.join(outdir, f"{prefix}_optNMF_panel.png")
    pdf = os.path.join(outdir, f"{prefix}_optNMF_panel.pdf")
    fig.savefig(png, dpi=150); fig.savefig(pdf); plt.close(fig)


def make_holdout_mask(shape: Tuple[int, int], frac: float = 0.0, mode: str = "entry", seed: int = 0
                     ) -> Optional[np.ndarray]:
    if frac <= 0.0:
        return None
    rng = np.random.default_rng(seed)
    m, n = shape
    if mode == "entry":
        mask = (rng.random((m, n)) < frac).astype(float)
    elif mode == "row":
        row_sel = rng.random(m) < frac; mask = np.tile(row_sel[:, None], (1, n)).astype(float)
    elif mode == "col":
        col_sel = rng.random(n) < frac; mask = np.tile(col_sel[None, :], (m, 1)).astype(float)
    else:
        raise ValueError("holdout-mode must be one of {'entry','row','col'}")
    return mask


# ---------------------------------------------------------------------
# Interactive Sankey (Plotly)
# ---------------------------------------------------------------------

def _hoyer_sparsity_vec(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    l1 = np.sum(np.abs(x))
    l2 = math.sqrt(np.sum(x ** 2))
    if l2 == 0 or n <= 1:
        return 0.0
    return (math.sqrt(n) - l1 / l2) / (math.sqrt(n) - 1.0)


def export_node_metrics(results_by_rank: Dict[int, FitResult], path_csv: str) -> pd.DataFrame:
    rows = []
    for r, fr in sorted(results_by_rank.items()):
        rsize = fr.A_norm.shape[0]
        for k in range(rsize):
            sparsL = _hoyer_sparsity_vec(fr.L_bin[:, k])
            sparsA = _hoyer_sparsity_vec(fr.A_bin[k, :])
            rows.append({
                "rank": r,
                "k": k,
                "mass": float(fr.mass_fracs[k]) if fr.mass_fracs.size else np.nan,
                "mass_robust": float(fr.mass_fracs_robust[k]) if fr.mass_fracs_robust.size else np.nan,
                "dominance_L": float(fr.dominance_L[k]) if fr.dominance_L.size else np.nan,
                "dominance_A": float(fr.dominance_A[k]) if fr.dominance_A.size else np.nan,
                "sparsity_Lbin": float(sparsL),
                "sparsity_Abin": float(sparsA),
                "size_L_support": int(fr.L_bin[:, k].sum()),
                "size_A_support": int(fr.A_bin[k, :].sum()),
            })
    df = pd.DataFrame(rows)
    df.to_csv(path_csv, index=False)
    return df


def _build_sankey(edges: List[Tuple[int,int,int,int,float]],
                  node_metrics: pd.DataFrame,
                  out_html: str,
                  min_weight: float = 0.0,
                  title: str = "Component inheritance") -> Optional[str]:

    # Filter edges by min weight
    edges_f = [(rf, i, rt, j, w) for (rf, i, rt, j, w) in edges if w >= min_weight]
    if not edges_f:
        print("[warn] No edges after filtering; Sankey not created.")
        return None

    # Map nodes (rank,k) to indices
    nodes = sorted({(rf, i) for (rf, i, _, _, _) in edges_f} | {(rt, j) for (_, _, rt, j, _) in edges_f})
    node_index = {nk: idx for idx, nk in enumerate(nodes)}
    rank_values = sorted({r for (r, _) in nodes})

    # Colors by rank (simple gradient)
    rank_to_color = {r: f"rgba({int(40 + 180 * (ri / max(1, len(rank_values)-1))):d},"
                         f"{int(140 + 60 * (ri / max(1, len(rank_values)-1))):d},"
                         f"{int(240 - 200 * (ri / max(1, len(rank_values)-1))):d},0.8)"
                     for ri, r in enumerate(rank_values)}

    # Assemble node arrays
    labels = []
    colors = []
    hover = []
    for (r, k) in nodes:
        labels.append(f"r{r}:k{k}")
        colors.append(rank_to_color[r])
        # Pull metrics
        row = node_metrics[(node_metrics['rank'] == r) & (node_metrics['k'] == k)]
        if len(row) == 1:
            row = row.iloc[0]
            hv = (f"rank={r}, k={k}<br>"
                  f"mass={row['mass']:.4f}<br>"
                  f"mass_robust={row['mass_robust']:.4f}<br>"
                  f"dom_L={row['dominance_L']:.4f}, dom_A={row['dominance_A']:.4f}<br>"
                  f"sparsity_Lbin={row['sparsity_Lbin']:.3f}, sparsity_Abin={row['sparsity_Abin']:.3f}<br>"
                  f"|L⁺|={int(row['size_L_support'])}, |A⁺|={int(row['size_A_support'])}")
        else:
            hv = f"rank={r}, k={k}"
        hover.append(hv)

    # Assemble link arrays
    sources = [node_index[(rf, i)] for (rf, i, rt, j, w) in edges_f]
    targets = [node_index[(rt, j)] for (rf, i, rt, j, w) in edges_f]
    values  = [float(w) for (rf, i, rt, j, w) in edges_f]

    sankey = go.Sankey(
        node=dict(label=labels, color=colors, hovertemplate="%{customdata}<extra></extra>", customdata=hover),
        link=dict(source=sources, target=targets, value=values, hovertemplate="w=%{value:.3f}<extra></extra>")
    )
    fig = go.Figure(data=[sankey])
    fig.update_layout(title=title, font_size=12)
    plotly_plot(fig, filename=out_html, auto_open=False)
    return out_html


# ---------------------------- CV runner --------------------------------

def run_dual_cv(
    data: pd.DataFrame,
    ranks: List[int],
    seeds: List[int],
    beta_loss: str,
    init: str,
    max_iter: int,
    tol: float,
    alpha_W: float,
    alpha_H: float,
    threads_per_job: int,
    cv_k_genes: int,
    cv_k_strains: int,
    outdir: str,
    prefix: str,
) -> Optional[Dict[int, Dict[str, float]]]:
    """
    Execute the two K-fold CVs and compute weighted geometric-mean perplexities.
    Returns dict: rank -> {'perplexity_rows','perplexity_cols','perplexity_geom_weighted'}
    """
    X = data.values.astype(float)
    if (cv_k_genes is None or cv_k_genes <= 1) and (cv_k_strains is None or cv_k_strains <= 1):
        return None

    rows_cv = _cv_on_rows(
        X, ranks, seeds, cv_k_genes if (cv_k_genes and cv_k_genes > 1) else 0,
        beta_loss, init, max_iter, tol, alpha_W, alpha_H, threads_per_job
    )
    cols_cv = _cv_on_cols_via_transpose(
        X, ranks, seeds, cv_k_strains if (cv_k_strains and cv_k_strains > 1) else 0,
        beta_loss, init, max_iter, tol, alpha_W, alpha_H, threads_per_job
    )

    out: Dict[int, Dict[str, float]] = {}
    for r in ranks:
        ll_r, n_r = rows_cv.get(r, (0.0, 0))
        ll_c, n_c = cols_cv.get(r, (0.0, 0))

        # Convert to perplexities (lower=better)
        perp_rows = math.exp(-ll_r / max(1, n_r)) if n_r > 0 else np.nan
        perp_cols = math.exp(-ll_c / max(1, n_c)) if n_c > 0 else np.nan

        # Weighted geometric mean on the log scale
        w_total = max(1, n_r + n_c)
        log_geo = 0.0
        if n_r > 0:
            log_geo += (n_r / w_total) * math.log(perp_rows)
        if n_c > 0:
            log_geo += (n_c / w_total) * math.log(perp_cols)
        perp_geo = math.exp(log_geo) if (n_r + n_c) > 0 else np.nan

        out[r] = {
            "perplexity_rows": float(perp_rows) if not np.isnan(perp_rows) else np.nan,
            "perplexity_cols": float(perp_cols) if not np.isnan(perp_cols) else np.nan,
            "perplexity_geom_weighted": float(perp_geo) if not np.isnan(perp_geo) else np.nan,
            "n_rows_entries": int(n_r),
            "n_cols_entries": int(n_c),
        }

    # Save to disk
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{prefix}_cv_dual.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    # Also CSV for convenience
    rows = []
    for r in ranks:
        d = out[r]
        rows.append({"rank": r, **d})
        # end loop
    pd.DataFrame(rows).to_csv(os.path.join(outdir, f"{prefix}_cv_dual.csv"), index=False)

    # Console recommendation
    valid = [(r, d["perplexity_geom_weighted"]) for r, d in out.items() if np.isfinite(d["perplexity_geom_weighted"])]
    if valid:
        r_star, p_star = min(valid, key=lambda kv: kv[1])
        print(f"[CV] Suggested rank (min weighted geo-perplexity): r={r_star}  (geo-perp={p_star:.6f})")

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run NMF across ranks × seeds; OptICA-style trees on L; dual CV option.")
    p.add_argument("--csv", required=True, help="Input CSV (rows×cols), index allowed.")
    p.add_argument("--index-col", type=int, default=None, help="Column index to use as row index.")
    p.add_argument("--ranks", required=True, help="Comma list or slice 'start:end[:step]'.")
    p.add_argument("--seeds", required=True, help="Comma list or slice 'start:end[:step]'.")
    p.add_argument("--beta-loss", choices=["frobenius", "kullback-leibler", "itakura-saito"],
                   default="frobenius")
    p.add_argument("--init", choices=["nndsvd", "nndsvda", "random"], default=None,
                   help="If omitted: 'nndsvd' for Frobenius, 'nndsvda' otherwise.")
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--alpha-W", type=float, default=0.0)
    p.add_argument("--alpha-H", type=float, default=0.0)
    p.add_argument("--aic-dof", choices=["minus_r", "minus_r2"], default="minus_r",
                   help=("Effective DoF for AIC/BIC. "
                         "'minus_r' → k = r*(m+n)-r; 'minus_r2' → k = r*(m+n)-r^2."))
    p.add_argument("--mask-in-fit", action="store_true",
                   help="[Info] sklearn NMF does not support per-entry masking; mask is used only for scoring.")
    p.add_argument("--x-binarize",
                   choices=["kmeans", "gmm", "log-gmm", "beta", "otsu", "percentile"],
                   default=None,
                   help="Global binarizer for both L and A (kept for back-compat).")
    p.add_argument("--x-binarize-L",
                   choices=["kmeans", "gmm", "log-gmm", "beta", "otsu", "percentile"],
                   default=None,
                   help="Override binarizer for L (gene memberships).")
    p.add_argument("--x-binarize-A",
                   choices=["kmeans", "gmm", "log-gmm", "beta", "otsu", "percentile"],
                   default=None,
                   help="Override binarizer for A (sample memberships).")
    p.add_argument("--x-threshold", type=float, default=None,
                   help="Percentile q∈(0,1] or absolute value (used if method is 'percentile').")
    p.add_argument("--x-threshold-L", type=float, default=None,
                   help="Override percentile/absolute threshold for L.")
    p.add_argument("--x-threshold-A", type=float, default=None,
                   help="Override percentile/absolute threshold for A.")
    p.add_argument("--gmm-reg-covar", type=float, default=1e-6)

    # Holdout (optional scoring-only)
    p.add_argument("--holdout-frac", type=float, default=0.0)
    p.add_argument("--holdout-mode", choices=["entry", "row", "col"], default="entry")
    p.add_argument("--mask-seed", type=int, default=0)

    # Plotting thresholds
    p.add_argument("--corr-threshold", type=float, default=0.3,
                   help="Cosine threshold for correlation-based conservation (OptICA).")
    p.add_argument("--dominance-metric", choices=["hhi", "gini"], default="hhi")
    p.add_argument("--dominance-threshold", type=float, default=0.3,
                   help="Threshold on per-component dominance score (HHI or Gini).")
    p.add_argument("--mass-trim-q", type=float, default=0.99,
                   help="Winsorization quantile for robust mass fractions.")
    p.add_argument("--dens-vanish-mult", type=float, default=0.5,
                   help="Vanish criterion multiplier relative to uniform mass (r×mass < mult).")
    p.add_argument("--recon-early-eps", type=float, default=0.05,
                   help="eps for earliest rank within (1−eps) of metric max in the Reconstruction panel "
                   "(default 0.05 → within 95% of metric max).")

    # Visuals
    p.add_argument("--make-sankey", action="store_true", help="Write Plotly HTML Sankey diagrams for OptICA trees.")
    p.add_argument("--sankey-min-weight", type=float, default=0.3,
                   help="Min edge weight to include in Sankey (same scale as the corresponding tree metric).")

    # Dual CV
    p.add_argument("--cv-k-genes", type=int, default=0, help="K for gene-held-out (row) CV. 0/1 disables.")
    p.add_argument("--cv-k-strains", type=int, default=0, help="K for strain-held-out (column) CV. 0/1 disables.")

    # Execution
    p.add_argument("--outdir", default="optNMF_out")
    p.add_argument("--name", default="optNMF")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--threads-per-job", type=int, default=0, help="BLAS/OMP threads per job; 0 → library default.")
    p.add_argument("--format", choices=["csv", "parquet"], default="csv")
    return p.parse_args()


def main():
    args = parse_args()
    # Validate eps range for early reconstruction threshold
    if not (0.0 < args.recon_early_eps < 1.0):
        raise SystemExit("--recon-early-eps must be in (0,1).")

    # Read data
    data = pd.read_csv(args.csv, index_col=args.index_col, dtype='object')

    # Explicitly ensure index and column names are strings (preserves trailing zeroes)
    data.index = data.index.astype(str)
    data.columns = data.columns.astype(str)

    # Convert all data columns to float in place (throws error if non-numeric)
    data = data.astype(float)

    # Defaults for initialization
    init = args.init
    if init is None:
        init = "nndsvd" if args.beta_loss == "frobenius" else "nndsvda"

    ranks = parse_list_or_slice(args.ranks)
    seeds = parse_list_or_slice(args.seeds)

    # Hold-out mask for scoring (likelihoods only)
    holdout_mask = make_holdout_mask(data.shape, frac=args.holdout_frac, mode=args.holdout_mode, seed=args.mask_seed)
    if args.mask_in_fit:
        print("[info] sklearn.decomposition.NMF does not support masked fitting; mask is used only for scoring.")

    # Determine per-side methods (priority: side-specific -> global -> existing default)
    method_global = args.x_binarize
    method_L = args.x_binarize_L or method_global
    method_A = args.x_binarize_A or method_global
    
    # If still None, fall back to your file's existing default policy
    # (example: k-means as default; adapt this one-liner to whatever you used)
    if method_L is None:
        method_L = "kmeans"
    if method_A is None:
        method_A = "kmeans"
    
    # Thresholds
    thr_L = args.x_threshold_L if args.x_threshold_L is not None else args.x_threshold
    thr_A = args.x_threshold_A if args.x_threshold_A is not None else args.x_threshold
    
    results = run_grid(
        data=data, ranks=ranks, seeds=seeds,
        beta_loss=args.beta_loss, init=init, outdir=args.outdir, prefix=args.name,
        n_jobs=args.n_jobs, max_iter=args.max_iter, tol=args.tol,
        alpha_W=args.alpha_W, alpha_H=args.alpha_H, aic_dof=args.aic_dof,
        bin_method_L=method_L, bin_method_A=method_A,
        x_threshold_L=thr_L, x_threshold_A=thr_A,
        gmm_reg_covar=args.gmm_reg_covar,
        holdout_mask=holdout_mask, threads_per_job=args.threads_per_job,
        out_format=args.format, mass_trim_q=args.mass_trim_q,
        dominance_metric=args.dominance_metric,
        corr_threshold=args.corr_threshold,
        make_sankey=args.make_sankey, sankey_min_weight=args.sankey_min_weight,
    )

    # Dual CV (optional, does not affect reconstruction metrics)
    cv_summary = run_dual_cv(
        data=data, ranks=ranks, seeds=seeds, beta_loss=args.beta_loss,
        init=init, max_iter=args.max_iter, tol=args.tol,
        alpha_W=args.alpha_W, alpha_H=args.alpha_H,
        threads_per_job=args.threads_per_job,
        cv_k_genes=args.cv_k_genes, cv_k_strains=args.cv_k_strains,
        outdir=args.outdir, prefix=args.name
    )

    plot_panels(
        data=data, results=results, beta_loss=args.beta_loss,
        outdir=args.outdir, prefix=args.name, title=os.path.basename(args.csv),
        dens_vanish_mult=args.dens_vanish_mult, used_holdout=(holdout_mask is not None),
        dominance_threshold=args.dominance_threshold, corr_threshold=args.corr_threshold,
        make_sankey=args.make_sankey,
        sankey_min_weight=args.sankey_min_weight,
        cv_summary=cv_summary,
        recon_early_eps=args.recon_early_eps,
    )

    # Console diagnostics
    print("=== Summary (best seed per rank, BIC-selected) ===")
    for r in sorted(results.keys()):
        fr = results[r]
        if fr.loglike_test is not None and fr.n_obs_test:
            perp_te = math.exp(-fr.loglike_test / max(1, fr.n_obs_test))
            print(f"rank={r:>3d} seed={fr.seed:>4d}  perplexity(te)={perp_te:.6f}  "
                  f"AIC(te)={fr.aic_test:.3f}  BIC(te)={fr.bic_test:.3f}  "
                  f"AIC(tr)={fr.aic_train:.3f}  BIC(tr)={fr.bic_train:.3f}  "
                  f"JAIC-h={fr.jaic_h:.3f}  Jaccard={fr.jaccard_index:.4f}")
        else:
            perp_tr = math.exp(-fr.loglike_train / max(1, fr.n_obs_train))
            print(f"rank={r:>3d} seed={fr.seed:>4d}  perplexity(tr)={perp_tr:.6f}  "
                  f"AIC(tr)={fr.aic_train:.3f}  BIC(tr)={fr.bic_train:.3f}  "
                  f"JAIC-h={fr.jaic_h:.3f}  Jaccard={fr.jaccard_index:.4f}")

    # Optional CV recommendation
    if cv_summary is not None:
        best = min(((r, d["perplexity_geom_weighted"]) for r, d in cv_summary.items() if np.isfinite(d["perplexity_geom_weighted"])),
                   key=lambda kv: kv[1], default=None)
        if best is not None:
            print(f"[CV] Suggested rank = {best[0]} (min weighted geo-perplexity = {best[1]:.6f})")


if __name__ == "__main__":
    main()