from AlgorithmImports import *
import numpy as np
from collections import deque


class ResidualPolicyEnsembleQC(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2018, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # --- Universe (simple para ejemplo) ---
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        # --- Config ---
        self.lookback = 252  # para features/normalización
        self.rebalance_days = 21  # mensual aprox
        self.selection_days = 63  # trimestral aprox
        self.target_leverage = 1.0

        # Ensemble config
        self.N = 200  # número de policies candidatas
        self.k = 16  # dimensión subespacio delta (direcciones)
        self.topX = 12  # policies finales en el portfolio
        self.sigma = 0.05  # amplitud del ruido en coeficientes alpha
        self.beta = 0.35  # penalización por correlación en selección
        self.turnover_penalty = 0.25

        # Throttle trading
        self.trade_threshold = 0.05  # no rebalance si cambio de target es pequeño

        # --- Data buffers ---
        self.prices = deque(maxlen=self.lookback + 5)
        self.returns = deque(maxlen=self.lookback + 5)

        # Per-policy tracking (returns y turnover)
        self.policy_returns = {i: deque(maxlen=252) for i in range(self.N)}
        self.policy_prev_p = {i: 0.0 for i in range(self.N)}

        # Selected policies
        self.selected = []
        self.weights = {}

        # Schedule
        self.last_rebalance = None
        self.last_selection = None

        # --- Build base model + directions for deltas ---
        # MLP: d -> 32 -> 16 -> 1 (logit z). Residual uses logit difference then tanh.
        self.d = self._feature_dim()
        self.hidden1 = 32
        self.hidden2 = 16

        # Base weights w0 (placeholder random but small).
        # En producción: cargar pesos entrenados offline con Opción B.
        rng = np.random.default_rng(123)
        self.w0 = self._init_base_weights(rng)

        # Subspace directions in parameter space of (ONLY last layer) recommended:
        # last layer has shape (16, 1) + bias (1,)
        # We'll build u_j over flattened last-layer params.
        self.u = self._make_last_layer_directions(rng, self.k)

        # Sample alphas for N policies
        self.alpha = rng.normal(0.0, self.sigma, size=(self.N, self.k))

        # Warmup so rolling stats exist
        self.SetWarmUp(self.lookback)

        self.Debug("Initialized residual-policy ensemble.")

    def OnData(self, data: Slice):
        if self.IsWarmingUp:
            self._update_buffers(data)
            return

        if not data.ContainsKey(self.symbol):
            return

        self._update_buffers(data)
        if len(self.returns) < 50:
            return

        t = self.Time

        # Periodic selection (quarterly)
        if self.last_selection is None or (t - self.last_selection).days >= self.selection_days:
            self._select_policies()
            self.last_selection = t

        # Periodic rebalance (monthly-ish)
        if self.last_rebalance is None or (t - self.last_rebalance).days >= self.rebalance_days:
            self._rebalance()
            self.last_rebalance = t

    # ---------------------------
    # Core: selection + execution
    # ---------------------------
    def _select_policies(self):
        # Need enough history
        min_hist = 60
        valid = []
        scores = {}

        for i in range(self.N):
            r = np.array(self.policy_returns[i], dtype=float)
            if r.size < min_hist:
                continue

            # Score: Sharpe approx minus turnover penalty
            mu = r.mean()
            sd = r.std() + 1e-12
            sharpe = np.sqrt(252) * mu / sd

            # turnover proxy: average abs delta in p
            # we tracked via p changes accumulating into policy_returns already, but
            # we'll approximate using recent p changes stored in prev (not perfect).
            # Better: track per-policy turnover explicitly if you want.
            # Here: mild penalty based on sd of returns as a proxy for aggressiveness.
            turnover_proxy = min(1.0, sd * 10.0)
            score = sharpe - self.turnover_penalty * turnover_proxy

            scores[i] = score
            valid.append(i)

        if len(valid) < max(10, self.topX):
            self.Debug(f"Selection skipped: only {len(valid)} valid policies.")
            return

        # Preselect top M by score
        M = min(50, len(valid))
        topM = sorted(valid, key=lambda i: scores[i], reverse=True)[:M]

        # Build correlation matrix on returns for topM
        R = []
        for i in topM:
            R.append(np.array(self.policy_returns[i], dtype=float))
        # Align by truncating to min length
        L = min(len(x) for x in R)
        R = np.stack([x[-L:] for x in R], axis=0)
        # If variance is zero for some, add tiny noise
        R = R + 1e-12 * np.random.randn(*R.shape)

        corr = np.corrcoef(R)

        # Greedy diversity selection
        chosen_idx = []
        # pick best by score
        best = 0
        for j in range(1, len(topM)):
            if scores[topM[j]] > scores[topM[best]]:
                best = j
        chosen_idx.append(best)

        while len(chosen_idx) < self.topX and len(chosen_idx) < len(topM):
            best_j = None
            best_val = -1e18
            for j in range(len(topM)):
                if j in chosen_idx:
                    continue
                mean_corr = float(np.mean([abs(corr[j, c]) for c in chosen_idx]))
                val = scores[topM[j]] - self.beta * mean_corr
                if val > best_val:
                    best_val = val
                    best_j = j
            chosen_idx.append(best_j)

        self.selected = [topM[j] for j in chosen_idx]
        # weights: equal weight
        w = 1.0 / len(self.selected)
        self.weights = {i: w for i in self.selected}

        # Log
        sel_scores = [scores[i] for i in self.selected]
        self.Debug(
            f"Selected {len(self.selected)} policies. "
            f"Score range: {min(sel_scores):.2f}..{max(sel_scores):.2f}"
        )

    def _rebalance(self):
        if not self.selected:
            # If nothing selected, stay flat
            self.SetHoldings(self.symbol, 0.0)
            return

        s = self._compute_state()
        # Aggregate signal
        p_total = 0.0
        for i in self.selected:
            p_i = self._policy_output(i, s)
            p_total += self.weights[i] * p_i

        # Cap + leverage
        p_total = float(np.clip(p_total, -1.0, 1.0)) * self.target_leverage

        # Throttle tiny changes
        current = self.Portfolio[self.symbol].HoldingsValue / max(
            1.0, self.Portfolio.TotalPortfolioValue
        )
        if abs(p_total - current) < self.trade_threshold:
            return

        self.SetHoldings(self.symbol, p_total)

    # ---------------------------
    # Policy mechanics
    # ---------------------------
    def _policy_output(self, i: int, s: np.ndarray) -> float:
        # Residual in logits: z(w0+Δ) - z(w0) then tanh
        z0 = self._forward_logit(s, self.w0)

        w1 = self._apply_delta_to_last_layer(self.w0, self.alpha[i])
        z1 = self._forward_logit(s, w1)

        z_res = z1 - z0
        p = np.tanh(z_res)
        return float(p)

    def _forward_logit(self, s: np.ndarray, w: dict) -> float:
        # MLP: s -> relu -> relu -> linear (logit)
        x = s
        h1 = np.maximum(0.0, x @ w["W1"] + w["b1"])
        h2 = np.maximum(0.0, h1 @ w["W2"] + w["b2"])
        z = float(h2 @ w["W3"] + w["b3"])
        return z

    def _apply_delta_to_last_layer(self, w0: dict, alpha_i: np.ndarray) -> dict:
        # Create a copy with modified last layer params only.
        # Δ = sum_j alpha_j * u_j, where u_j spans last-layer param vector.
        delta_vec = np.zeros(self.hidden2 + 1, dtype=float)  # W3 flattened (16) + b3(1)
        for j in range(self.k):
            delta_vec += alpha_i[j] * self.u[j]

        w = {
            "W1": w0["W1"],
            "b1": w0["b1"],
            "W2": w0["W2"],
            "b2": w0["b2"],
            "W3": w0["W3"].copy(),
            "b3": float(w0["b3"]),
        }

        w["W3"] = (w["W3"].reshape(-1) + delta_vec[: self.hidden2]).reshape(
            self.hidden2, 1
        )
        w["b3"] = float(w["b3"] + delta_vec[-1])
        return w

    # ---------------------------
    # Data + features
    # ---------------------------
    def _update_buffers(self, data: Slice):
        price = float(data[self.symbol].Close)
        if self.prices:
            prev = self.prices[-1]
            r = np.log(price / prev)
            self.returns.append(r)
        self.prices.append(price)

        # Update per-policy returns using "paper" pnl from policy positions
        # We compute each policy's position at close and apply next-day return style approximation.
        # This is an approximation but works for selection.
        if len(self.returns) >= 2:
            s = self._compute_state()
            last_r = self.returns[-1]

            for i in range(self.N):
                p = self._policy_output(i, s)

                # policy pnl: position * return
                pnl = p * last_r

                # turnover penalty inside returns stream (optional)
                dp = abs(p - self.policy_prev_p[i])
                pnl_adj = pnl - 0.0005 * dp  # crude cost proxy
                self.policy_prev_p[i] = p

                self.policy_returns[i].append(pnl_adj)

    def _compute_state(self) -> np.ndarray:
        # Build features; all normalized roughly.
        # Keep it stable and simple.
        prices = np.array(self.prices, dtype=float)
        rets = np.array(self.returns, dtype=float)

        # Basic stats
        r1 = rets[-1]
        r5 = rets[-5:].sum() if rets.size >= 5 else rets.sum()
        r20 = rets[-20:].sum() if rets.size >= 20 else rets.sum()

        vol20 = rets[-20:].std() if rets.size >= 20 else rets.std()
        vol60 = rets[-60:].std() if rets.size >= 60 else rets.std()

        # Moving averages
        ema50 = self._ema(prices, 50)
        ema200 = self._ema(prices, 200)
        p = prices[-1]
        dist_ema200 = (p - ema200) / (ema200 + 1e-12)

        # Trend slope proxy: difference of EMA50 over 20 days
        ema50_series = self._ema_series(prices, 50)
        if len(ema50_series) >= 21:
            slope = (ema50_series[-1] - ema50_series[-21]) / (
                abs(ema50_series[-21]) + 1e-12
            )
        else:
            slope = 0.0

        # VWAP proxy (daily data: use SMA as placeholder)
        sma20 = prices[-20:].mean() if prices.size >= 20 else prices.mean()
        z_vwap = (p - sma20) / (prices[-20:].std() + 1e-12) if prices.size >= 20 else 0.0

        # Normalize returns by vol20
        scale = vol20 + 1e-6
        feats = np.array(
            [
                r1 / scale,
                r5 / scale,
                r20 / scale,
                vol20 * np.sqrt(252),
                vol60 * np.sqrt(252),
                slope,
                dist_ema200,
                z_vwap,
            ],
            dtype=float,
        )

        # Pad/trim to d
        if feats.size < self.d:
            feats = np.pad(feats, (0, self.d - feats.size))
        elif feats.size > self.d:
            feats = feats[: self.d]

        # Final stabilization
        feats = np.clip(feats, -10, 10)
        return feats

    def _feature_dim(self) -> int:
        # Match compute_state() feature count
        return 8

    def _ema(self, x: np.ndarray, period: int) -> float:
        if x.size < 2:
            return float(x[-1]) if x.size else 0.0
        alpha = 2.0 / (period + 1.0)
        ema = x[0]
        for v in x[1:]:
            ema = alpha * v + (1 - alpha) * ema
        return float(ema)

    def _ema_series(self, x: np.ndarray, period: int):
        if x.size == 0:
            return []
        alpha = 2.0 / (period + 1.0)
        ema = x[0]
        out = [float(ema)]
        for v in x[1:]:
            ema = alpha * v + (1 - alpha) * ema
            out.append(float(ema))
        return out

    # ---------------------------
    # Base weights + directions
    # ---------------------------
    def _init_base_weights(self, rng):
        # Small random weights: en tu caso real, aquí cargas w0 entrenado con la loss Opción B.
        W1 = rng.normal(0, 0.02, size=(self.d, self.hidden1))
        b1 = np.zeros(self.hidden1)
        W2 = rng.normal(0, 0.02, size=(self.hidden1, self.hidden2))
        b2 = np.zeros(self.hidden2)
        W3 = rng.normal(0, 0.02, size=(self.hidden2, 1))
        b3 = 0.0
        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    def _make_last_layer_directions(self, rng, k):
        # Directions span last-layer params only: 16 weights + 1 bias
        dim = self.hidden2 + 1
        U = []
        # Create random directions and normalize (approx orthonormal-ish)
        for _ in range(k):
            v = rng.normal(0, 1.0, size=dim)
            v = v / (np.linalg.norm(v) + 1e-12)
            U.append(v)
        return U
