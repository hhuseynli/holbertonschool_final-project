"""ML enrichment engine — maps each listing to pre-computed analytics.

Each listing has (lat, lon) which maps to an H3 cell via ``h3.latlng_to_cell``.
Pre-computed artifacts (predictions, panel, cell_features) are loaded once at
startup and keyed by H3 cell for O(1) lookup.
"""

from __future__ import annotations

from pathlib import Path

import h3
import numpy as np
import pandas as pd

from bakuml import config
from bakuml.geo import haversine_km
from bakuml.viz import load_artifacts


class ListingEnricher:
    """Attaches ML analytics to individual listings."""

    def __init__(self, artifacts_dir: Path = config.ARTIFACTS_DIR) -> None:
        self.arts = load_artifacts(artifacts_dir)
        self._build_lookups()

    def _build_lookups(self) -> None:
        predictions = self.arts.get("predictions")
        panel = self.arts.get("panel")
        features = self.arts.get("cell_features")

        # Prediction lookup: (h3, scenario, horizon) -> row dict
        self._pred: dict[tuple, dict] = {}
        if predictions is not None:
            for _, row in predictions.iterrows():
                key = (row["h3"], row["scenario"], int(row["horizon_months"]))
                self._pred[key] = row.to_dict()

        # Last observed price per cell
        self._last_price: dict[str, float] = {}
        self._momentum: dict[str, float] = {}
        if panel is not None:
            sorted_panel = panel.sort_values("month", kind="stable")
            for cell, grp in sorted_panel.groupby("h3"):
                self._last_price[cell] = float(grp["price_azn_m2_median"].iloc[-1])
                prices = grp["price_azn_m2_median"].values
                if len(prices) >= 4:
                    recent = prices[-3:].mean()
                    earlier = prices[-6:-3].mean() if len(prices) >= 7 else prices[:-3].mean()
                    self._momentum[cell] = float((recent - earlier) / earlier * 100) if earlier > 0 else 0.0

        # Cell features lookup: h3 -> dict
        self._features: dict[str, dict] = {}
        if features is not None:
            for _, row in features.iterrows():
                self._features[row["h3"]] = row.to_dict()

        # Metro stations for distance calc
        self._metro_stations = [
            (s.lat, s.lon) for s in config.METRO_STATIONS
            if s.opened or s.simulated_open
        ]

        # Compute percentile distributions for score normalization
        self._compute_distributions()

    def _compute_distributions(self) -> None:
        """Pre-compute value ranges for percentile-based scoring."""
        appreciations = []
        prices = []
        for key, row in self._pred.items():
            if key[1] == "baseline" and key[2] == 12:
                appreciations.append(row.get("appreciation_pct", 0))
        for v in self._last_price.values():
            prices.append(v)

        self._appr_vals = np.array(appreciations) if appreciations else np.array([0.0])
        self._price_vals = np.array(prices) if prices else np.array([0.0])

        gravities = [f.get("node_gravity", 0) for f in self._features.values()]
        self._gravity_vals = np.array(gravities) if gravities else np.array([0.0])

        metro_dists = [f.get("dist_metro_km", 50) for f in self._features.values()]
        self._metro_vals = np.array(metro_dists) if metro_dists else np.array([50.0])

        mom_vals = list(self._momentum.values())
        self._mom_vals = np.array(mom_vals) if mom_vals else np.array([0.0])

    def _percentile_score(self, value: float, distribution: np.ndarray, invert: bool = False) -> float:
        """Map a value to 0-100 based on its percentile in the distribution."""
        pct = float(np.searchsorted(np.sort(distribution), value) / len(distribution) * 100)
        return 100 - pct if invert else pct

    def enrich(self, listing: dict) -> dict:
        """Add an 'analytics' sub-dict to a listing."""
        lat = listing.get("lat")
        lon = listing.get("lon")
        if lat is None or lon is None:
            listing["analytics"] = None
            return listing

        cell = h3.latlng_to_cell(float(lat), float(lon), config.H3_RESOLUTION)
        price_m2 = listing.get("price_azn_m2") or 0

        # Prediction lookup (baseline, 12m)
        pred_12 = self._pred.get((cell, "baseline", 12), {})
        pred_24 = self._pred.get((cell, "baseline", 24), {})

        q10 = pred_12.get("q10")
        q50 = pred_12.get("q50")
        q90 = pred_12.get("q90")

        # Price position
        if q50 and price_m2 > 0:
            if price_m2 < q10:
                price_position = "underpriced"
            elif price_m2 > q90:
                price_position = "overpriced"
            elif price_m2 < q50:
                price_position = "below average"
            else:
                price_position = "above average"
            price_percentile = float(np.clip((price_m2 - q10) / (q90 - q10) * 100, 0, 100)) if q90 != q10 else 50
        else:
            price_position = "unknown"
            price_percentile = 50

        # Cell features
        feat = self._features.get(cell, {})
        dist_metro = feat.get("dist_metro_km")
        node_gravity = feat.get("node_gravity")

        # Momentum
        momentum = self._momentum.get(cell, 0.0)

        # Investment score
        score = self._investment_score(
            appreciation=pred_12.get("appreciation_pct", 0),
            price_m2=price_m2,
            q50=q50,
            dist_metro=dist_metro,
            node_gravity=node_gravity,
            momentum=momentum,
        )

        analytics = {
            "h3_cell": cell,
            "appreciation_12m_pct": pred_12.get("appreciation_pct"),
            "appreciation_24m_pct": pred_24.get("appreciation_pct"),
            "hotspot_prob": pred_12.get("hotspot_prob"),
            "q10": q10,
            "q50": q50,
            "q90": q90,
            "price_position": price_position,
            "price_percentile": round(price_percentile, 1),
            "current_cell_price": self._last_price.get(cell),
            "dist_metro_km": round(dist_metro, 2) if dist_metro is not None else None,
            "node_gravity": round(node_gravity, 3) if node_gravity is not None else None,
            "dist_centre_km": round(feat.get("dist_centre_km", 0), 1) if feat else None,
            "in_redev_zone": bool(feat.get("in_redev_zone", 0)) if feat else False,
            "momentum_3m_pct": round(momentum, 1),
            "investment_score": score,
        }
        listing["analytics"] = analytics
        return listing

    def _investment_score(
        self,
        appreciation: float,
        price_m2: float,
        q50: float | None,
        dist_metro: float | None,
        node_gravity: float | None,
        momentum: float,
    ) -> int:
        """Compute composite investment score 0-100."""
        # 30% appreciation
        s_appr = self._percentile_score(appreciation, self._appr_vals)

        # 25% price positioning (underpriced = high score)
        if q50 and price_m2 > 0:
            discount_pct = (q50 - price_m2) / q50 * 100
            s_price = float(np.clip(50 + discount_pct * 2, 0, 100))
        else:
            s_price = 50

        # 20% metro proximity (closer = higher, inverted)
        s_metro = self._percentile_score(dist_metro or 50, self._metro_vals, invert=True)

        # 15% masterplan alignment
        s_plan = self._percentile_score(node_gravity or 0, self._gravity_vals)

        # 10% momentum
        s_mom = self._percentile_score(momentum, self._mom_vals)

        raw = 0.30 * s_appr + 0.25 * s_price + 0.20 * s_metro + 0.15 * s_plan + 0.10 * s_mom
        return int(np.clip(round(raw), 0, 100))

    def cell_analytics(self, h3_id: str, scenario: str = "baseline") -> dict:
        """Full analytics for a single H3 cell."""
        from bakuml.viz import cell_forecast, cell_history

        panel = self.arts.get("panel")
        predictions = self.arts.get("predictions")

        history = []
        if panel is not None:
            hist_df = cell_history(panel, h3_id)
            history = hist_df.to_dict(orient="records")

        forecast = []
        if predictions is not None:
            fc_df = cell_forecast(predictions, h3_id, scenario)
            forecast = fc_df.to_dict(orient="records")

        feat = self._features.get(h3_id, {})

        return {
            "h3": h3_id,
            "history": history,
            "forecast": forecast,
            "features": {k: v for k, v in feat.items() if k != "h3"},
            "current_price": self._last_price.get(h3_id),
            "momentum_3m_pct": round(self._momentum.get(h3_id, 0), 1),
        }
