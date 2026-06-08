"""
Typed hierarchical hurdle — Tier 3 prototype.

The first paper's stated future work is to stop treating all sightings as
equivalent and instead model their *type* (seen / proof-of-concept / exploited),
to "establish a link with the actual exploitation of a vulnerability". Tiers 1–2
pooled the *total* daily count; here we pool **each type separately**.

Why per-type and not one pooled total?

* The types have genuinely different dynamics and base rates (see the per-type
  population priors reported by ``run_typed``): ``exploited`` activity is rarer
  and more concentrated than the chatter-like ``seen`` stream. A single pooled
  model blends these, whereas a per-type model can shrink each toward *its own*
  population.
* It yields the operationally valuable output directly: a forecast of the
  ``exploited`` signal specifically, with its own prediction interval — not
  buried inside an all-types total.

The construction simply reuses the Tier-2 ``HierarchicalHurdle`` once per type,
each with a type-specific ``PopulationPrior``. The total forecast is the sum of
the per-type predictive samples (drawn independently — an acknowledged
simplification, since in reality a PoC release and exploitation are correlated;
modelling that dependence is left for future work).
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ..data import SIGHTING_TYPES
from .hierarchical import HierarchicalHurdle, PopulationPrior, fit_population_prior


def fit_typed_population_priors(
    typed_corpus: dict[str, dict[str, pd.Series]], types: list[str] = SIGHTING_TYPES
) -> dict[str, PopulationPrior]:
    """Estimate one population prior per sighting type from the corpus.

    ``typed_corpus`` maps ``vuln_id -> {type -> daily series}``. For each type we
    pool that type's series across all CVEs.
    """
    return {t: fit_population_prior([cve[t] for cve in typed_corpus.values() if t in cve]) for t in types}


class TypedHierarchicalHurdle:
    """One pooled hurdle per sighting type; total = sum of per-type forecasts."""

    def __init__(self, priors: dict[str, PopulationPrior], seed: int = 0) -> None:
        self.name = "typed_hier_hurdle"
        # Distinct seeds per type so the independent per-type sampling streams do
        # not move in lockstep when summed into the total.
        self._models = {
            t: HierarchicalHurdle(prior, seed=seed + i) for i, (t, prior) in enumerate(priors.items())
        }

    @property
    def types(self) -> list[str]:
        return list(self._models)

    def fit(self, typed_train: dict[str, pd.Series]) -> "TypedHierarchicalHurdle":
        for t, model in self._models.items():
            model.fit(typed_train[t])
        return self

    def sample_type(self, sighting_type: str, horizon: int, n_samples: int = 1000) -> np.ndarray:
        return self._models[sighting_type].sample(horizon, n_samples)

    def sample_total(self, horizon: int, n_samples: int = 1000) -> np.ndarray:
        """Predictive samples of the typed total (sum over types), per-type independent."""
        total = np.zeros((n_samples, horizon), dtype=np.int64)
        for model in self._models.values():
            total = total + model.sample(horizon, n_samples)
        return total
