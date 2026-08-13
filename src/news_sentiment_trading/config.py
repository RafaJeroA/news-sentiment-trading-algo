"""Validated configuration and deterministic configuration hashes."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal, cast

FeatureKind = Literal["ratio", "log_ratio"]
MissingPolicy = Literal["no_fill", "ffill_1"]
PortfolioKind = Literal["long_only", "market_neutral", "directional"]
ExecutionConvention = Literal["next_adjusted_open", "lagged_adjusted_close"]


@dataclass(frozen=True, order=True)
class SignalParameters:
    """A single preregistered signal parameter combination."""

    window: int
    z_threshold: float
    rvt_quantile: float
    feature_kind: FeatureKind = "ratio"
    missing_policy: MissingPolicy = "no_fill"
    holding_days: int = 1


@dataclass(frozen=True)
class ResearchConfig:
    seed: int
    annualization: int
    risk_free_rate: float
    primary_cost_bps: int
    cost_scenarios_bps: tuple[int, ...]


@dataclass(frozen=True)
class UniverseConfig:
    primary: tuple[str, ...]
    start: str
    end: str


@dataclass(frozen=True)
class ExecutionConfig:
    convention: ExecutionConvention
    holding_days: int
    missing_policy: MissingPolicy
    bounded_forward_fill_days: int


@dataclass(frozen=True)
class PortfolioConfig:
    primary: PortfolioKind
    long_only_weight_per_asset: float
    neutral_gross_limit: float
    neutral_net_limit: float
    one_sided_neutral_policy: Literal["cash"]


@dataclass(frozen=True)
class WalkForwardConfig:
    outer_initial_sessions: int
    outer_test_sessions: int
    inner_initial_sessions: int
    inner_test_sessions: int
    minimum_active_fraction: float
    minimum_active_asset_days: int
    maximum_annualized_turnover: float
    selection_cost_bps: int


@dataclass(frozen=True)
class FeatureGridConfig:
    primary_kind: FeatureKind
    windows: tuple[int, ...]
    z_thresholds: tuple[float, ...]
    rvt_quantiles: tuple[float, ...]
    fixed_window: int
    fixed_z_threshold: float
    fixed_rvt_quantile: float

    def parameters(self, execution: ExecutionConfig) -> tuple[SignalParameters, ...]:
        return tuple(
            SignalParameters(
                window=int(window),
                z_threshold=float(z_threshold),
                rvt_quantile=float(rvt_quantile),
                feature_kind=self.primary_kind,
                missing_policy=execution.missing_policy,
                holding_days=execution.holding_days,
            )
            for window, z_threshold, rvt_quantile in product(
                self.windows, self.z_thresholds, self.rvt_quantiles
            )
        )

    def fixed(self, execution: ExecutionConfig) -> SignalParameters:
        return SignalParameters(
            window=self.fixed_window,
            z_threshold=self.fixed_z_threshold,
            rvt_quantile=self.fixed_rvt_quantile,
            feature_kind=self.primary_kind,
            missing_policy=execution.missing_policy,
            holding_days=execution.holding_days,
        )


@dataclass(frozen=True)
class PrimaryConfig:
    research: ResearchConfig
    universe: UniverseConfig
    execution: ExecutionConfig
    portfolio: PortfolioConfig
    walk_forward: WalkForwardConfig
    features: FeatureGridConfig

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_EXPECTED_SECTIONS: dict[str, set[str]] = {
    "research": {
        "seed",
        "annualization",
        "risk_free_rate",
        "primary_cost_bps",
        "cost_scenarios_bps",
    },
    "universe": {"primary", "start", "end"},
    "execution": {
        "convention",
        "holding_days",
        "missing_policy",
        "bounded_forward_fill_days",
    },
    "portfolio": {
        "primary",
        "long_only_weight_per_asset",
        "neutral_gross_limit",
        "neutral_net_limit",
        "one_sided_neutral_policy",
    },
    "walk_forward": {
        "outer_initial_sessions",
        "outer_test_sessions",
        "inner_initial_sessions",
        "inner_test_sessions",
        "minimum_active_fraction",
        "minimum_active_asset_days",
        "maximum_annualized_turnover",
        "selection_cost_bps",
    },
    "features": {
        "primary_kind",
        "windows",
        "z_thresholds",
        "rvt_quantiles",
        "fixed_window",
        "fixed_z_threshold",
        "fixed_rvt_quantile",
    },
}


def _validate_keys(raw: dict[str, Any]) -> None:
    unknown_sections = set(raw) - set(_EXPECTED_SECTIONS)
    missing_sections = set(_EXPECTED_SECTIONS) - set(raw)
    if unknown_sections or missing_sections:
        raise ValueError(
            f"configuration sections differ: missing={sorted(missing_sections)}, "
            f"unknown={sorted(unknown_sections)}"
        )
    for section, expected in _EXPECTED_SECTIONS.items():
        actual = set(cast(dict[str, Any], raw[section]))
        if actual != expected:
            raise ValueError(
                f"configuration keys for {section} differ: "
                f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
            )


def load_config(path: str | Path) -> PrimaryConfig:
    """Load a TOML configuration, rejecting unknown and missing fields."""

    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    _validate_keys(raw)

    research = raw["research"]
    universe = raw["universe"]
    execution = raw["execution"]
    portfolio = raw["portfolio"]
    walk_forward = raw["walk_forward"]
    features = raw["features"]

    config = PrimaryConfig(
        research=ResearchConfig(
            seed=int(research["seed"]),
            annualization=int(research["annualization"]),
            risk_free_rate=float(research["risk_free_rate"]),
            primary_cost_bps=int(research["primary_cost_bps"]),
            cost_scenarios_bps=tuple(int(x) for x in research["cost_scenarios_bps"]),
        ),
        universe=UniverseConfig(
            primary=tuple(str(x) for x in universe["primary"]),
            start=str(universe["start"]),
            end=str(universe["end"]),
        ),
        execution=ExecutionConfig(
            convention=cast(ExecutionConvention, execution["convention"]),
            holding_days=int(execution["holding_days"]),
            missing_policy=cast(MissingPolicy, execution["missing_policy"]),
            bounded_forward_fill_days=int(execution["bounded_forward_fill_days"]),
        ),
        portfolio=PortfolioConfig(
            primary=cast(PortfolioKind, portfolio["primary"]),
            long_only_weight_per_asset=float(portfolio["long_only_weight_per_asset"]),
            neutral_gross_limit=float(portfolio["neutral_gross_limit"]),
            neutral_net_limit=float(portfolio["neutral_net_limit"]),
            one_sided_neutral_policy=cast(Literal["cash"], portfolio["one_sided_neutral_policy"]),
        ),
        walk_forward=WalkForwardConfig(
            outer_initial_sessions=int(walk_forward["outer_initial_sessions"]),
            outer_test_sessions=int(walk_forward["outer_test_sessions"]),
            inner_initial_sessions=int(walk_forward["inner_initial_sessions"]),
            inner_test_sessions=int(walk_forward["inner_test_sessions"]),
            minimum_active_fraction=float(walk_forward["minimum_active_fraction"]),
            minimum_active_asset_days=int(walk_forward["minimum_active_asset_days"]),
            maximum_annualized_turnover=float(walk_forward["maximum_annualized_turnover"]),
            selection_cost_bps=int(walk_forward["selection_cost_bps"]),
        ),
        features=FeatureGridConfig(
            primary_kind=cast(FeatureKind, features["primary_kind"]),
            windows=tuple(int(x) for x in features["windows"]),
            z_thresholds=tuple(float(x) for x in features["z_thresholds"]),
            rvt_quantiles=tuple(float(x) for x in features["rvt_quantiles"]),
            fixed_window=int(features["fixed_window"]),
            fixed_z_threshold=float(features["fixed_z_threshold"]),
            fixed_rvt_quantile=float(features["fixed_rvt_quantile"]),
        ),
    )
    _validate_values(config)
    return config


def _validate_values(config: PrimaryConfig) -> None:
    if len(set(config.universe.primary)) != len(config.universe.primary):
        raise ValueError("primary tickers must be unique")
    if not config.universe.primary:
        raise ValueError("primary universe cannot be empty")
    if config.execution.holding_days < 1:
        raise ValueError("holding_days must be positive")
    if config.execution.bounded_forward_fill_days != 1:
        raise ValueError("bounded_forward_fill_days must be exactly one")
    if config.execution.missing_policy not in {"no_fill", "ffill_1"}:
        raise ValueError("unsupported missing policy")
    if config.execution.convention not in {
        "next_adjusted_open",
        "lagged_adjusted_close",
    }:
        raise ValueError("unsupported execution convention")
    if not 0 < config.portfolio.long_only_weight_per_asset <= 1:
        raise ValueError("long-only per-asset weight must be in (0, 1]")
    if config.portfolio.neutral_net_limit != 0:
        raise ValueError("the preregistered neutral net limit is exactly zero")
    if config.walk_forward.outer_initial_sessions < max(config.features.windows):
        raise ValueError("outer initial history is shorter than a feature window")
    if config.walk_forward.inner_initial_sessions != (3 * config.walk_forward.inner_test_sessions):
        raise ValueError("preregistered inner design requires three equal validation blocks")
    if config.features.fixed(config.execution) not in config.features.parameters(config.execution):
        raise ValueError("the fixed baseline must be inside the preregistered grid")
