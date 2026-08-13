from __future__ import annotations

import json
from pathlib import Path

import pytest

from news_sentiment_trading.config import load_config
from news_sentiment_trading.reporting import (
    _hypothesis_sentence,
    _validate_restricted_report_root,
    file_hash,
    generate_markdown_reports,
    generate_robustness_report,
    load_artifact_provenance,
    validate_primary_artifacts,
    validate_robustness_artifacts,
    write_artifact_provenance,
    write_json,
    write_primary_artifacts,
    write_robustness_artifacts,
)
from news_sentiment_trading.robustness import run_robustness
from news_sentiment_trading.walk_forward import run_walk_forward


def _assert_cost_basis(
    basis: dict, rate_bps: int, turnover_convention: str = "target_weight"
) -> None:
    assert basis["direction"] == "one_way"
    assert basis["rate_bps"] == rate_bps
    assert basis["turnover_convention"] == turnover_convention
    expected_charge = (
        "absolute target-weight change"
        if turnover_convention == "target_weight"
        else "absolute drift-adjusted risky-weight change"
    )
    assert basis["charged_on"] == expected_charge
    assert basis["applies_to"]


def _assert_one_way_cost_basis(payload: dict, rate_bps: int) -> None:
    _assert_cost_basis(payload["cost_basis"], rate_bps)


@pytest.mark.parametrize(
    ("predictive", "trading", "expected"),
    [
        (True, True, "Both preregistered hypotheses met"),
        (False, False, "Neither preregistered hypothesis met"),
        (True, False, "One preregistered hypothesis met"),
        (False, True, "One preregistered hypothesis met"),
    ],
)
def test_hypothesis_language_is_conditioned_on_validated_decisions(
    predictive: bool, trading: bool, expected: str
) -> None:
    conclusion = {
        "predictive_hypothesis_confirmed": predictive,
        "trading_hypothesis_confirmed": trading,
    }
    assert _hypothesis_sentence(conclusion).startswith(expected)


def test_report_templates_do_not_embed_viewed_outcome_claims() -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "news_sentiment_trading" / "reporting.py"
    text = source.read_text(encoding="utf-8").casefold()
    forbidden = (
        "no confirmatory" + " evidence",
        "four folds" + " lost money",
        "favorable" + " crisis slice",
        "selection" + " underperformed the fixed rule",
    )
    assert all(phrase not in text for phrase in forbidden)


def test_direct_report_generation_rejects_tracked_checkout_roots(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(RuntimeError, match="must remain under reports/artifacts"):
        _validate_restricted_report_root(root)
    _validate_restricted_report_root(tmp_path / "standalone-local-report")


def test_primary_artifacts_are_immutable_validated_and_report_driven(panel, tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "primary.toml")
    result = run_walk_forward(panel, config)
    artifact_dir = tmp_path / "artifacts"
    manifest = write_primary_artifacts(
        result,
        panel,
        config,
        artifact_dir,
        {ticker: ticker.lower() for ticker in config.universe.primary},
        "a" * 40,
        "b" * 64,
        "c" * 40,
        "d" * 64,
    )
    assert "timing_by_return_end.csv" in manifest
    assert "fixed_baseline_daily.csv" in manifest
    provenance_path = tmp_path / "primary-provenance.json"
    write_artifact_provenance(artifact_dir, provenance_path)
    provenance = load_artifact_provenance(provenance_path)
    validate_primary_artifacts(artifact_dir, provenance)
    candidate_scores = json.loads(
        (artifact_dir / "candidate_scores.json").read_text(encoding="utf-8")
    )
    for fold in candidate_scores:
        for candidate in fold:
            _assert_one_way_cost_basis(candidate, config.walk_forward.selection_cost_bps)
            assert "parameter-selection objective" in candidate["cost_basis"]["applies_to"]
    selection_header = (
        (artifact_dir / "parameters_by_date.csv").read_text(encoding="utf-8").splitlines()[0]
    )
    for field in ("rate_bps", "direction", "charged_on", "applies_to", "turnover_convention"):
        assert f"selection_cost_{field}" in selection_header
    primary_daily_files = (
        "portfolio_daily.csv",
        "fixed_baseline_daily.csv",
        "long_only_daily.csv",
        "market_neutral_daily.csv",
        "directional_daily.csv",
    )
    for name in primary_daily_files:
        header = (artifact_dir / name).read_text(encoding="utf-8").splitlines()[0]
        for field in (
            "rate_bps",
            "direction",
            "charged_on",
            "applies_to",
            "turnover_convention",
        ):
            assert f"evaluation_cost_{field}" in header
    run_metadata = json.loads((artifact_dir / "run_metadata.json").read_text(encoding="utf-8"))
    _assert_cost_basis(run_metadata["evaluation_cost_basis"], config.research.primary_cost_bps)
    _assert_cost_basis(run_metadata["selection_cost_basis"], config.walk_forward.selection_cost_bps)

    report_root = tmp_path / "report-root"
    generate_markdown_reports(artifact_dir, report_root, provenance)
    assert (report_root / "reports" / "tables" / "primary-summary.json").is_file()
    assert (report_root / "docs" / "EMPIRICAL_EVIDENCE_CARD.md").is_file()
    assert (report_root / "LOCAL_REPORT_INDEX.md").is_file()
    assert (report_root / "reports" / "figures" / "fold-performance.png").is_file()
    assert not (report_root / "reports" / "figures" / "primary-wealth.png").exists()
    public_metrics = json.loads(
        (report_root / "reports" / "tables" / "metrics.json").read_text(encoding="utf-8")
    )
    assert "active_spread" in public_metrics
    assert "relative_wealth" in public_metrics
    _assert_one_way_cost_basis(public_metrics, config.research.primary_cost_bps)
    for name in (
        "inference.json",
        "primary-summary.json",
        "crisis_decomposition.json",
        "parameter_stability.json",
    ):
        payload = json.loads(
            (report_root / "reports" / "tables" / name).read_text(encoding="utf-8")
        )
        _assert_one_way_cost_basis(payload, config.research.primary_cost_bps)
    public_folds = json.loads(
        (report_root / "reports" / "tables" / "per_fold_metrics.json").read_text(encoding="utf-8")
    )
    assert len(public_folds) == 6
    for fold in public_folds:
        _assert_one_way_cost_basis(fold, config.research.primary_cost_bps)
    public_selections = json.loads(
        (report_root / "reports" / "tables" / "selected_parameters.json").read_text(
            encoding="utf-8"
        )
    )
    for selection in public_selections:
        _assert_one_way_cost_basis(selection, config.walk_forward.selection_cost_bps)
    cost_scenarios = json.loads(
        (report_root / "reports" / "tables" / "cost_scenarios.json").read_text(encoding="utf-8")
    )
    for rate, scenario in cost_scenarios.items():
        _assert_one_way_cost_basis(scenario, int(rate))
    portfolio_variants = json.loads(
        (report_root / "reports" / "tables" / "portfolio_variants.json").read_text(encoding="utf-8")
    )
    for variant in portfolio_variants.values():
        _assert_one_way_cost_basis(variant, config.research.primary_cost_bps)
    corrected = json.loads(
        (report_root / "reports" / "tables" / "corrected_fb_jpm.json").read_text(encoding="utf-8")
    )
    for ticker in ("FB", "JPM"):
        for scenarios in corrected[ticker]["regimes"].values():
            for rate, scenario in scenarios.items():
                _assert_one_way_cost_basis(scenario, int(rate))
    figure_manifest = json.loads(
        (report_root / "reports" / "figures" / "manifest.json").read_text(encoding="utf-8")
    )
    _assert_one_way_cost_basis(figure_manifest[0], config.research.primary_cost_bps)
    assert "10 bps one-way cost" in figure_manifest[0]["title"]
    artifact_figure_manifest = json.loads(
        (artifact_dir / "generated_figure_manifest.json").read_text(encoding="utf-8")
    )
    _assert_one_way_cost_basis(artifact_figure_manifest[0], config.research.primary_cost_bps)
    assert "10 bps one-way cost" in artifact_figure_manifest[0]["title"]
    limitations = (report_root / "reports" / "limitations.md").read_text(encoding="utf-8")
    assert "{_pct" not in limitations
    assert "available sentiment scores" in limitations
    inference = json.loads((artifact_dir / "inference.json").read_text(encoding="utf-8"))
    assert "benchmark beta/intercept" in inference["cost_basis"]["applies_to"]
    assert inference["inference_specification"]["moving_block_bootstrap"] == {
        "active_return_seed": config.research.seed + 1,
        "block_length": 10,
        "method": "circular synchronized-date moving-block bootstrap",
        "samples": 10_000,
        "strategy_seed": config.research.seed,
    }
    assert set(inference["assumption_sensitivity"]["moving_block_by_length"]) == {
        "5",
        "10",
        "20",
    }
    evidence_text = (report_root / "docs" / "EMPIRICAL_EVIDENCE_CARD.md").read_text(
        encoding="utf-8"
    )
    index_text = (report_root / "LOCAL_REPORT_INDEX.md").read_text(encoding="utf-8")
    research_text = (report_root / "reports" / "research-report.md").read_text(encoding="utf-8")
    summary_text = (report_root / "reports" / "results-summary.md").read_text(encoding="utf-8")
    interpretation_text = (report_root / "docs" / "RESULTS_INTERPRETATION.md").read_text(
        encoding="utf-8"
    )
    assert "Restricted local empirical output" in evidence_text
    assert "not part of the distributable repository" in evidence_text
    assert "No matching robustness bundle was supplied" in evidence_text
    assert "article timestamps are unavailable" in evidence_text
    assert "Strategy gross / net total return (10 bps one-way cost)" in index_text
    assert "Mean daily active spread (10 bps one-way cost)" in index_text
    assert "Active HAC p / Holm p (10 bps one-way cost)" in index_text
    assert "Content-addressed provenance does not grant redistribution rights" in index_text
    assert "net total return at 10 bps one-way cost" in research_text
    assert "Walk-forward strategy net (10 bps one-way cost)" in summary_text
    assert "Mean daily active spread (10 bps one-way cost)" in summary_text
    assert "At 10 bps one-way cost, benchmark exposure was beta" in summary_text
    assert "primary net result at 10 bps one-way cost" in interpretation_text
    assert "Mean daily active spread at 10 bps one-way cost" in interpretation_text
    assert {path.name for path in (report_root / "docs").glob("*.md")} == {
        "EMPIRICAL_EVIDENCE_CARD.md",
        "REPRODUCIBILITY.md",
        "RESULTS_INTERPRETATION.md",
    }
    assert not (report_root / "README.md").exists()
    per_fold = json.loads((artifact_dir / "per_fold_metrics.json").read_text(encoding="utf-8"))
    assert per_fold[0]["strategy_net"] == per_fold[0]["strategy"]
    assert "strategy_gross" in per_fold[0]
    assert set(per_fold[0]["cost_and_exposure"]) == {
        "exposure_active_days",
        "gross_exposure_mean",
        "financing_balance_mean",
        "long_exposure_mean",
        "net_exposure_mean",
        "short_exposure_mean",
        "transaction_cost_total",
        "turnover_total",
        "unused_gross_capacity_mean",
    }
    attribution = json.loads(
        (artifact_dir / "per_asset_attribution.json").read_text(encoding="utf-8")
    )
    assert attribution["metric"] == "arithmetic_sum_of_daily_gross_weight_times_return"
    assert attribution["reconciliation"]["contribution_sum"] == pytest.approx(
        attribution["reconciliation"]["gross_return_arithmetic_sum"]
    )

    with pytest.raises(FileExistsError, match="new or empty"):
        write_primary_artifacts(
            result,
            panel,
            config,
            artifact_dir,
            {},
            "a" * 40,
            "b" * 64,
            "c" * 40,
            "d" * 64,
        )

    metrics = artifact_dir / "metrics.json"
    metrics.write_text(metrics.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_primary_artifacts(artifact_dir, provenance)
    manifest_path = artifact_dir / "ARTIFACT_MANIFEST.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["metrics.json"] = file_hash(metrics)
    write_json(manifest_path, manifest_payload)
    with pytest.raises(ValueError, match="external provenance"):
        validate_primary_artifacts(artifact_dir, provenance)


def test_robustness_artifacts_and_report_cover_full_matrix(panel, tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "primary.toml")
    primary = run_walk_forward(panel, config)
    result = run_robustness(panel, config, primary_result=primary)
    artifact_dir = tmp_path / "robustness"
    manifest = write_robustness_artifacts(
        result,
        artifact_dir,
        git_commit="a" * 40,
        preregistration_commit="b" * 40,
        configuration_hash="c" * 64,
        lock_hash="d" * 64,
        source_gate_hash="e" * 64,
        input_hashes={},
    )
    assert "robustness_summaries.csv" in manifest
    variant_prefixes = {name.split("/")[1] for name in manifest if name.startswith("variants/")}
    assert len(variant_prefixes) == len(result.summaries)
    assert any(name.endswith("/weights.csv") for name in manifest)
    assert any(name.endswith("/portfolio_daily.csv") for name in manifest)
    provenance_path = tmp_path / "robustness-provenance.json"
    write_artifact_provenance(artifact_dir, provenance_path)
    provenance = load_artifact_provenance(provenance_path)
    report_root = tmp_path / "report-root"
    generate_robustness_report(artifact_dir, report_root, provenance=provenance)
    assert (report_root / "reports" / "robustness-summary.md").is_file()
    assert (report_root / "reports" / "tables" / "robustness-summaries.csv").is_file()


def test_complete_report_composes_only_matching_primary_and_robustness(
    panel, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "primary.toml")
    primary = run_walk_forward(panel, config)
    primary_artifacts = tmp_path / "primary"
    write_primary_artifacts(
        primary,
        panel,
        config,
        primary_artifacts,
        {},
        "a" * 40,
        "b" * 64,
        "c" * 40,
        "d" * 64,
    )
    primary_provenance_path = tmp_path / "primary-provenance.json"
    write_artifact_provenance(primary_artifacts, primary_provenance_path)
    primary_provenance = load_artifact_provenance(primary_provenance_path)
    robustness = run_robustness(panel, config, primary_result=primary)
    robustness_artifacts = tmp_path / "robustness"
    write_robustness_artifacts(
        robustness,
        robustness_artifacts,
        git_commit="a" * 40,
        preregistration_commit="c" * 40,
        configuration_hash=config.digest(),
        lock_hash="b" * 64,
        source_gate_hash="d" * 64,
        input_hashes={},
    )
    robustness_provenance_path = tmp_path / "robustness-provenance.json"
    write_artifact_provenance(robustness_artifacts, robustness_provenance_path)
    robustness_provenance = load_artifact_provenance(robustness_provenance_path)
    report_root = tmp_path / "report-root"
    validate_robustness_artifacts(
        robustness_artifacts,
        robustness_provenance,
        primary_artifacts,
        primary_provenance,
    )
    variant_daily_paths = sorted((robustness_artifacts / "variants").glob("*/portfolio_daily.csv"))
    variant_metadata_paths = sorted(
        (robustness_artifacts / "variants").glob("*/variant_metadata.json")
    )
    variant_selection_paths = sorted(
        (robustness_artifacts / "variants").glob("*/selected_parameters.json")
    )
    assert len(variant_daily_paths) == 44
    assert len(variant_metadata_paths) == 44
    assert len(variant_selection_paths) == 43
    differing_selection_and_evaluation_rates = 0
    for metadata_path in variant_metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _assert_cost_basis(
            metadata["evaluation_cost_basis"],
            int(metadata["cost_bps"]),
            metadata["evaluation_cost_basis"]["turnover_convention"],
        )
        _assert_cost_basis(metadata["selection_cost_basis"], config.walk_forward.selection_cost_bps)
        if (
            metadata["evaluation_cost_basis"]["rate_bps"]
            != metadata["selection_cost_basis"]["rate_bps"]
        ):
            differing_selection_and_evaluation_rates += 1
    assert differing_selection_and_evaluation_rates == 15
    for daily_path in variant_daily_paths:
        header = daily_path.read_text(encoding="utf-8").splitlines()[0]
        for field in (
            "rate_bps",
            "direction",
            "charged_on",
            "applies_to",
            "turnover_convention",
        ):
            assert f"evaluation_cost_{field}" in header
    for selection_path in variant_selection_paths:
        selections = json.loads(selection_path.read_text(encoding="utf-8"))
        assert selections
        for selection in selections:
            _assert_one_way_cost_basis(selection, config.walk_forward.selection_cost_bps)
    generate_markdown_reports(
        primary_artifacts,
        report_root,
        primary_provenance,
        robustness_artifact_dir=robustness_artifacts,
        robustness_provenance=robustness_provenance,
    )
    report = (report_root / "reports" / "research-report.md").read_text(encoding="utf-8")
    evidence_text = (report_root / "docs" / "EMPIRICAL_EVIDENCE_CARD.md").read_text(
        encoding="utf-8"
    )
    index_text = (report_root / "LOCAL_REPORT_INDEX.md").read_text(encoding="utf-8")
    robustness_text = (report_root / "reports" / "robustness-summary.md").read_text(
        encoding="utf-8"
    )
    robustness_summary_header = (
        (report_root / "reports" / "tables" / "robustness-summaries.csv")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    robustness_selection_header = (
        (report_root / "reports" / "tables" / "robustness-selections.csv")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert report.count("## Robustness evidence") == 1
    assert "At 10 bps one-way cost, leave-one-asset-out returns ranged" in report
    assert "leave-one-asset-out returns ranged" in evidence_text
    assert "return-end-based crisis component" in evidence_text
    assert "pre-crisis returns were" in evidence_text
    assert "The table reports the strategy and corresponding benchmark" in robustness_text
    assert "one-session forward fill returned" in report
    assert "Period components are descriptive" in robustness_text
    assert "| Portfolio | One-way cost (bps) |" in robustness_text
    assert "At 10 bps one-way cost, full-sample beta was" in robustness_text
    assert "Post-v1.0 turnover and benchmark accounting" in robustness_text
    assert "cost_bps,cost_direction" in robustness_summary_header
    assert "cost_charged_on,cost_applies_to" in robustness_summary_header
    assert "selection_cost_bps,selection_cost_direction" in robustness_selection_header
    assert "selection_cost_charged_on,selection_cost_applies_to" in robustness_selection_header
    assert "Crisis and pre-crisis rows are reported separately" in index_text
    assert "bounded-forward-fill, fixed-policy" in report
    assert "selected-minus-fixed difference" in report
    assert "These descriptive variants do not alter" in report
    assert "At 10 bps one-way cost, full-sample benchmark beta was" in report
    assert "not part of the distributable repository" in evidence_text

    metadata_path = robustness_artifacts / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["git_commit"] = "d" * 40
    write_json(metadata_path, metadata)
    manifest_path = robustness_artifacts / "ARTIFACT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_metadata.json"] = file_hash(metadata_path)
    write_json(manifest_path, manifest)
    tampered_provenance_path = tmp_path / "tampered-robustness-provenance.json"
    write_artifact_provenance(robustness_artifacts, tampered_provenance_path)
    tampered_provenance = load_artifact_provenance(tampered_provenance_path)
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_robustness_artifacts(
            robustness_artifacts,
            tampered_provenance,
            primary_artifacts,
            primary_provenance,
        )
