#!/usr/bin/env python3
"""Extract in-text fragments from the locally available manuscript tables.

No model fitting, effect-size calculation, or inference from figures. Retain
the tables' reported precision. Adjacent-rank results use only the separate
ANOVA fragments from repeated-measures tables. Rank-change results use validated fitted CSVs and model
diagnostics. Temporal results reproduce the observation-timestep tables.
"""
import argparse
import csv
import hashlib
import math
from pathlib import Path
import re
import uuid
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from project_paths import REPO_ROOT as ROOT, STATS_ROOT as DEFAULT_STATS_ROOT, TABLE_DIR, INTEXT_DIR, LEGACY_STATS_DIR
FINAL_EXPORTS_ONLY = True


def concise_caption_matches(text, kind):
    """Accept the concise captions; shared inference conventions now live in Methods.

    This only recognizes caption wording. All existing row, coefficient, and
    saved-model validation remains in place.
    """
    phrases = {
        "adjacent": ("Adjacent path-rank comparisons", "planning condition; leaf and non-leaf probes",
                     "lowest numerical rank among complete paths containing it",
                     "participants observed at both ranks", "pooled within participant and rank",
                     "larger numerical rank minus accuracy at the smaller rank"),
        "best_branch": ("Best-versus-worst branch presentation timing", "middle-branch leaf probes (positions 6--8)",
                        "outside the middle branch and in different branches",
                        "Worst-branch-first is the reference condition", "higher recall when the best branch appears first"),
        "temporal": ("Observation-timestep effects on planning-condition recall accuracy",
                     "adjusting for leaf status (non-leaf reference)", "timestep indexes task order"),
        "parameters": ("Separate Gaussian BAM regressions", r"fitted $\lambda$, fitted $\alpha$, and their interaction",
                       "parameters and outcomes remain in their original units",
                       "Main effects are conditional on the other parameter equaling zero"),
        "memory": ("Associations between participant-specific contextual memory effects",
                   "entered together into an ordinary least-squares regression predicting mean normalized reward",
                   "one equally weighted row per participant", "Contextual-effect predictors are standardized"),
        "pairwise": ("Cross-experiment pairwise comparisons", "larger minus smaller tree",
                     "pooled residual variance from a one-way ANOVA", "confidence intervals and $p$ values are unadjusted"),
        "exposure_memory": ("ordinary least-squares regressions predict participant mean normalized reward and recall accuracy",
                            r"standardized exposure-fitted memory spread ($\sigma$)",
                            "one equally weighted mean", "unadjusted two-sided $p$ values"),
        "exposure_lambda": (r"ordinary least-squares regressions predict raw planning-fitted $\lambda$",
                            r"from standardized exposure-fitted memory spread ($\sigma$),",
                            "one observation", "unadjusted two-sided $p$ values"),
        "context": ("are fit jointly across planning and exposure", "planning as the reference condition",
                    "exposure-minus-planning slope differences", "variants are fit to planning only"),
        "branch": ("Competing-branch reward", "earlier-minus-later difference in the context-reward slope",
                   "controlling for observation timestep", "maximizing branches span both timing conditions are excluded"),
        "tree_order": ("Presentation-order and tree-structure effects", "deep $=-0.5$, wide $=+0.5$",
                       "BFS $=-0.5$, DFS $=+0.5$", "DFS minus BFS averaged across tree structures",
                       "wide minus deep averaged across orders", "DFS--BFS difference in wide versus deep trees"),
        "rank": ("Rank-change effects", "improved $=1$ and worsened $=0$; unchanged ranks are excluded",
                 "Final rank is numeric and unscaled", "deep/BFS $=-0.5$ and wide/DFS $=+0.5$",
                 "rank-improvement effects and interactions hold final rank constant"),
    }
    return all(phrase in text for phrase in phrases[kind])

GROUPS = {"Participants": "participants", "Sequential model": "model"}
EFFECTS = {
    "probed": ("Node reward", "node_reward_z"),
    "sibling": ("Sibling reward", "sibling_reward_z"),
    "parent": ("Parent reward", "parent_reward_z"),
    "aunt": ("Aunt reward", "aunt_reward_z"),
}
NUMBER = r"-?\d+\.\d{3}"


def display_path(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def read_table(path, groups=GROUPS, outcome_column=False):
    text = path.read_text()
    return parse_table(text, path, groups, outcome_column)


def parse_table(text, path, groups=GROUPS, outcome_column=False):
    rows = {}
    for line in text.splitlines():
        if not any(line.startswith(group + " & ") for group in groups):
            continue
        cells = [cell.strip() for cell in line.removesuffix(r"\\").split(" & ")]
        if outcome_column and len(cells) == 7:
            cells = [cells[0], cells[1] + ": " + cells[2], *cells[3:]]
        if len(cells) != 6:
            raise ValueError(f"Unexpected table schema in {path}: {line}")
        group, effect, estimate, interval, df, p = cells
        match = re.fullmatch(rf"\[({NUMBER}), ({NUMBER})\]", interval)
        if not re.fullmatch(NUMBER, estimate) or not match:
            raise ValueError(f"Invalid coefficient/CI in {path}: {line}")
        low, high = match.groups()
        if not float(low) <= float(estimate) <= float(high):
            raise ValueError(f"Coefficient outside confidence interval in {path}")
        if p == "$< .001$":
            p_text = "$p<.001$"
        elif re.fullmatch(r"\$(?:\.\d{3}|1\.000)\$", p):
            value = p.strip("$")
            if float(value) < .001:
                raise ValueError(f"Rounded-zero p value cannot be safely exported: {path}")
            p_text = f"$p={value}$"
        else:
            raise ValueError(f"Unsupported p format in {path}: {p}")
        key = (group, effect)
        if key in rows:
            raise ValueError(f"Duplicate effect {key} in {path}")
        rows[key] = dict(estimate=estimate, ci_low=low, ci_high=high,
                         p_reported=p, text=f"$b={estimate}$, 95\\% CI $[{low},{high}]$, {p_text}")
    return text, rows


def read_adjacent_anova_table(path):
    text = path.read_text()
    for required in ("Classical repeated-measures ANOVAs", "Planning-condition leaf and non-leaf probes",
                     "lowest numerical rank among complete paths containing it",
                     "weighting participants equally", "Reported $p$ values are unadjusted",
                     "larger numerical rank minus accuracy at the smaller rank", "paired 95\\% t confidence intervals"):
        if required not in text and not concise_caption_matches(text, "adjacent"):
            raise ValueError(f"Cannot verify adjacent ANOVA table {path}: {required}")
    rows = {}
    for line in text.splitlines():
        if not any(line.startswith(group + " & ") for group in GROUPS):
            continue
        cells = [c.strip() for c in line.removesuffix(r"\\").split(" & ")]
        if len(cells) != 9:
            raise ValueError(f"Invalid ANOVA columns: {path}")
        group, ranks, n, difference, interval, f, df, eta, p = cells
        ci = re.fullmatch(rf"\[({NUMBER}), ({NUMBER})\]", interval)
        if (not re.fullmatch(NUMBER, difference) or not ci or
                not float(ci[1]) <= float(difference) <= float(ci[2])):
            raise ValueError(f"Invalid paired difference/CI: {path}: {line}")
        match = re.fullmatch(r"(\d+) vs\.\\ (\d+)", ranks)
        dfs = re.fullmatch(r"1, (\d+)", df)
        if (not match or not dfs or int(match[2]) != int(match[1]) + 1 or
                not n.isdigit() or int(dfs[1]) != int(n) - 1 or
                not re.fullmatch(NUMBER, f) or float(f) < 0 or
                not re.fullmatch(NUMBER, eta) or not 0 <= float(eta) <= 1):
            raise ValueError(f"Invalid or non-estimable ANOVA row: {path}: {line}")
        if p == "$< .001$":
            p_text = "$p<.001$"
        elif re.fullmatch(r"\$(?:\.\d{3}|1\.000)\$", p) and float(p.strip("$")) >= .001:
            p_text = f"$p={p.strip('$')}$"
        else:
            raise ValueError(f"Invalid ANOVA p value: {path}: {p}")
        key = (group, int(match[1]))
        if key in rows:
            raise ValueError(f"Duplicate ANOVA contrast: {path}: {key}")
        rows[key] = dict(estimate=difference, ci_low=ci[1], ci_high=ci[2], p_reported=p,
            text=f"$\\Delta={difference}$, 95\\% CI $[{ci[1]},{ci[2]}]$, $F(1,{dfs[1]})={f}$, $\\eta_p^2={eta}$, {p_text}")
    expected = {(g, rank) for g in GROUPS for rank in range(1, 9 if path.name.startswith("exp2_") else 4)}
    if set(rows) != expected:
        raise ValueError(f"Expected all adjacent pairs for participants and model: {path}")
    return rows


def read_best_branch_timing_table(path):
    caption, rows = read_table(path, ("Participants", "Sequential model"))
    for required in ("Binomial-logit glmer fits", "independent participant intercept and timing slope",
                     "worst branch first is the reference", "unadjusted two-sided"):
        if required not in caption and not concise_caption_matches(caption, "best_branch"):
            raise ValueError(f"Cannot verify best-branch timing model in {path}: {required}")
    return rows


def read_temporal_table(path):
    caption, rows = read_table(path)
    for required in ("planning condition", "binomial-logit glmer fits (Laplace ML)",
                     "uncorrelated participant random intercept, leaf-status slope, and timestep slope",
                     "Timestep is standardized", "Leaf status compares leaves against non-leaves (reference)",
                     "unstandardized log-odds",
                     "normal-Wald", "unadjusted Wald z-test"):
        if required not in caption and not concise_caption_matches(caption, "temporal"):
            raise ValueError(f"Cannot verify temporal model/coding in {path}: {required}")
    for group in GROUPS:
        for label in ("Observation timestep",):
            if (group, label) not in rows:
                raise ValueError(f"Missing temporal effect in {path}: {group}, {label}")
    return rows


def read_path_rank_table(path, experiment):
    text = path.read_text()
    for required in ("larger numerical rank minus", "binomial-logit glmer fits (Laplace ML)",
                     "Rank comparison & Participants & Sequential model"):
        if required not in text:
            raise ValueError(f"Cannot verify adjacent-rank table coding in {path}: {required}")
    rows, seen = {}, []
    pattern = (rf"\$(?:\\beta|b) = ({NUMBER})\$; 95\\% CI "
               rf"\[({NUMBER}), ({NUMBER})\]; df = (?:---|\d+(?:\.\d+)?); "
               r"\$p\$ (?:= )?(\$(?:< \.001|\.\d{3}|1\.000)\$)")
    for line in text.splitlines():
        if not line.startswith("Rank ") or line.startswith("Rank comparison"):
            continue
        cells = [cell.strip() for cell in line.removesuffix(r"\\").split(" & ")]
        if len(cells) != 3:
            raise ValueError(f"Invalid adjacent-rank row: {path}: {line}")
        label = re.fullmatch(r"Rank (\d+) vs\.\\ (\d+)", cells[0])
        if not label:
            raise ValueError(f"Invalid adjacent-rank label: {path}: {cells[0]}")
        first, second = map(int, label.groups())
        if second != first + 1 or first in seen:
            raise ValueError(f"Non-adjacent or duplicate rank comparison: {path}: {cells[0]}")
        seen.append(first)
        for group, cell in zip(GROUPS, cells[1:]):
            match = re.fullmatch(pattern, cell)
            if not match:
                raise ValueError(f"Invalid adjacent-rank coefficient: {path}: {cell}")
            estimate, low, high, p = match.groups()
            if not float(low) <= float(estimate) <= float(high):
                raise ValueError(f"Coefficient outside CI: {path}: {cell}")
            if p == "$< .001$":
                p_text = "$p<.001$"
            else:
                if float(p.strip("$")) < .001:
                    raise ValueError(f"Rounded-zero p value cannot be safely exported: {path}")
                p_text = f"$p={p.strip('$')}$"
            rows[(group, first)] = dict(estimate=estimate, ci_low=low, ci_high=high,
                p_reported=p, text=f"$b={estimate}$, 95\\% CI $[{low},{high}]$, {p_text}")
    expected = list(range(1, 4 if experiment == "exp1" else 9))
    if sorted(seen) != expected:
        raise ValueError(f"Missing adjacent-rank comparisons in {path}: expected {expected}, got {seen}")
    return rows


def read_rank_change(directory):
    """Reject obsolete formulas/codings rather than silently reuse old fits."""
    rewards = ["node_reward_z", "sibling_reward_z", "parent_reward_z",
               "aunt_reward_z", "max_child_reward_z", "avg_niece_reward_z"]
    contrasts = ["rank_conditionbetter", "observation_step_z"] + rewards + [
        "observation_step_z:" + reward for reward in rewards]
    expected = ("accuracy ~ rank_condition + observation_step_z * (" +
                " + ".join(rewards) + ') + (1 | participant)' +
                "".join(f' + (0 + .random_{i} | participant)' for i in range(1, 15)))
    paths = {name: directory / name for name in
             ("coefficients.csv", "model.txt", "random_design.csv", "scaling.csv")}
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"Required rank-change input is missing: {path}")
    diagnostics = dict(line.split(": ", 1) for line in paths["model.txt"].read_text().splitlines() if ": " in line)
    if re.sub(r"\s+", "", diagnostics.get("Formula", "")) != re.sub(r"\s+", "", expected):
        raise ValueError(f"Rank-change formula is not the current timestep-by-six-rewards model: {directory}")
    if diagnostics.get("Estimator") != "lme4::glmer; binomial-logit GLMM; maximum likelihood, Laplace approximation (nAGQ=1)" or \
            diagnostics.get("Family") != "binomial Link: logit":
        raise ValueError(f"Unexpected rank-change estimator/family: {directory}")
    with paths["random_design.csv"].open() as handle:
        design = list(csv.DictReader(handle))
    if design != [dict(random_column=f".random_{i}", fixed_contrast=term)
                  for i, term in enumerate(contrasts, 1)]:
        raise ValueError(f"Unexpected rank-change contrasts/random slopes: {directory}")
    with paths["scaling.csv"].open() as handle:
        scaling = {row["variable"]: row for row in csv.DictReader(handle)}
    for variable in ["observation_step"] + [reward.removesuffix("_z") for reward in rewards]:
        row = scaling.get(variable)
        if not row or not all(math.isfinite(float(row[key])) for key in ("mean", "sd")) or float(row["sd"]) <= 0:
            raise ValueError(f"Missing/invalid standardization for {variable}: {directory}")
    with paths["coefficients.csv"].open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["term"] == "rank_conditionbetter"]
    if len(rows) != 1:
        raise ValueError(f"Expected one rank_conditionbetter coefficient: {directory}")
    values = {key: float(rows[0][key]) for key in ("beta", "ci_low", "ci_high", "p", "se", "z")}
    if not all(math.isfinite(value) for value in values.values()) or not 0 <= values["p"] <= 1 or \
            not values["ci_low"] <= values["beta"] <= values["ci_high"] or values["se"] <= 0:
        raise ValueError(f"Invalid rank-change coefficient: {directory}")
    estimate, low, high = (f"{values[key]:.3f}" for key in ("beta", "ci_low", "ci_high"))
    p_text = "$p<.001$" if values["p"] < .001 else f"$p={values['p']:.3f}$".replace("=0.", "=.")
    row = dict(estimate=estimate, ci_low=low, ci_high=high, p_reported=rows[0]["p"],
               text=f"$b={estimate}$, 95\\% CI $[{low},{high}]$, {p_text}")
    audit = {key: display_path(paths[name]) for key, name in
             (("model_diagnostics", "model.txt"), ("random_design", "random_design.csv"), ("scaling", "scaling.csv"))}
    for key in list(audit):
        audit[key + "_sha256"] = hashlib.sha256((ROOT / audit[key]).read_bytes()).hexdigest()
    audit.update(singular_fit=diagnostics.get("Singular", ""), convergence=diagnostics.get("Convergence", ""))
    return row, audit


def read_nll_comparisons(stats_root, experiment):
    """Read saved paired tests; never compute new tests or substitute unadjusted p."""
    results = []
    for variant in ("simultaneous", "independent"):
        directory = stats_root / experiment / variant / "nll_vs_sequential"
        path = directory / "comparisons.csv"
        model_path = directory / "model.txt"
        if not path.exists() or not model_path.exists():
            raise ValueError(f"NLL statistics missing: {directory}; run --tests nll and download the results")
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError(f"Expected one NLL comparison in {path}")
        row = rows[0]
        expected = dict(term=f"{variant}_minus_sequential", variant=variant, reference="sequential",
                        alternative="variant NLL > sequential NLL")
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Invalid NLL contrast/direction: {path}")
        keys = ("beta", "ci_low", "ci_high", "p", "p_holm", "t", "df", "participants",
                "mean_full_nll", "mean_variant_nll", "reward_nll_scale")
        values = {key: float(row[key]) for key in keys}
        if (not all(math.isfinite(v) for v in values.values()) or
                not values["ci_low"] <= values["beta"] <= values["ci_high"] or
                not 0 <= values["p"] <= values["p_holm"] <= 1 or
                values["participants"] < 2 or not values["participants"].is_integer() or
                values["df"] != values["participants"] - 1 or
                not math.isclose(values["reward_nll_scale"], .5, abs_tol=1e-8) or
                not math.isclose(values["beta"], values["mean_variant_nll"] - values["mean_full_nll"],
                                 rel_tol=1e-8, abs_tol=1e-8)):
            raise ValueError(f"Invalid NLL values: {path}")
        model = model_path.read_text()
        for required in ("Paired t test", "Difference = variant minus full sequential model",
                         "one-sided greater-than-zero", "adjusts BOTH variant comparisons",
                         "TWO-SIDED 95% t interval", "in-sample selected-fit"):
            if required not in model:
                raise ValueError(f"Cannot verify NLL inference in {model_path}: {required}")
        estimate, low, high = (f"{values[key]:.3f}" for key in ("beta", "ci_low", "ci_high"))
        p = values["p"]
        p_text = (r"$p<.001$" if p < .001 else
                  r"$p=" + f"{p:.3f}".removeprefix("0") + "$")
        result = dict(estimate=estimate, ci_low=low, ci_high=high, p_reported=p_text,
                      text=rf"$\Delta\mathrm{{NLL}}={estimate}$, 95\% CI $[{low},{high}]$, {p_text}")
        audit = dict(model_diagnostics=display_path(model_path),
                     model_diagnostics_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                     p_unadjusted=values["p"], p_holm=values["p_holm"],
                     t=values["t"], df=values["df"], participants=values["participants"])
        results.append((path, variant, result, audit, values))
    first, second = (result[-1] for result in results)
    if (first["participants"] != second["participants"] or
            not math.isclose(first["mean_full_nll"], second["mean_full_nll"], rel_tol=1e-8, abs_tol=1e-8)):
        raise ValueError("NLL comparisons have inconsistent participant counts or full-model means")
    # Validate the recorded two-test Holm correction without replacing it.
    ordered = sorted((first, second), key=lambda value: value["p"])
    expected_holm = (min(1, 2 * ordered[0]["p"]), min(1, max(2 * ordered[0]["p"], ordered[1]["p"])))
    if any(not math.isclose(value["p_holm"], expected, abs_tol=1e-12, rel_tol=1e-10)
           for value, expected in zip(ordered, expected_holm)):
        raise ValueError("NLL comparisons have inconsistent Holm-adjusted p values")
    return results


def performance_effects(tables, output, experiments):
    """Yield validated table-backed fragments; never use manuscript numbers."""
    path = tables / "fitted_parameters_performance.tex"
    text = path.read_text()
    for required in (r"fitted $\lambda$, fitted $\alpha$, and their interaction",
                     "Parameters and outcomes remain in their original units"):
        if required not in text and not concise_caption_matches(text, "parameters"):
            raise ValueError(f"Cannot verify parameter-performance model in {path}: {required}")
    lines = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.removesuffix(r"\\").split(" & ")]
        if cells[0] not in ("1", "2", "3"):
            continue
        if len(cells) != 7:
            raise ValueError(f"Invalid parameter-performance row: {path}: {line}")
        exp, outcome, term, *values = cells
        if term == "Overall model":
            continue
        lines.append(" & ".join([exp, outcome + ": " + term, *values]) + r" \\")
    _, rows = parse_table("\n".join(lines), path, ("1", "2", "3"))
    for experiment in experiments:
        for outcome, label in (("norm_reward", "Normalized reward"), ("accuracy", "Recall accuracy")):
            yield (path, rows, experiment[-1], label + r": Fitted $\lambda$", "fitted_lambda",
                   output / experiment / f"parameter_performance_{outcome}_lambda.tex",
                   f"Gaussian BAM: {outcome} ~ fitted_lambda * fitted_alpha; participant means",
                   "lambda slope at alpha=0", "raw parameters and participant-mean outcome", "b")

    for experiment in (e for e in experiments if e in ("exp1", "exp2") and not FINAL_EXPORTS_ONLY):
        path = tables / f"{experiment}_memory_allocation_performance.tex"
        text = path.read_text()
        for required in ("These regressions jointly include probed, sibling, parent, and aunt reward predictors",
                         "Continuous predictors were standardized; outcomes were not"):
            if required not in text and not concise_caption_matches(text, "memory"):
                raise ValueError(f"Cannot verify memory-performance model in {path}: {required}")
        lines = []
        for line in text.splitlines():
            if not any(line.startswith(g + " & ") for g in GROUPS):
                continue
            cells = [c.strip() for c in line.removesuffix(r"\\").split(" & ")]
            if len(cells) != 6:
                raise ValueError(f"Invalid memory-performance row: {path}: {line}")
            if cells[1] == "Overall model":
                continue
            if cells[2] == "Not estimable":
                if cells[3:] != ["---"] * 3:
                    raise ValueError(f"Invalid not-estimable row: {path}: {line}")
            else:
                lines.append(line)
        _, rows = parse_table("\n".join(lines), path)
        for group, suffix in GROUPS.items():
            for short, label in (("probed", "Node"), ("sibling", "Sibling"), ("parent", "Parent"), ("aunt", "Aunt")):
                effect = label + "-reward coefficient"
                if (group, effect) not in rows:
                    # Only explicit not-estimable rows may omit a fragment.
                    if f"{group} & {effect} & Not estimable" not in text:
                        raise ValueError(f"Missing memory-performance coefficient: {path}: {group}, {effect}")
                    continue
                yield (path, rows, group, effect, label.lower() + "_reward_coefficient_z",
                       output / experiment / f"memory_performance_{short}_{suffix}.tex",
                       "Joint Gaussian regression of normalized reward on varying memory-allocation coefficients",
                       "conditional association holding other included coefficients constant",
                       "predictors standardized; normalized reward unscaled", "b")

    if set(experiments) == {"exp1", "exp2", "exp3"}:
        path = tables / "cross_experiment_planning_accuracy_pairwise.tex"
        text = path.read_text()
        for required in ("Positive differences indicate higher accuracy in the larger tree",
                         "one-way ANOVA pooled residual variance"):
            if required not in text and not concise_caption_matches(text, "pairwise"):
                raise ValueError(f"Cannot verify set-size contrasts in {path}: {required}")
        lines = []
        for line in text.splitlines():
            if not any(line.startswith(g + " & ") for g in GROUPS):
                continue
            cells = [c.strip() for c in line.removesuffix(r"\\").split(" & ")]
            if len(cells) not in (6, 7):
                raise ValueError(f"Invalid set-size row: {path}: {line}")
            # Accept legacy tables too; always use the unadjusted p column.
            lines.append(" & ".join(cells[:6]) + r" \\")
        _, rows = parse_table("\n".join(lines), path)
        for group, suffix in GROUPS.items():
            for short, label in (("exp2_minus_exp1", "13 minus 7 nodes"), ("exp3_minus_exp2", "31 minus 13 nodes")):
                row = rows[(group, label)]
                row["text"] = row["text"].replace("$b=", r"$\Delta=")
                yield (path, rows, group, label, short,
                       output / "cross_experiment" / f"planning_accuracy_{short}_{suffix}.tex",
                       "Pairwise contrasts using one-way ANOVA pooled residual variance",
                       label, "participant-mean accuracy differences; unadjusted CI and p", r"\Delta")


def exposure_memory_effects(tables, output, experiments):
    path = tables / "exposure_memory_performance.tex"
    text = path.read_text()
    for required in ("ordinary least-squares regressions", "planning condition",
                     "memory-distribution standard deviation", "Sigma is standardized",
                     "unadjusted two-sided t-test"):
        if required not in text and not concise_caption_matches(text, "exposure_memory"):
            raise ValueError(f"Cannot verify exposure-memory model in {path}: {required}")
    lines = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.removesuffix(r"\\").split(" & ")]
        if cells[0] not in ("1", "2"):
            continue
        if len(cells) != 8:
            raise ValueError(f"Invalid exposure-memory row: {line}")
        exp, outcome, predictor, n, estimate, ci, df, p = cells
        if not n.isdigit() or int(n) < 5 or not math.isfinite(float(df)) or float(df) <= 0:
            raise ValueError(f"Invalid exposure-memory sample/df: {line}")
        lines.append(" & ".join([exp, outcome + ": " + predictor, estimate, ci, df, p]) + r" \\")
    _, rows = parse_table("\n".join(lines), path, ("1", "2"))
    for experiment in (e for e in experiments if e in ("exp1", "exp2")):
        for outcome, label in (("norm_reward", "Normalized reward"), ("accuracy", "Recall accuracy")):
            for short, predictor in (("sigma", r"Memory spread ($\sigma$)"),):
                key = (experiment[-1], label + ": " + predictor)
                if key not in rows:
                    raise ValueError(f"Missing exposure-memory coefficient: {key}")
                row = rows[key]
                p_text = row["text"].split(", ")[-1]
                for suffix, fragment in (("", row["text"]), ("_bp", f"$b={row['estimate']}$, {p_text}"), ("_p", p_text)):
                    selected = {key: dict(row, text=fragment)}
                    yield (path, selected, key[0], key[1], short + "_z",
                           output / experiment / f"exposure_memory_{outcome}_{short}{suffix}.tex",
                           f"OLS: participant mean {outcome} ~ scale(sigma)",
                           "planning performance; exposure-fitted parameters; sigma only; no guessing covariate",
                           "predictors z-scored; outcome unstandardized; unadjusted two-sided p", "b")


def exposure_lambda_effects(tables, output, experiments):
    path = tables / "exposure_memory_lambda.tex"
    text = path.read_text()
    # Older generated captions explicitly said "as the sole predictor";
    # line wrapping also must not invalidate an otherwise identical model.
    caption_text = " ".join(text.split())
    sigma_only = re.search(
        r"from standardized sigma(?: as the sole predictor)?"
        r"(?:, without an interaction or log transformation)?\.", caption_text)
    for required in ("ordinary least-squares regressions", "raw planning-fitted",
                     "from standardized sigma.",
                     "unadjusted two-sided t-test"):
        verified = (bool(sigma_only) if required == "from standardized sigma."
                    else required in caption_text)
        if not verified and not concise_caption_matches(caption_text, "exposure_lambda"):
            raise ValueError(f"Cannot verify exposure-lambda model in {path}: {required}")
    lines = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.removesuffix(r"\\").split(" & ")]
        if cells[0] not in ("1", "2"):
            continue
        if len(cells) != 7:
            raise ValueError(f"Invalid exposure-lambda row: {line}")
        exp, predictor, n, estimate, ci, df, p = cells
        if not n.isdigit() or int(n) < 5 or not math.isfinite(float(df)) or float(df) <= 0:
            raise ValueError(f"Invalid exposure-lambda sample/df: {line}")
        lines.append(" & ".join([exp, predictor, estimate, ci, df, p]) + r" \\")
    _, rows = parse_table("\n".join(lines), path, ("1", "2"))
    for experiment in (e for e in experiments if e in ("exp1", "exp2")):
        for short, label in (("sigma", r"Memory spread ($\sigma$)"),):
            key = (experiment[-1], label)
            if key not in rows:
                raise ValueError(f"Missing exposure-lambda coefficient: {key}")
            row = rows[key]
            p_text = row["text"].split(", ")[-1]
            for suffix, fragment in (("", row["text"]), ("_bp", f"$b={row['estimate']}$, {p_text}"), ("_p", p_text)):
                yield (path, {key: dict(row, text=fragment)}, key[0], label, short + "_z",
                       output / experiment / f"exposure_memory_lambda_{short}{suffix}.tex",
                       "OLS: fitted_lambda ~ scale(sigma)",
                       "full sequential planning lambda; sigma only; no guessing covariate",
                       "predictors z-scored; raw lambda; unadjusted two-sided p", "b")


def export_quadratic_performance(tables, output, experiments):
    path = tables / "quadratic_node_reward_performance.tex"
    text = path.read_text()
    for required in ("Participant quadratic probed-reward effects", "sole predictor", "sibling reward squared"):
        if required not in text:
            raise ValueError(f"Cannot verify quadratic performance model: {path}")
    lines = []
    for line in text.splitlines():
        if not line.startswith(("1 & ", "2 & ")):
            continue
        cells = [c.strip() for c in line.removesuffix(r"\\").split(" & ")]
        if len(cells) != 7 or not cells[2].isdigit() or int(cells[2]) < 5:
            raise ValueError(f"Invalid quadratic performance row: {line}")
        lines.append(" & ".join(cells[:2] + cells[3:]) + r" \\")
    _, rows = parse_table("\n".join(lines), path, ("1", "2"))
    for experiment in (e for e in experiments if e in ("exp1", "exp2")):
        row = rows[(experiment[-1], "Quadratic probed reward")]
        target = output / experiment / "quadratic_node_reward_performance.tex"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(row["text"].rstrip())
        print("Generated:", target)


def export_engagement(tables, output, experiments, legacy_stats_dir=None):
    """Copy participant-weighted mean/SD values from the engagement tables."""
    scopes = list(experiments)
    if FINAL_EXPORTS_ONLY and "exp2" in experiments:
        scopes.append("exp2_40n")
    for experiment in scopes:
        path = tables / f"{experiment}_engagement.tex"
        if not path.exists():
            if FINAL_EXPORTS_ONLY:
                raise FileNotFoundError(path)
            continue
        sources = ("participants",) if experiment == "exp2_40n" else (
            ("participants", "model") if experiment == "exp3" else
            ("participants", "model", "simultaneous", "independent"))
        expected = {(measure, condition) for measure, condition in
                    (("Normalized reward", "Planning"), ("Recall accuracy", "Planning"))}
        if experiment != "exp3":
            expected.add(("Recall accuracy", "Exposure"))
        seen = set()
        pending = {}
        for line in path.read_text().splitlines():
            if not line.startswith(("Normalized reward &", "Recall accuracy &")):
                continue
            cells = [c.strip() for c in line.removesuffix(r"\\").split(" & ")]
            if len(cells) != 2 + len(sources) or tuple(cells[:2]) in seen:
                raise ValueError(f"Invalid engagement row: {path}: {line}")
            seen.add(tuple(cells[:2]))
            stem = "norm_reward" if cells[0] == "Normalized reward" else "accuracy_" + cells[1].lower()
            for source, value in zip(sources, cells[2:]):
                if value == "---" and cells[1] == "Exposure" and source in ("simultaneous", "independent"):
                    continue
                match = re.fullmatch(rf"({NUMBER}) \(({NUMBER})\)", value)
                if not match or float(match[2]) < 0:
                    raise ValueError(f"Invalid engagement mean/SD: {path}: {value}")
                for statistic, text in zip(("mean", "sd"), match.groups()):
                    pending[output / experiment / f"{stem}_{statistic}_{source}.tex"] = text
                    if experiment == "exp2_40n" and legacy_stats_dir is not None:
                        pending[legacy_stats_dir / "exp2_40n_beh" / f"{stem}_{statistic}.tex"] = text
        if seen != expected:
            raise ValueError(f"Missing/unexpected engagement rows: {path}")
        for destination, text in pending.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text)


def export(tables, output, stats_root=DEFAULT_STATS_ROOT, experiments=("exp1", "exp2", "exp3"), performance_only=False,
           exposure_memory_only=False):
    fragments, records, unavailable = {}, [], []
    if (tables / "quadratic_node_reward_performance.tex").exists():
        export_quadratic_performance(tables, output, experiments)

    def effect(path, rows, group, label, term, output_path, model, contrast,
               standardization="predictors standardized; response unscaled (not fully standardized beta)", symbol="b"):
        row = rows[(group, label)]
        records.append(dict(
            output=display_path(output_path), source=display_path(path),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            group=group, model=model, term=term, contrast=contrast,
            coefficient_symbol=symbol, standardization=standardization,
            estimate=row["estimate"], ci_low=row["ci_low"], ci_high=row["ci_high"],
            p_reported=row["p_reported"], precision="existing table precision; no raw-CSV reformatting"))
        return row["text"]

    if FINAL_EXPORTS_ONLY and "exp2" in experiments and not (performance_only or exposure_memory_only):
        path = tables / "exp2_40n_context_reward_effects.tex"
        text = path.read_text()
        if "mgcv::bam (fREML; discrete=TRUE)" not in text or "Planning is the reference" not in text:
            raise ValueError(f"Cannot verify pilot context model: {path}")
        lines, labels = [], []
        for line in text.splitlines():
            cells = [c.strip() for c in line.removesuffix(r"\\").split(" & ")]
            if len(cells) != 5 or not re.fullmatch(NUMBER, cells[1]):
                continue
            labels.append(cells[0])
            lines.append(" & ".join(["Participants", *cells[:3], "---", cells[4]]) + r" \\")
        _, rows = parse_table("\n".join(lines), path, ("Participants",))
        if len(labels) != 13:
            raise ValueError(f"Expected all 13 pilot fixed-effect contrasts: {path}")
        for label in labels:
            stem = label.replace(r"$\times$", "by").replace("vs.", "vs")
            stem = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
            destination = output / "exp2_40n" / f"context_{stem}.tex"
            fragments[destination] = effect(path, rows, "Participants", label, stem, destination,
                "Participant-only binomial-logit BAM: six reward terms * condition; independent random slopes",
                "planning reference; exposure-minus-planning interactions", "standardized predictors; log-odds coefficients")

    if exposure_memory_only or (tables / "exposure_memory_performance.tex").exists():
        for spec in exposure_memory_effects(tables, output, experiments):
            fragments[spec[5]] = effect(*spec)
    if (tables / "exposure_memory_lambda.tex").exists():
        for spec in exposure_lambda_effects(tables, output, experiments):
            fragments[spec[5]] = effect(*spec)

    if not exposure_memory_only and (performance_only or (tables / "fitted_parameters_performance.tex").exists()):
        for spec in performance_effects(tables, output, experiments):
            fragments[spec[5]] = effect(*spec)

    for experiment in (e for e in experiments if e in ("exp1", "exp2") and not performance_only and not exposure_memory_only):
        # Adjacent logistic contrasts (path_rank.tex) are retired; ANOVA remains.
        anova_path = tables / f"{experiment}_adjacent_path_ranks_anova.tex"
        if (experiment == "exp1" or FINAL_EXPORTS_ONLY) and anova_path.exists():
            anova_rows = read_adjacent_anova_table(anova_path)
            anova_out = output / experiment / "path_rank_anova.tex"
            anova_groups = []
            for group, suffix in GROUPS.items():
                results = []
                for first in range(1, 9 if experiment == "exp2" else 4):
                    result = effect(anova_path, anova_rows, group, first,
                        f"rank_pair{first + 1}", anova_out,
                        "Classical two-level repeated-measures ANOVA; planning leaf and non-leaf probes; best containing path rank; paired participant means",
                        f"rank {first + 1} minus rank {first}; unadjusted F-test p",
                        "participant-mean accuracy difference; paired t CI; partial eta squared", r"\Delta")
                    results.append(f"rank {first} vs.\\ {first + 1}, " + result)
                anova_groups.append(("Participants" if suffix == "participants" else "Model") + ": " + "; ".join(results))
            fragments[anova_out] = "; ".join(anova_groups)

        path = tables / f"{experiment}_context_reward_effects.tex"
        caption, rows = read_table(path)
        if not any(phrase in caption.lower() for phrase in
                   ("planning is the reference condition", "planning as the reference condition")):
            raise ValueError(f"Cannot verify planning reference condition in {path}")
        for required in ("exposure-minus-planning",
                         "outcomes were not standardized", "binomial-logit glmer fits (Laplace ML)"):
            if required not in caption and not concise_caption_matches(caption, "context"):
                raise ValueError(f"Cannot verify context coding/estimator in {path}: {required}")
        model = "context_joint_conditions: six standardized reward terms * condition; independent participant intercept/slopes; binomial-logit glmer, Laplace ML, Wald z inference"
        interactions = []
        interaction_path = output / experiment / "planning_exposure_interactions.tex"
        for group, suffix in GROUPS.items():
            terms = []
            for short, (label, term) in EFFECTS.items():
                out = output / experiment / f"{short}_{suffix}.tex"
                result = effect(path, rows, group, label, term, out, model,
                    "planning slope; includes quadratic node/sibling terms")
                # Exp1 inserts standalone results; Exp2 puts these results
                # inside existing figure-reference parentheses.
                fragments[out] = "(" + result + ")" if experiment == "exp1" else result
                interaction = effect(path, rows, group, label + r" $\times$ condition",
                    term + ":conditionexposure", interaction_path, model,
                    "exposure minus planning reward-slope difference")
                terms.append(short + ", " + interaction)
            interactions.append(("Participants" if suffix == "participants" else "Model") + ": " + "; ".join(terms))
        # These grouped results follow figure/table references inside existing
        # parentheses in the manuscript; the caller supplies the "; " separator.
        fragments[interaction_path] = "; ".join(interactions)

        path = tables / f"{experiment}_branch_timing.tex"
        caption, rows = read_table(path)
        for required in ("earlier-minus-later", "standardized observation step", "binomial-logit glmer fits"):
            if required not in caption and not concise_caption_matches(caption, "branch"):
                raise ValueError(f"Cannot verify branch timing coding/estimator in {path}: {required}")
        out = output / experiment / "branch_timing.tex"
        terms = []
        for group, suffix in GROUPS.items():
            result = effect(path, rows, group, r"Context reward $\times$ timing",
                "context_reward_z:relative_branch_timingbefore", out,
                "branch_timing: context_reward_z * relative_branch_timing + observation_step_z; independent participant intercept/slopes; binomial-logit glmer, Laplace ML, Wald z inference",
                "before minus after context-reward slope; after is reference")
            terms.append(("Participants" if suffix == "participants" else "Model") + ": " + result)
        fragments[out] = "; ".join(terms)

        if experiment == "exp2":
            path = tables / "exp2_best_branch_timing.tex"
            # Manuscript fragments report humans and the full model only;
            # the companion table retains both alternative models.
            timing_groups = ("Participants", "Sequential model")
            rows = read_best_branch_timing_table(path)
            out = output / experiment / "best_branch_timing.tex"
            fragments[out] = "; ".join(
                group + ": " + effect(path, rows, group,
                    r"Best branch first vs.\ worst branch first", "best_branch_timingbefore", out,
                    "accuracy ~ best_branch_timing + (1 + best_branch_timing || participant); binomial-logit glmer, Laplace ML; independent intercept/timing slope; Wald z inference",
                    "best branch first minus worst branch first; middle-branch leaf probes (positions 6-8), extrema in distinct outside branches",
                    "unstandardized log-odds contrast; no continuous covariates")
                for group in timing_groups)

        out = output / experiment / "rank_change.tex"
        rank_results = []
        for group, source in (() if FINAL_EXPORTS_ONLY else (("Participants", "participants"), ("Sequential model", "sequential"))):
            directory = stats_root / experiment / source / "rank_change"
            row, audit = read_rank_change(directory)
            result = effect(directory / "coefficients.csv", {(group, "rank_conditionbetter"): row},
                group, "rank_conditionbetter", "rank_conditionbetter", out,
                "rank_change: rank_condition + observation_step_z * (node_reward_z + sibling_reward_z + parent_reward_z + aunt_reward_z + max_child_reward_z + avg_niece_reward_z); full independent participant intercept/slopes; binomial-logit glmer, Laplace ML, Wald z inference",
                "final better minus final worse; worse is reference; equal ranks and final observation excluded by current stats script",
                "categorical rank-condition contrast; unstandardized accuracy difference; continuous covariates standardized before interactions")
            records[-1].update(audit, precision="full-precision fitted CSV rounded to three decimals; p < .001 threshold")
            rank_results.append(("Participants" if source == "participants" else "Model") + ": " + result)
        if not FINAL_EXPORTS_ONLY:
            fragments[out] = "; ".join(rank_results)

        out = output / experiment / "nll_comparisons.tex"
        nll_results = []
        for path, variant, row, audit, values in read_nll_comparisons(stats_root, experiment):
            group = "Simultaneous model" if variant == "simultaneous" else "Independent model"
            result = effect(path, {(group, variant): row}, group, variant,
                f"{variant}_minus_sequential", out,
                "participant-paired t test; one-sided variant NLL > full sequential NLL; unadjusted p",
                "variant minus full sequential model; positive mean difference favors full model",
                "raw total NLL = path NLL + 0.5 * recall NLL; no standardization")
            records[-1].update(audit, coefficient_symbol=r"\Delta\mathrm{NLL}",
                precision="full-precision fitted CSV rounded to three decimals; unadjusted one-sided p; unadjusted two-sided 95% CI")
            nll_results.append(group + " vs.\\ full model: " + result)
        fragments[out] = "; ".join(nll_results)

        path = tables / f"{experiment}_observation_timestep.tex"
        rows = read_temporal_table(path)
        out = output / experiment / "temporal.tex"
        temporal_results = []
        for group, suffix in GROUPS.items():
            terms = []
            for label, term, display, contrast, standardization in (
                ("Observation timestep", "observation_step_z", "observation timestep",
                 "one-SD increase in observation step; adjusted for leaf status",
                 "observation step standardized; log-odds coefficient"),):
                result = effect(path, rows, group, label, term, out,
                    "observation_timestep: accuracy ~ is_leaf + observation_step_z + (1 + is_leaf + observation_step_z || participant); binomial-logit glmer, Laplace ML, Wald z inference; planning condition",
                    contrast, standardization)
                terms.append(result)
            temporal_results.append(("Participants" if suffix == "participants" else "Model") + ": " + "; ".join(terms))
        fragments[out] = "; ".join(temporal_results)

    if "exp3" in experiments and not performance_only and not exposure_memory_only:
        path = tables / "exp3_tree_order_effects.tex"
        caption, rows = read_table(path, outcome_column=True)
        for required in ("BAM fits (fREML; discrete=TRUE)", "mgcv BAM fits (fREML)", "binomial-logit", "Gaussian models",
                         "DFS minus BFS averaged equally", "wide minus deep averaged equally",
                         "deep = $-0.5$", "DFS = $+0.5$"):
            if required not in caption and not concise_caption_matches(caption, "tree_order"):
                raise ValueError(f"Cannot verify Exp3 tree/order model in {path}: {required}")
        predictors = (("order", "Presentation order", "order_c", "DFS minus BFS, averaged equally over tree structures"),
                      ("tree", "Tree structure", "tree_c", "wide minus deep, averaged equally over presentation orders"),
                      ("tree_order", r"Order $\times$ tree", "tree_c:order_c", "DFS-minus-BFS difference in wide minus deep trees"))
        for outcome, label in (("norm_reward", "Normalized reward"), ("accuracy", "Recall accuracy")):
            for short, predictor, term, contrast in predictors:
                for group, suffix in GROUPS.items():
                    out = output / "exp3" / f"{outcome}_{short}_{suffix}.tex"
                    fragments[out] = effect(path, rows, group, label + ": " + predictor, term, out,
                        ("BAM fREML discrete=TRUE: " if outcome == "accuracy" else "BAM fREML: ") + "tree_c * order_c with independent participant intercept and contrast slopes; " +
                        ("binomial-logit, normal-Wald z inference" if outcome == "accuracy" else "Gaussian, approximate t inference"),
                        contrast, "centered -0.5/+0.5 predictors; " +
                        ("unstandardized log-odds" if outcome == "accuracy" else "unstandardized normalized reward"))
        path = tables / "exp3_rank_change.tex"
        caption, rows = read_table(path)
        for required in ("binomial-logit BAM fits (fREML; discrete=TRUE)", "Equal ranks are excluded",
                         "Improved = 1 and worsened = 0",
                         "Final rank is numeric and unscaled",
                         "centered tree and order", "independent participant intercept and slopes for all eleven fixed-effect terms"):
            if required not in caption and not concise_caption_matches(caption, "rank"):
                raise ValueError(f"Cannot verify Exp3 rank-change model in {path}: {required}")
        for short, term, contrast in (("main", "final_rank_better_than_encoding_rank", "rank-improvement effect averaged equally over tree structures and presentation orders on the log-odds scale"),
                                      ("order", "final_rank_better_than_encoding_rank:order_c", "rank-improvement effect: DFS minus BFS"),
                                      ("tree", "final_rank_better_than_encoding_rank:tree_c", "rank-improvement effect: wide minus deep"),
                                      ("tree_order", "final_rank_better_than_encoding_rank:tree_c:order_c", "DFS-minus-BFS modulation of rank improvement: wide minus deep")):
            for group, suffix in GROUPS.items():
                out = output / "exp3" / f"rank_change_{short}_{suffix}.tex"
                label = "Rank improvement" if short == "main" else r"Rank improvement $\times$ " + (r"tree $\times$ order" if short == "tree_order" else short)
                fragments[out] = effect(path, rows, group, label,
                    term, out,
                    "BAM binomial-logit fREML discrete=TRUE: final_rank_better_than_encoding_rank * tree_c * order_c + final_rank * tree_c * order_c; independent participant intercept and all eleven fixed-effect slopes; all trials and nodes; Wald z inference",
                    contrast + "; holding final rank constant; improved=1 versus worsened=0, equal ranks excluded",
                    "unstandardized log-odds main effect" if short == "main" else "unstandardized log-odds interaction")

    # Validate every source before writing anything. Never touch full tables.
    if not performance_only and not exposure_memory_only:
        for experiment in set(experiments) & {"exp1", "exp2"}:
            old = output / experiment / "path_rank.tex"
            if old.exists():
                archive = old.parent / "retired_logistic_rank"
                archive.mkdir(parents=True, exist_ok=True)
                target = archive / f"path_rank_{uuid.uuid4().hex}.tex"
                old.rename(target)
                print(f"Archived retired logistic fragment: {target}")
    for path, fragment in fragments.items():
        if fragment.endswith(".") or "$p=0.000$" in fragment or "$p=.000$" in fragment:
            raise ValueError(f"Invalid fragment: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        # A terminal newline becomes a space when LaTeX inputs the fragment.
        path.write_text(fragment.rstrip())
    for name, data in (("sources.csv", records), ("unavailable.csv", unavailable)):
        with (output / name).open("w", newline="") as handle:
            fields = list(dict.fromkeys(key for row in data for key in row)) or ["output", "reason"]
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)

    lines = ["# In-text statistics: " + ", ".join(experiments), "",
        "Exposure-memory fragments reproduce the unadjusted sigma-only OLS table. Base files contain b, CI, p; _bp files contain b and p; _p files contain only p. None has outer parentheses or a final period. Parameters are estimated from exposure, outcomes from planning; no parameter-estimation uncertainty is propagated.", "",
        "Exp3 fragments reproduce the current mixed-model tables: accuracy and rank-change coefficients are log-odds, normalized reward coefficients are outcome-scale. Separate files contain only b, CI and p, with no group label or outer parentheses. No supplied manuscript numbers are hard-coded.", "",
        f"{len(fragments)} fragments exported; {len(unavailable)} requested fragments not generated. No analyses were fitted, model definitions changed, manuscript edited, or full tables overwritten by this exporter.", "",
        "## Provenance and reporting", "",
        "Adjacent-rank, context and branch-timing fragments reproduce the reported local table values (three decimals / thresholded p values). Rank-change fragments instead use full-precision downloaded coefficients.csv values, validated against model.txt, random_design.csv and scaling.csv. The complete current formula, independent random slopes, better-minus-worse coding, estimator and inference method must match; obsolete/additive-only fits are rejected before writing. The exporter does not refit models. `sources.csv` records every coefficient, source SHA-256 and destination, plus rank-change diagnostic/scaling paths, hashes, singularity and convergence messages. Raw behavioral/simulation input hashes are not recorded in these downloaded model diagnostics and are not independently verified here.", "",
        "Adjacent logistic rank-contrast fragments (`path_rank.tex`) are no longer exported. Classical adjacent-rank ANOVA exports remain separate (`path_rank_anova.tex`). Rank-change analyses are unaffected.", "",
        "All coefficients are labeled **b**: continuous predictors were standardized, but accuracy was not. They are response-scale coefficients per SD of a continuous predictor, or differences between those slopes; they are not fully standardized response-and-predictor beta coefficients. This notation change does not rescale any estimate or CI. The source CSV column named `beta` and table beta header do not by themselves establish full standardization.", "",
        "Planning is the reference condition; contextual main effects are planning slopes and condition interactions are **exposure minus planning**. Branch timing uses after as reference and reports the **before minus after** slope difference. Signs were preserved. Node/sibling coefficients are the linear terms in a model that also includes their squared rewards; they are not omnibus or uniformly monotonic reward effects.", "",
        "Rank-change fragments report final **better minus worse** for participants and the sequential model. The current stats script excludes equal ranks and the last observation. The validated model includes timestep interactions with all six reward covariates and full independent participant random slopes. `rank_change.tex` omits outer parentheses for insertion after a figure/table reference with `; `. The fitted estimates are exported as saved even when the model is singular: inspect singularity/convergence in sources.csv before reporting. The sequential analyses are mixed-effects models, not fixed-effects-only regressions. No estimator was changed.", "",
        "## Generated files", "",
        "`nll_comparisons.tex` reports simultaneous-minus-full and independent-minus-full mean total NLL differences, labeled Delta NLL (not b or beta). Values come from downloaded nll_vs_sequential/comparisons.csv and validated model.txt files. The printed p values are one-sided and unadjusted; Holm-adjusted values remain in the source tables; confidence intervals are unadjusted two-sided 95% t intervals. These are exploratory in-sample selected-fit comparisons, not held-out prediction or likelihood-ratio tests. The fragment has no outer parentheses or final period. No manuscript text is changed automatically.", "",
        "Exp1 single-effect files include outer parentheses. Exp2 single-effect files, `planning_exposure_interactions.tex` and `branch_timing.tex` omit them so they can share existing figure/table parentheses, e.g. `(Figure~\\ref{fig:task2_1}e\\change{stat_13}{; Participants: \\input{intext_stats/exp2/probed_participants.tex}; Model: \\input{intext_stats/exp2/probed_model.tex}})`. Group labels distinguish participants from the full sequential model. No df or final period is printed. Use an explicit space before standalone change-wrapped fragments, e.g. `reward~\\change{stat_1}{\\input{intext_stats/exp1/probed_model.tex}}`. Upload the directory to Overleaf as `intext_stats/` (adjust references to its actual location).", ""]
    for path in fragments:
        relevant = [row for row in records if row["output"] == display_path(path)]
        lines.append(f"- `{display_path(path)}` — `{relevant[0]['source']}`; " +
                     "; ".join(row["group"] + ": " + row["term"] for row in relevant) + "; " + relevant[0]["coefficient_symbol"] + " (" + relevant[0]["standardization"] + ")")
    lines += ["", "## Not generated", ""]
    lines += [f"- `{row['output']}` — {row['reason']}" for row in unavailable]
    lines += ["", "`temporal.tex` reproduces the observation-timestep and leaf-status table results for participants and the full sequential model, with b, 95% CI and unadjusted p. Timestep is per one-SD increase; leaf status is leaf minus non-leaf (signs are not reversed). Both terms are precision only; no guessing covariate. The fragment has no outer parentheses or final period; insert it with `; \\input{intext_stats/exp1/temporal.tex}` inside existing figure-reference parentheses (replace exp1 with exp2 as needed). It reports the saved results without inferring whether they support the manuscript's prose. No manuscript is automatically edited. Do not infer missing statistics from figures or relabel adjacent-rank contrasts as ANOVAs.", "",
              "Regenerate with `python analysis/scripts/stats/fitted/export_intext_stats.py`. Rank inputs default to `.local_artifacts/summaries/analysis/stats/fitted`; use `--stats-root PATH` to select another downloaded statistics directory."]
    if FINAL_EXPORTS_ONLY:
        lines = ["# Final rebuttal in-text statistics", "",
            "Exports correspond to the selected manuscript tables. Retired Exp1/2 rank-change and four-coefficient memory-performance fragments are not generated.",
            "Coefficient fragments retain the validated source table/CSV estimates and inference. See sources.csv for inferential provenance.",
            "Engagement fragments contain participant-weighted means and SDs copied from the corresponding tables; they are descriptive, not regression coefficients.",
            "The six legacy 40-node fragments are written to --legacy-stats-dir/exp2_40n_beh. No generated statistic fragment has trailing whitespace.",
            "Existing obsolete files are not deleted; use a fresh output directory for a clean export.", "",
            "## Inferential fragments", ""] + [f"- `{display_path(path)}`" for path in fragments]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    for path in fragments:
        print("Generated:", display_path(path))
    for row in unavailable:
        print("Unavailable:", row["output"], "—", row["reason"])
    return fragments, records, unavailable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=INTEXT_DIR)
    parser.add_argument("--stats-root", type=Path, default=DEFAULT_STATS_ROOT)
    parser.add_argument("--legacy-stats-dir", type=Path, default=LEGACY_STATS_DIR,
                        help="Destination for stats/exp2_40n_beh descriptive fragments")
    parser.add_argument("--engagement-only", action="store_true")
    parser.add_argument("--experiments", default="exp1,exp2,exp3",
                        help="Comma-separated experiments (default: exp1,exp2,exp3)")
    parser.add_argument("--performance-only", action="store_true",
                        help="Export only parameter-performance, memory-performance, and set-size fragments")
    parser.add_argument("--quadratic-performance-only", action="store_true",
                        help="Export participant quadratic-coefficient performance fragments only")
    parser.add_argument("--exposure-memory-only", action="store_true",
                        help="Export exposure-mixture performance fragments and lambda fragments when their table exists")
    args = parser.parse_args()
    experiments = tuple(args.experiments.split(","))
    if not experiments or any(e not in ("exp1", "exp2", "exp3") for e in experiments):
        parser.error("--experiments must select exp1, exp2 and/or exp3")
    if sum((args.performance_only, args.exposure_memory_only, args.quadratic_performance_only, args.engagement_only)) > 1:
        parser.error("Choose only one restricted export mode")
    if not (args.performance_only or args.exposure_memory_only or args.quadratic_performance_only):
        export_engagement(args.tables_dir, args.output_dir, experiments, args.legacy_stats_dir)
    if args.engagement_only:
        pass
    elif args.quadratic_performance_only:
        export_quadratic_performance(args.tables_dir, args.output_dir, experiments)
    else:
        export(args.tables_dir, args.output_dir, args.stats_root, experiments, args.performance_only, args.exposure_memory_only)
