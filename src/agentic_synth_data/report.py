from pathlib import Path
import pprint
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from agentic_synth_data.schemas.attack_model import AttackMetrics
from agentic_synth_data.schemas.models import SynthesisRequest
from agentic_synth_data.schemas.syndat_model import SyndatMetrics
from agentic_synth_data.utils import OUTPUT_DIR


class InputReport(BaseModel):
    """Initial dataset review and version for report."""
    path: str | None = None
    original_shape: tuple[int, int] | None = None
    prepared_shape: tuple[int, int] | None = None
    version_id: int | None = None
    parent_version: int | None = None


class PrepReport(BaseModel):
    """Preparation steps and information gathered prior to generation for report."""
    missing_markers_detected: dict = Field(default_factory=dict)
    missing_markers_replaced: dict = Field(default_factory=dict)
    dtypes: dict = Field(default_factory=dict)
    stypes: dict = Field(default_factory=dict)
    profile: dict = Field(default_factory=dict)
    missing_before: dict = Field(default_factory=dict)
    missing_after: dict = Field(default_factory=dict)


class GenerationReport(BaseModel):
    """Information gathered from synthetic data generation for report."""
    prep_path: str | None = None
    synth_path: str | None = None
    synth_shape: tuple[int, int] | None = None
    params: dict[str, Any] | None = None


class EvalReport(BaseModel):
    """Evaluation metrics with plots comparing real and synthetic data for report."""
    attack: AttackMetrics | None = None
    syndat: SyndatMetrics | None = None
    plots: dict[str, list[Path]] = Field(default_factory=dict)


class ReviewReport(BaseModel):
    # TODO: After splitting, as this is, it only has the same information as AgentEval. Possibly add more regarding user input in response.
    """Agent identified input for report."""
    problem_columns: list[str] = Field(default_factory=list)
    status: str | None = None
    recommendation: str | None = None
    rationale: str | None = None
    # user_decision - finish | revise
    # additional details...


class UpdatePlanReport(BaseModel):
    """The plan update that was selected for a subsequent run to be included in the current report.
    """
    selected_plan_updates: dict = Field(default_factory=dict)


class Report(BaseModel):
    """Report of main changes made during a given workflow iteration."""
    input: InputReport = Field(default_factory=InputReport)
    prep: PrepReport = Field(default_factory=PrepReport)
    gen: GenerationReport = Field(default_factory=GenerationReport)
    evaluate: EvalReport = Field(default_factory=EvalReport)
    review: ReviewReport = Field(default_factory=ReviewReport)
    update: UpdatePlanReport = Field(default_factory=UpdatePlanReport)
    revisions: list = Field(default_factory=list)


# possible TODO: break into section pieces, but first determine what the report should look like/sections should look like...
def write_report(
    report: Report, synthesis_request: SynthesisRequest, data_path: str, version_id: int,
) -> None:
    """Write a workflow report based on decisions made during the synthetic data generation process.

    Args:
        report (Report): Dictionary containing information gathered throughout the workflow steps.
        synthesis_request (SynthesisRequest): Dictionary with the user's preferences.
        data_path (str): Initial path of the data (used for naming the report).
        version_id (int): Denotes the version of the workflow attempt.
    """
    data_name = Path(data_path).stem
    output_filename = f"report_{data_name}_v{version_id}.md"

    output_dir = OUTPUT_DIR / f"version_{version_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_filename

    parent = (
        f"; parent_version_{report.input.parent_version}"
        if report.input.parent_version else ""
    )

    dtype_table = pd.DataFrame({
        "Feature": report.prep.stypes.keys(),
        "Data type": [report.prep.dtypes[col] for col in report.prep.stypes.keys()],
        "Semantic type": report.prep.stypes.values(),
    })

    # TODO: add train/test split information

    def display_plots(paths: list[Path]) -> str:
        return "\n\n".join(
            f"![{path.stem}]({path.relative_to(output_dir).as_posix()})"
            for path in paths
        )

    # Distribution plots
    distribution_paths = report.evaluate.plots.get("distributions", [])
    top_plots = display_plots(distribution_paths[:3])
    remaining_distributions = display_plots(distribution_paths[3:])

    # Correlation plots
    correlation_paths = report.evaluate.plots.get("correlations", [])

    original_correlations = display_plots(
        [path for path in correlation_paths if "real_corr" in path.stem.lower()]
    )

    synthetic_correlations = display_plots( # NOTE: synthetic_corr is misspelled.
        [path for path in correlation_paths if "syntehtic_corr" in path.stem.lower()]
    )

    # SHAP plots
    shap_plots = display_plots(report.evaluate.plots.get("shap", []))

    def fmt(value):
        return pprint.pformat(value, sort_dicts=False)

    missing_markers = ""
    if report.prep.missing_markers_detected:
        missing_markers = f"""
Missing value marker detection

Detected markers:
```python
{fmt(report.prep.missing_markers_detected)}
```

Replaced markers:
```python
{fmt(report.prep.missing_markers_replaced)}
```
"""
    # Syndat and anonymeter metrics
    syndat = report.evaluate.syndat
    attack = report.evaluate.attack
    syndat_output = syndat.output_messages
    attack_output = attack.output_messages

    jsd_output = (
        f"- **Output (Jensen-Shannon Distance):** {syndat_output.jsd}" if syndat_output.jsd else ""
    )
    norm_corr_diff_output = (
        f"- **Output (Normalized correlation difference):** {syndat_output.norm_corr_diff}"
        if syndat_output.norm_corr_diff else ""
    )
    discrim_auc_output = (
        f"- **Output (Discriminator AUC):** {syndat_output.discrim_auc}" 
        if syndat_output.discrim_auc else ""
    )
    singling_out_output = (
        f"- **Output (Singling out risk):** {attack_output.singling_out}"
        if attack_output.singling_out else ""
    )
    inference_output = (
        f"- **Output (Inference risk):** {attack_output.inference}"
        if attack_output.inference else ""
    )
    linkability_output = (
        f"- **Output (Linkability risk):** {attack_output.linkability}"
        if attack_output.linkability else ""
    )

    # Main structure of the report and information content
    content = f"""# SYNTHETIC DATA GENERATION REPORT
(version_{version_id}{parent})

### Information about the original (input) dataset:
- **Location (path):** `{report.input.path}`

- **Original shape:** {report.input.original_shape[0]} rows, {report.input.original_shape[1]} columns.

## Preprocessing:
{missing_markers}

### Python data types
```python
{fmt(report.prep.dtypes)}
```

### Semantic data types
```python
{fmt(report.prep.stypes)}
```

### Data types
{dtype_table.to_markdown()}


### Profile

This information was gathered to inform the generator choice by viewing the characteristics of the original dataset before imputation.
- Shape of the original dataframe (before missing handling) after an 80% train 20% test split: {
    report.prep.profile['shape'][0]
} rows and {report.prep.profile['shape'][1]} columns.

The number of unique values per feature (cardinality):
```python
{fmt(report.prep.profile['col_nunique'])}
```

Summary statistics for numeric values:
```python
{fmt(report.prep.profile['num_summary'])}
```

{report.prep.profile['corr_method'].capitalize()}'s correlation for numeric features:
```python
{fmt(report.prep.profile['num_corr'])}
```

### Missing Value Handling
- Number of missing values found before missing value handling was applied: {
    report.prep.missing_before
}

- Number of missing values after missing value handling was applied: {report.prep.missing_after}

- Prepared dataset shape (this is input to the generator model for synthetic generation): {
    report.input.prepared_shape[0]
} rows, {report.input.prepared_shape[1]} columns.

### Generation:
**Chosen generator:** {synthesis_request.generator.name}

**Parameters:** {report.gen.params}

**Prepared dataset path:** `{report.gen.prep_path}`

**Synthetic dataset path:** `{report.gen.synth_path}`

**Synthetic shape:** {report.gen.synth_shape[0]} rows, {report.gen.synth_shape[1]} columns.

### Results:
**Jensen-Shannon Distance:**
```python
{fmt(syndat.jsd)}
```
{jsd_output}

**Normalized correlation difference:** {syndat.norm_corr_diff}

{norm_corr_diff_output}

**Discriminator AUC:** {syndat.discrim_auc}

{discrim_auc_output}

### Plots

Generated plots are stored in:
`{output_dir / "plots"}`

The first three plots are displayed below. All additional plots are available at the end of this report.

{top_plots}

---

**Singling out risk:** {round(attack.singling_out.value, 4)}
- **Confidence interval:** {round(attack.singling_out.ci[0], 4)} – {round(attack.singling_out.ci[1], 4)}

{singling_out_output}

**Inference risk:** {round(attack.inference.value, 4)}
- **Confidence interval:** {round(attack.inference.ci[0], 4)} – {round(attack.inference.ci[1], 4)}

{inference_output}

**Linkability risk:** {round(attack.linkability.value, 4)}
- **Confidence interval:** {round(attack.linkability.ci[0], 4)} – {round(attack.linkability.ci[1], 4)}

{linkability_output}

### Agent review:

- **Status:** {report.review.status}

- **Recommendation:** {report.review.recommendation}

- **Rationale behind the recommendation:** {report.review.rationale}

- **Problem columns:** {report.review.problem_columns}

---

## Additional plots

### Distributions

{remaining_distributions or "No additional distribution plots."}

### Correlations

#### Original dataset

{original_correlations or "No original correlation plots."}

#### Synthetic dataset

{synthetic_correlations or "No synthetic correlation plots."}

### SHAP discrimination

{shap_plots or "No SHAP plots."}
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
