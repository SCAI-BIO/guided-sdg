from pathlib import Path
import pprint
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field
import streamlit as st

from agentic_synth_data.schemas.privacy import AttackMetrics
from agentic_synth_data.schemas.synthesis_request import SynthesisRequest
from agentic_synth_data.schemas.syndat import SyndatMetrics
from agentic_synth_data.utils import OUTPUT_DIR


class InputReport(BaseModel):
    """Initial dataset review and version for report."""
    path: str | None = None
    original_shape: tuple[int, int] | None = None
    train_shape: tuple[int, int] | None = None
    test_shape: tuple[int, int] | None = None
    prep_train_shape: tuple[int, int] | None = None
    prep_test_shape: tuple[int, int] | None = None
    version_id: int | None = None
    parent_version: int | None = None


class PrepReport(BaseModel):
    """Preparation steps and information gathered prior to generation for report."""
    missing_markers_detected: dict = Field(default_factory=dict)
    missing_markers_replaced: dict = Field(default_factory=dict)
    dtypes: dict = Field(default_factory=dict)
    stypes: dict = Field(default_factory=dict)
    profile: dict = Field(default_factory=dict)
    train_missing_before: dict = Field(default_factory=dict)
    test_missing: dict = Field(default_factory=dict)
    train_missing_after: dict = Field(default_factory=dict)


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


# Note: tabulate package is required for dtype_table.to_markdown(), think if needed/alternative can be used
# possible TODO: break into section pieces, but first determine what the report should look like/sections should look like...
def write_report(
    report: Report, request: SynthesisRequest, data_path: str, version_id: int,
) -> None:
    """Write a workflow report based on decisions made during the synthetic data generation process.

    Args:
        report (Report): Dictionary containing information gathered throughout the workflow steps.
        request (SynthesisRequest): Dictionary with the user's preferences.
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
    # Syndat and Anonymeter metrics
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

    def missing_summary(count: int, shape: tuple[int, int]) -> str:
        """Format missing values as count, total cells, and percentage."""
        total = shape[0] * shape[1]
        percentage = f"{count / total:.2%}" if total else "N/A"
        return f"{count:,} / {total:,} ({percentage})"

    train_before = report.prep.train_missing_before
    train_after = report.prep.train_missing_after

    test_missing = report.prep.test_missing

    cardinalities = {
        col: values["n_unique"] for col, values in report.prep.profile["col_nunique"].items()
    }

    highest_cardinality = sorted(
        cardinalities.items(), key=lambda item: item[1], reverse=True,
    )[:2]

    highest_cardinality_text = ", ".join(
        f"`{col}` ({count} unique values)" for col, count in highest_cardinality
    )

    # Main structure of the report and information content
    content = f"""# SYNTHETIC DATA GENERATION REPORT
(version_{version_id}{parent})

### Information about the original (input) dataset:
- **Location (path):** `{report.input.path}`

- **Original shape:** {report.input.original_shape[0]} rows, {report.input.original_shape[1]} columns.

## Preprocessing:
{missing_markers}

### Data types
{dtype_table.to_markdown()}

### Train/test split information
The dataset was split into training (80%) and testing (20%) subsets using a random seed of 42.

| Dataset | Shape |
|---|---|
| Training | {report.input.train_shape[0]} rows, {report.input.train_shape[1]} columns |
| Testing | {report.input.test_shape[0]} rows, {report.input.test_shape[1]} columns |
| Prepared training | {report.input.prep_train_shape[0]} rows, {report.input.prep_train_shape[1]} columns |

**Dataset usage:**
- **Original training:** Syndat fidelity evaluation and Anonymeter privacy evaluation.
- **Original testing:** Held-out reference for Anonymeter privacy evaluation.
- **Prepared training:** Synthetic data generation.

### Training data profile

The following characteristics were calculated from the original training subset before missing-value handling to inform generator selection.

**Dataset characteristics**
- **Features:** {report.input.train_shape[1]}
- **Highest cardinality:** {highest_cardinality_text}

**Feature cardinality**

The number of unique values per feature (cardinality):
```python
{fmt(report.prep.profile['col_nunique'])}
```

**Summary statistics for numeric values:**
```python
{fmt(report.prep.profile['num_summary'])}
```

**{report.prep.profile['corr_method'].capitalize()}'s correlation for numeric features:**
```python
{fmt(report.prep.profile['num_corr'])}
```

### Missing-value handling

**Selected method:** {request.missing_values.mode}

| Dataset | Missing values |
|---|---|
| Training (before handling) | {missing_summary(train_before, report.input.train_shape)} |
| Training (after handling) | {missing_summary(train_after, report.input.prep_train_shape)} |
| Testing (untreated) | {missing_summary(test_missing, report.input.test_shape)} |

**Total remaining in prepared training:** {train_after:,} missing values.

**Note:** Missing-value handling was applied only to the training dataset.
The original testing dataset was retained for privacy evaluation.

### Generation:
**Chosen generator:** {request.generator.name}

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
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def display_report_ui(revision: "DataGen") -> None:
    """Display the generated report for a particular workflow version.

    Args:
        revision (DataGen): Revision workflow object.
    """
    version_id = revision.version_id
    data_name = Path(revision.data_path).stem

    path = OUTPUT_DIR / f"version_{version_id}" / f"report_{data_name}_v{version_id}.md"

    if not path.is_file():
        st.info("A report has not yet been generated for this version.")
        return

    st.markdown(path.read_text())
