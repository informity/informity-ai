from __future__ import annotations

from informity.diagnostics.translate_quality import compute_translate_quality


class _Section:
    def __init__(self, text: str, completed: bool = True, truncated: bool = False) -> None:
        self.text = text
        self.completed = completed
        self.truncated = truncated


def test_compute_translate_quality_passes_clean_two_section_run() -> None:
    metrics = compute_translate_quality(
        [
            _Section('Project overview and key findings.'),
            _Section('Key findings and next steps for the project.'),
        ],
        section_count=2,
        truncated_sections=0,
    )

    assert metrics.completion_rate == 1.0
    assert metrics.structural_fidelity == 1.0
    assert metrics.recommendation == 'pass'
    assert metrics.quality_score >= 80
    assert metrics.term_consistency is not None


def test_compute_translate_quality_penalizes_missing_and_truncated_sections() -> None:
    metrics = compute_translate_quality(
        [
            _Section('One section only.', truncated=True),
            _Section('', completed=False),
        ],
        section_count=2,
        truncated_sections=1,
    )

    assert metrics.completion_rate == 0.5
    assert metrics.truncation_rate == 0.5
    assert metrics.structural_fidelity < 1.0
    assert metrics.recommendation in {'hold', 'review'}
