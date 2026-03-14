from informity.llm.rag_runtime.strict_output_contract import (
    _build_contract_prompt_requirements,
    _build_output_contract_plan,
    _evaluate_output_contract,
)


def test_strict_output_contract_plan_extracts_headings_and_order() -> None:
    plan = _build_output_contract_plan(
        question='Create sections in order: 1) Scope, 2) Method.',
        format_requirements=[
            'use the required headings exactly and in the requested order',
            'include heading: Scope',
            'include heading: Method',
            'use nested bullet lists with exactly 3 levels where requested',
            'explicitly call out missing evidence by requested group and/or year',
        ],
    )
    assert plan.required_headings == ('Scope', 'Method')
    assert plan.enforce_order is True
    assert plan.required_bullet_depth == 3
    assert plan.requires_missing_evidence_callout is True


def test_strict_output_contract_evaluation_detects_missing_heading() -> None:
    plan = _build_output_contract_plan(
        question='Create sections in order: 1) Scope, 2) Method.',
        format_requirements=[
            'use the required headings exactly and in the requested order',
            'include heading: Scope',
            'include heading: Method',
        ],
    )
    check = _evaluate_output_contract(
        answer='## 1) Scope\nDetails here.',
        plan=plan,
    )
    assert check['passed'] is False
    assert check['missing_headings'] == ['Method']


def test_strict_output_contract_evaluation_passes_when_complete() -> None:
    plan = _build_output_contract_plan(
        question='Create sections in order: 1) Scope, 2) Method.',
        format_requirements=[
            'use the required headings exactly and in the requested order',
            'include heading: Scope',
            'include heading: Method',
        ],
    )
    check = _evaluate_output_contract(
        answer='## 1) Scope\n- detail\n\n## 2) Method\n- detail',
        plan=plan,
    )
    assert check['passed'] is True


def test_strict_output_contract_builds_prompt_requirements_for_order_and_depth() -> None:
    plan = _build_output_contract_plan(
        question='Create sections in order: 1) Scope, 2) Method.',
        format_requirements=[
            'use the required headings exactly and in the requested order',
            'include heading: Scope',
            'include heading: Method',
            'use nested bullet lists with exactly 3 levels where requested',
            'explicitly call out missing evidence by requested group and/or year',
        ],
    )
    requirements = _build_contract_prompt_requirements(plan)
    assert any('section skeleton in order' in item for item in requirements)
    assert any('at the level shown' in item for item in requirements)
    assert any('heading template shape' in item for item in requirements)
    assert any('reaches depth 3' in item for item in requirements)
    assert any('Missing Evidence:' in item for item in requirements)
    assert any('starts exactly with "Missing Evidence:"' in item for item in requirements)


def test_strict_output_contract_plan_extracts_exact_order_headings_from_question() -> None:
    plan = _build_output_contract_plan(
        question=(
            'Required headings in exact order: ## Scope, ## Method, ## Findings by Year, '
            '## Cross-Year Deltas.'
        ),
        format_requirements=[],
    )
    assert plan.enforce_order is True
    assert plan.required_headings == (
        '## Scope',
        '## Method',
        '## Findings by Year',
        '## Cross-Year Deltas',
    )


def test_strict_output_contract_evaluation_enforces_word_and_bullet_limits() -> None:
    plan = _build_output_contract_plan(
        question='Return exactly 3 bullets and keep total <= 10 words.',
        format_requirements=[],
    )
    check = _evaluate_output_contract(
        answer='- one\n- two\n- three\n- four\n\nthis sentence has many extra words beyond allowed total budget.',
        plan=plan,
    )
    assert check['passed'] is False
    assert check['word_count_ok'] is False
    assert check['top_level_bullet_count_ok'] is False


def test_strict_output_contract_plan_extracts_top_level_bullet_phrase() -> None:
    plan = _build_output_contract_plan(
        question='Rewrite to exactly 3 top-level bullets in total and keep total <= 180 words.',
        format_requirements=[],
    )
    assert plan.max_words == 180
    assert plan.exact_top_level_bullets == 3
    assert plan.exact_top_level_bullets_section is None
    requirements = _build_contract_prompt_requirements(plan)
    assert any('do not use sub-bullets' in item for item in requirements)


def test_strict_output_contract_plan_extracts_section_scoped_top_level_bullets() -> None:
    plan = _build_output_contract_plan(
        question=(
            'Under ## Missing Evidence include exactly 5 bullets and each bullet must name one missing item.'
        ),
        format_requirements=[],
    )
    assert plan.exact_top_level_bullets == 5
    assert plan.exact_top_level_bullets_section == '## Missing Evidence'

    requirements = _build_contract_prompt_requirements(plan)
    assert any('under heading "## Missing Evidence"' in item for item in requirements)
    assert any('using "- " prefix' in item for item in requirements)


def test_strict_output_contract_missing_evidence_accepts_bullet_prefixed_canonical_line() -> None:
    plan = _build_output_contract_plan(
        question='Call out missing evidence explicitly.',
        format_requirements=['explicitly call out missing evidence by requested group and/or year'],
    )
    check = _evaluate_output_contract(
        answer='- Missing Evidence: No verified document found for 2024 payroll summary.',
        plan=plan,
    )
    assert check['missing_evidence_callout_ok'] is True
    assert check['missing_evidence_callout_canonical'] is True
    assert check['missing_evidence_callout_legacy_only'] is False


def test_strict_output_contract_plan_requires_evidence_grounding_from_format_requirement() -> None:
    plan = _build_output_contract_plan(
        question='Build a compliance reconciliation brief across years.',
        format_requirements=[
            (
                'require canonical evidence grounding for every claim-bearing bullet/list item '
                'and narrative claim paragraph'
            ),
        ],
    )
    assert plan.requires_evidence_grounding is True
    assert plan.evidence_grounding_excluded_sections == (
        'executive summary',
        'scope',
        'method',
        'confidence notes',
    )


def test_strict_output_contract_plan_requires_evidence_grounding_from_evidence_quality_phrase() -> None:
    plan = _build_output_contract_plan(
        question='Produce an audit brief comparing year-over-year evidence quality across indexed records.',
        format_requirements=[],
    )
    assert plan.requires_evidence_grounding is True


def test_strict_output_contract_evidence_grounding_excludes_structural_sections() -> None:
    plan = _build_output_contract_plan(
        question=(
            'Required headings in exact order: ## Scope, ## Findings. '
            'Keep all claims evidence-grounded.'
        ),
        format_requirements=[],
    )
    check = _evaluate_output_contract(
        answer=(
            '## Scope\n'
            'This section summarizes methodology assumptions without explicit citation.\n\n'
            '## Findings\n'
            '- Claim A with support. Evidence: ledger.pdf, page 2\n'
        ),
        plan=plan,
    )
    assert check['evidence_grounding_ok'] is True
    assert check['passed'] is True


def test_strict_output_contract_evidence_grounding_counts_numbered_list_items() -> None:
    plan = _build_output_contract_plan(
        question='Provide evidence-grounded verification steps.',
        format_requirements=[
            (
                'require canonical evidence grounding for every claim-bearing bullet/list item '
                'and narrative claim paragraph'
            ),
        ],
    )
    check = _evaluate_output_contract(
        answer=(
            '## Next Verification Steps\n'
            '1. Validate tax amount against source\n'
            '2. Confirm year mapping. Evidence: tax.pdf, page 4\n'
        ),
        plan=plan,
    )
    assert check['evidence_claim_block_count'] == 2
    assert check['evidence_grounding_ok'] is False


def test_strict_output_contract_prompt_adds_verification_and_delta_guidance() -> None:
    plan = _build_output_contract_plan(
        question=(
            'Required headings in exact order: ## Scope, ## Largest Increase, ## Largest Decrease, '
            '## Next Verification Steps. Keep claims evidence-grounded.'
        ),
        format_requirements=[],
    )
    requirements = _build_contract_prompt_requirements(plan)
    assert any('every numbered or bulleted step must include canonical' in item for item in requirements)
    assert any('express numeric delta claims as bullet items' in item for item in requirements)


def test_strict_output_contract_prompt_adds_findings_cross_year_evidence_guidance() -> None:
    plan = _build_output_contract_plan(
        question=(
            'Required headings in exact order: ## Scope, ## Findings by Year, ## Cross-Year Deltas, '
            '## Next Verification Steps. Keep claims evidence-grounded.'
        ),
        format_requirements=[],
    )
    requirements = _build_contract_prompt_requirements(plan)
    assert any('every source, amount, and contradiction bullet' in item for item in requirements)


def test_strict_output_contract_missing_evidence_accepts_table_cell_marker() -> None:
    plan = _build_output_contract_plan(
        question='Show matrix gaps and explicitly call out missing evidence by requested group and/or year.',
        format_requirements=['explicitly call out missing evidence by requested group and/or year'],
    )
    check = _evaluate_output_contract(
        answer=(
            '## Matrix\n'
            '| Year | Gaps |\n'
            '| --- | --- |\n'
            '| 2024 | Missing Evidence: insurance document not indexed. |\n'
        ),
        plan=plan,
    )
    assert check['missing_evidence_callout_ok'] is True
    assert check['missing_evidence_callout_canonical'] is True


def test_strict_output_contract_missing_evidence_accepts_bold_prefix_marker() -> None:
    plan = _build_output_contract_plan(
        question='Call out missing evidence explicitly.',
        format_requirements=['explicitly call out missing evidence by requested group and/or year'],
    )
    check = _evaluate_output_contract(
        answer='- **Missing Evidence**: no verified contradiction record for 2022.',
        plan=plan,
    )
    assert check['missing_evidence_callout_ok'] is True
    assert check['missing_evidence_callout_canonical'] is True


def test_strict_output_contract_evidence_accepts_filename_page_shorthand() -> None:
    plan = _build_output_contract_plan(
        question='Provide an evidence-grounded source inventory.',
        format_requirements=[
            (
                'require canonical evidence grounding for every claim-bearing bullet/list item '
                'and narrative claim paragraph'
            ),
        ],
    )
    check = _evaluate_output_contract(
        answer='- 2022-2023 Secured Tax Bill.pdf (Page 1)',
        plan=plan,
    )
    assert check['evidence_grounding_ok'] is True
    assert check['passed'] is True


def test_strict_output_contract_evidence_accepts_filename_section_shorthand() -> None:
    plan = _build_output_contract_plan(
        question='Provide an evidence-grounded source inventory.',
        format_requirements=[
            (
                'require canonical evidence grounding for every claim-bearing bullet/list item '
                'and narrative claim paragraph'
            ),
        ],
    )
    check = _evaluate_output_contract(
        answer='- 2023 Paperwork Lady - Intake Form, etc.pdf (Section: MUST READ, SIGN & DATE)',
        plan=plan,
    )
    assert check['evidence_grounding_ok'] is True
    assert check['passed'] is True


def test_strict_output_contract_evidence_ignores_no_contradictions_placeholder() -> None:
    plan = _build_output_contract_plan(
        question='Keep all claims evidence-grounded.',
        format_requirements=[
            (
                'require canonical evidence grounding for every claim-bearing bullet/list item '
                'and narrative claim paragraph'
            ),
        ],
    )
    check = _evaluate_output_contract(
        answer=(
            '- Verified amount is $120. Evidence: ledger.pdf, page 2\n'
            '- No contradictions found in the 2023 documents.\n'
        ),
        plan=plan,
    )
    assert check['evidence_claim_block_count'] == 1
    assert check['evidence_grounding_ok'] is True
    assert check['passed'] is True


def test_strict_output_contract_placeholder_only_fallback_satisfies_evidence_grounding() -> None:
    plan = _build_output_contract_plan(
        question='Output exactly 4 top-level bullets and keep claims evidence-grounded.',
        format_requirements=[],
    )
    check = _evaluate_output_contract(
        answer=(
            '- Conflict Statement: Not found; Involved Documents: Not found; '
            'Conflicting Values: Not found; Likely Reason: Not found; '
            'Missing Evidence: no comparable source pair found.\n'
            '- Conflict Statement: Not found; Involved Documents: Not found; '
            'Conflicting Values: Not found; Likely Reason: Not found; '
            'Missing Evidence: no comparable source pair found.\n'
            '- Conflict Statement: Not found; Involved Documents: Not found; '
            'Conflicting Values: Not found; Likely Reason: Not found; '
            'Missing Evidence: no comparable source pair found.\n'
            '- Conflict Statement: Not found; Involved Documents: Not found; '
            'Conflicting Values: Not found; Likely Reason: Not found; '
            'Missing Evidence: no comparable source pair found.\n'
        ),
        plan=plan,
    )
    assert check['top_level_bullet_count_ok'] is True
    assert check['evidence_grounding_ok'] is True
    assert check['evidence_note'] == 'placeholder_only_fallback'
    assert check['passed'] is True


def test_strict_output_contract_evidence_ignores_bold_prefixed_no_contradictions_placeholder() -> None:
    plan = _build_output_contract_plan(
        question='Keep all claims evidence-grounded.',
        format_requirements=[
            (
                'require canonical evidence grounding for every claim-bearing bullet/list item '
                'and narrative claim paragraph'
            ),
        ],
    )
    check = _evaluate_output_contract(
        answer=(
            '- Verified amount is $120. Evidence: ledger.pdf, page 2\n'
            '**Contradictions:** - No contradictions found in the 2024 documents.\n'
        ),
        plan=plan,
    )
    assert check['evidence_claim_block_count'] == 1
    assert check['evidence_grounding_ok'] is True
    assert check['passed'] is True


def test_strict_output_contract_evidence_ignores_multiline_no_contradictions_placeholder() -> None:
    plan = _build_output_contract_plan(
        question='Keep all claims evidence-grounded.',
        format_requirements=[
            (
                'require canonical evidence grounding for every claim-bearing bullet/list item '
                'and narrative claim paragraph'
            ),
        ],
    )
    check = _evaluate_output_contract(
        answer=(
            '- Verified amount is $120. Evidence: ledger.pdf, page 2\n\n'
            '**Contradictions:**\n'
            '- No contradictions found in the 2024 documents.\n'
        ),
        plan=plan,
    )
    assert check['evidence_claim_block_count'] == 1
    assert check['evidence_grounding_ok'] is True
    assert check['passed'] is True


def test_strict_output_contract_excludes_bullets_under_structural_sections() -> None:
    plan = _build_output_contract_plan(
        question='Required headings in exact order: ## Method, ## Findings. Keep all claims evidence-grounded.',
        format_requirements=[],
    )
    check = _evaluate_output_contract(
        answer=(
            '## Method\n'
            '- Identify relevant documents.\n'
            '- Compare values by year.\n\n'
            '## Findings\n'
            '- Verified amount is $120. Evidence: ledger.pdf, page 2\n'
        ),
        plan=plan,
    )
    assert check['evidence_claim_block_count'] == 1
    assert check['evidence_grounding_ok'] is True
    assert check['passed'] is True


def test_strict_output_contract_ignores_structural_subheading_bullets_with_missing_evidence_marker() -> None:
    plan = _build_output_contract_plan(
        question='Keep all claims evidence-grounded.',
        format_requirements=[
            (
                'require canonical evidence grounding for every claim-bearing bullet/list item '
                'and narrative claim paragraph'
            ),
        ],
    )
    check = _evaluate_output_contract(
        answer=(
            '- **Extracted Amounts**:\n'
            '- **Missing Evidence**: no evidence was found for 2022 tax returns.\n'
            '- Verified amount is $120. Evidence: ledger.pdf, page 2\n'
        ),
        plan=plan,
    )
    assert check['evidence_claim_block_count'] == 1
    assert check['evidence_grounding_ok'] is True
    assert check['passed'] is True


def test_strict_output_contract_flags_uncited_no_contradictions_placeholder() -> None:
    plan = _build_output_contract_plan(
        question=(
            'Required headings in exact order: ## Executive Summary, ## Contradictions and Gaps. '
            'Keep all claims evidence-grounded.'
        ),
        format_requirements=[],
    )
    check = _evaluate_output_contract(
        answer=(
            '## Executive Summary\n'
            'Overview text.\n\n'
            '## Contradictions and Gaps\n'
            '**Contradictions:** - No contradictions found in the 2023 documents.\n'
        ),
        plan=plan,
    )
    assert check['contradiction_placeholder_ok'] is False
    assert check['passed'] is False


def test_strict_output_contract_flags_uncited_largest_delta_numeric_line() -> None:
    plan = _build_output_contract_plan(
        question=(
            'Required headings in exact order: ## Scope, ## Largest Increase, ## Largest Decrease. '
            'Keep all claims evidence-grounded.'
        ),
        format_requirements=[],
    )
    check = _evaluate_output_contract(
        answer=(
            '## Scope\n'
            'Overview of indexed records.\n\n'
            '## Largest Increase\n'
            '- Largest increase is $660,109 due to reassessment.\n\n'
            '## Largest Decrease\n'
            '- Largest decrease is $120,000. Evidence: tax.pdf, page 2\n'
        ),
        plan=plan,
    )
    assert check['uncited_delta_numeric_ok'] is False
    assert check['passed'] is False


def test_strict_output_contract_evaluation_handles_double_prefixed_markdown_headings() -> None:
    plan = _build_output_contract_plan(
        question='Required headings in exact order: ## Scope, ### 2022, ### 2023.',
        format_requirements=[],
    )
    check = _evaluate_output_contract(
        answer='## Scope\nA\n\n### ### 2022\nB\n\n### ### 2023\nC',
        plan=plan,
    )
    assert check['passed'] is True


def test_strict_output_contract_section_scoped_bullet_count_only_checks_target_section() -> None:
    plan = _build_output_contract_plan(
        question=(
            'Under ## Missing Evidence include exactly 2 bullets. '
            'Use other bullets in other sections as needed.'
        ),
        format_requirements=[],
    )
    check = _evaluate_output_contract(
        answer=(
            '## Coverage Snapshot\n'
            '- context bullet outside target section\n'
            '\n'
            '## Missing Evidence\n'
            '- missing item A\n'
            '- missing item B\n'
            '\n'
            '## Follow-up Plan\n'
            '- next action outside target section\n'
        ),
        plan=plan,
    )
    assert check['top_level_bullet_count_ok'] is True
    assert check['top_level_bullet_count'] == 2


def test_strict_output_contract_normalizes_annotated_headings_without_global_word_cap() -> None:
    plan = _build_output_contract_plan(
        question=(
            'Create sections in order: 1) Executive Summary (max 140 words), '
            '2) Year-by-Year Evidence Map (2022, 2023, 2024), 3) Action Checklist.'
        ),
        format_requirements=[
            'use the required headings exactly and in the requested order',
            'include heading: Executive Summary (max 140 words)',
            'include heading: Year-by-Year Evidence Map (2022, 2023, 2024)',
            'include heading: Action Checklist',
        ],
    )
    check = _evaluate_output_contract(
        answer='## Executive Summary\nA\n\n## Year-by-Year Evidence Map\nB\n\n## Action Checklist\nC',
        plan=plan,
    )
    assert check['passed'] is True
    assert check['max_words'] is None


def test_strict_output_contract_normalizes_trailing_group_list_annotation() -> None:
    plan = _build_output_contract_plan(
        question=(
            'Create sections in order: '
            '1) Document Group Deep Dive (Group A records, Group B records, Group C records, authority confirmations), '
            '2) Action Checklist.'
        ),
        format_requirements=[
            'use the required headings exactly and in the requested order',
            'include heading: Document Group Deep Dive (Group A records, Group B records, Group C records, authority confirmations)',
            'include heading: Action Checklist',
        ],
    )
    check = _evaluate_output_contract(
        answer='## Document Group Deep Dive\n- details\n\n## Action Checklist\n- next',
        plan=plan,
    )
    assert check['passed'] is True


def test_strict_output_contract_normalizes_inline_directive_tail() -> None:
    plan = _build_output_contract_plan(
        question=(
            'Output exactly 3 sections: ## Biggest Increase, ## Biggest Decrease, '
            '## Ambiguous Delta. In each section, cite evidence only from indexed documents.'
        ),
        format_requirements=[],
    )
    check = _evaluate_output_contract(
        answer=(
            '## Biggest Increase\n'
            'The strongest increase appears in a documented recurring charge category.\n'
            'Evidence: report-2024.pdf, page 3\n\n'
            '## Biggest Decrease\n'
            'The largest decrease is tied to a documented policy adjustment in 2023.\n'
            'Evidence: report-2023.pdf, page 5\n\n'
            '## Ambiguous Delta\n'
            'One delta remains ambiguous because values conflict across two source extracts.\n'
            'Evidence: reconciliation-notes.md'
        ),
        plan=plan,
    )
    assert check['missing_headings'] == []
    assert check['required_headings'] == ['## Biggest Decrease', '## Ambiguous Delta']


def test_strict_output_contract_normalizes_trailing_only_qualifier() -> None:
    plan = _build_output_contract_plan(
        question=(
            'Required headings in exact order: ## Cross-Year Deltas, '
            '## Confidence Notes, ## Next Verification Steps only.'
        ),
        format_requirements=[],
    )
    check = _evaluate_output_contract(
        answer='## Cross-Year Deltas\nA\n\n## Confidence Notes\nB\n\n## Next Verification Steps\nC',
        plan=plan,
    )
    assert check['passed'] is True


def test_strict_output_contract_plan_strips_instruction_tails_from_explicit_heading_order() -> None:
    plan = _build_output_contract_plan(
        question=(
            'Required headings in order: ## Scope, ## Method, ## Findings by Year, '
            '## Cross-Year Deltas, ## Confidence Notes, ## Next Verification Steps. '
            'Under ## Findings by Year include exactly three subsections: ### 2022, ### 2023, ### 2024.'
        ),
        format_requirements=[],
    )
    assert plan.required_headings == (
        '## Scope',
        '## Method',
        '## Findings by Year',
        '## Cross-Year Deltas',
        '## Confidence Notes',
        '## Next Verification Steps',
    )
