from informity.llm.rag_runtime.generation_runtime import (
    _apply_post_retrieval_budget_degradations,
    _apply_preflight_budget_degradations,
    _apply_source_scoped_coverage_guard,
    _apply_strict_format_prompt_controls,
    _apply_strict_ordered_output_budget,
    _apply_strict_pre_retrieval_guard,
    _augment_strict_ordered_format_requirements,
    _estimate_budget_ratio,
    _estimate_tokens_per_second,
    _has_remaining_scope,
    _should_apply_soft_stream_closeout,
)


def test_generation_runtime_disables_soft_closeout_for_strict_order() -> None:
    assert _should_apply_soft_stream_closeout(
        ['use the required headings exactly and in the requested order']
    ) is False
    assert _should_apply_soft_stream_closeout(['include heading: Scope']) is True


def test_generation_runtime_strict_budget_caps_and_disables_reasoning() -> None:
    constraints, max_tokens, reasoning_enabled, degradation = _apply_strict_ordered_output_budget(
        format_requirements=[
            'use the required headings exactly and in the requested order',
            'include heading: Scope',
            'include heading: Findings',
        ],
        query_type='coverage',
        output_constraints={},
        max_tokens=1536,
        reasoning_enabled=True,
    )
    assert constraints.get('max_words') == 420
    assert constraints.get('max_rows') == 18
    assert max_tokens >= 1536
    assert reasoning_enabled is False
    assert degradation is not None
    assert degradation.get('step') == 'strict_ordered_section_budget'


def test_generation_runtime_research_mode_skips_strict_ordered_short_caps() -> None:
    constraints, max_tokens, reasoning_enabled, degradation = _apply_strict_ordered_output_budget(
        format_requirements=[
            'use the required headings exactly and in the requested order',
            'include heading: Scope',
            'include heading: Findings',
        ],
        query_type='coverage',
        output_constraints={},
        max_tokens=4096,
        reasoning_enabled=True,
        response_mode='research',
    )
    assert constraints == {}
    assert max_tokens == 4096
    assert reasoning_enabled is True
    assert degradation is None


def test_generation_runtime_augment_adds_three_level_example() -> None:
    requirements = _augment_strict_ordered_format_requirements([
        'use the required headings exactly and in the requested order',
        'use nested bullet lists with exactly 3 levels where requested',
    ])
    assert any('3-level chain' in requirement for requirement in requirements)
    assert any('Parent\\n  - Child\\n    - Grandchild' in requirement for requirement in requirements)


def test_generation_runtime_budget_ratio_is_positive() -> None:
    projected_seconds, ratio = _estimate_budget_ratio(
        profile_name='Qwen3-14B-Q5_K_M',
        query_type='coverage',
        timeout_seconds=320,
        question_length=300,
        context_chunks=12,
        context_chars=14000,
        top_k=12,
        reasoning_enabled=False,
        max_tokens=1200,
    )
    assert projected_seconds > 0.0
    assert ratio > 0.0


def test_generation_runtime_tokens_per_second_uses_profile_metadata() -> None:
    assert _estimate_tokens_per_second('Qwen3 30B A3B') == 6.0
    assert _estimate_tokens_per_second('Qwen3 14B') == 9.0
    assert _estimate_tokens_per_second('Unknown profile name') == 12.0


def test_generation_runtime_has_remaining_scope_detects_degradation() -> None:
    assert _has_remaining_scope(
        timeout_reason=None,
        stream_recovery_reason=None,
        generation_skipped=False,
        applied_degradations=[{'step': 'reduce_top_k'}],
    ) is True
    assert _has_remaining_scope(
        timeout_reason=None,
        stream_recovery_reason=None,
        generation_skipped=False,
        applied_degradations=[],
    ) is False


def test_generation_runtime_apply_strict_prompt_controls_caps_context() -> None:
    def _derive(_question: str) -> list[str]:
        return [
            'use the required headings exactly and in the requested order',
            'include heading: Executive Summary',
        ]

    format_requirements, constraints, max_tokens, reasoning_enabled, chunks, degradations = (
        _apply_strict_format_prompt_controls(
            question='structured compliance brief',
            chunks=[{'chunk_text': 'x'} for _ in range(14)],
            query_type='coverage',
            output_constraints={},
            max_tokens=1536,
            reasoning_enabled=True,
            response_mode='balanced',
            derive_format_requirements_fn=_derive,
            applied_degradations=[],
        )
    )

    assert any('requested order' in item for item in format_requirements)
    assert len(chunks) == 8
    assert constraints.get('max_words') == 420
    assert max_tokens >= 1536
    assert reasoning_enabled is False
    assert any(item.get('step') == 'strict_ordered_context_cap' for item in degradations)


def test_generation_runtime_apply_strict_prompt_controls_respects_diagnostics_floor() -> None:
    def _derive(_question: str) -> list[str]:
        return [
            'use the required headings exactly and in the requested order',
            'include heading: Executive Summary',
        ]

    _requirements, constraints, max_tokens, _reasoning_enabled, _chunks, degradations = (
        _apply_strict_format_prompt_controls(
            question='structured compliance brief',
            chunks=[{'chunk_text': 'x'} for _ in range(6)],
            query_type='coverage',
            output_constraints={},
            max_tokens=1000,
            reasoning_enabled=False,
            response_mode='balanced',
            derive_format_requirements_fn=_derive,
            applied_degradations=[],
            min_output_budget_floor=900,
        )
    )

    assert constraints.get('max_words') == 900
    assert max_tokens >= 1300
    assert any(item.get('step') == 'diagnostics_min_output_budget_floor' for item in degradations)
    assert any(item.get('step') == 'diagnostics_min_token_floor' for item in degradations)


def test_generation_runtime_strict_pre_retrieval_guard_applies_caps() -> None:
    def _derive(_question: str) -> list[str]:
        return [
            'use the required headings exactly and in the requested order',
            'include heading: Scope',
            'include heading: Findings',
        ]

    timeout_seconds, top_k, reasoning_enabled, max_tokens, degradations, strict_ordered_mode = (
        _apply_strict_pre_retrieval_guard(
            question='structured policy request',
            query_type='coverage',
            timeout_seconds=320,
            top_k=20,
            reasoning_enabled=True,
            max_tokens=1536,
            applied_degradations=[],
            derive_format_requirements_fn=_derive,
            profile_name='Qwen3-14B-Q5_K_M',
        )
    )

    assert strict_ordered_mode is True
    # With default response_mode='analysis', amplification applies:
    # timeout cap = min(210, int(75 * 1.45)) = 108; top_k cap = min(14, 8+2) = 10;
    # timeout_aware_max_tokens floor = min(2180, 720+180) = 900
    assert timeout_seconds == 108
    assert top_k == 10
    assert reasoning_enabled is False
    assert max_tokens == 900
    assert any(item.get('step') == 'strict_pre_retrieval_top_k_cap' for item in degradations)
    assert any(item.get('step') == 'strict_pre_retrieval_timeout_cap' for item in degradations)
    assert any(item.get('step') == 'strict_pre_retrieval_disable_reasoning' for item in degradations)
    assert any(item.get('step') == 'strict_pre_retrieval_timeout_aware_max_tokens_cap' for item in degradations)


def test_generation_runtime_strict_pre_retrieval_guard_noop_without_strict_order() -> None:
    def _derive(_question: str) -> list[str]:
        return ['include heading: Scope']

    timeout_seconds, top_k, reasoning_enabled, max_tokens, degradations, strict_ordered_mode = (
        _apply_strict_pre_retrieval_guard(
            question='simple request',
            query_type='coverage',
            timeout_seconds=320,
            top_k=20,
            reasoning_enabled=True,
            max_tokens=1536,
            applied_degradations=[],
            derive_format_requirements_fn=_derive,
            profile_name='Qwen3-14B-Q5_K_M',
        )
    )

    assert strict_ordered_mode is False
    assert timeout_seconds == 320
    assert top_k == 20
    assert reasoning_enabled is True
    assert max_tokens == 1536
    assert degradations == []


def test_generation_runtime_strict_pre_retrieval_guard_relaxes_caps_for_complex_contracts() -> None:
    def _derive(_question: str) -> list[str]:
        return [
            'use the required headings exactly and in the requested order',
            'include heading: ## Scope',
            'include heading: ## Method',
            'include heading: ## Findings by Year',
            'include heading: ## Cross-Year Deltas',
            'include heading: ## Confidence Notes',
            'include heading: ## Next Verification Steps',
            'use nested bullet lists with exactly 3 levels where requested',
            'explicitly call out missing evidence by requested group and/or year',
        ]

    timeout_seconds, top_k, reasoning_enabled, max_tokens, degradations, strict_ordered_mode = (
        _apply_strict_pre_retrieval_guard(
            question='strict multi-section forensic report',
            query_type='coverage',
            timeout_seconds=320,
            top_k=20,
            reasoning_enabled=True,
            max_tokens=2150,
            applied_degradations=[],
            derive_format_requirements_fn=_derive,
            profile_name='Qwen3-14B-Q5_K_M',
        )
    )

    assert strict_ordered_mode is True
    # With default response_mode='analysis', amplification applies:
    # timeout cap = min(210, int(120 * 1.45)) = 174; top_k cap = min(14, 10+2) = 12;
    # timeout_aware_max_tokens = int((174 * 0.68 * 9.0) / 0.78) = 1365
    assert timeout_seconds == 174
    assert top_k == 12
    assert reasoning_enabled is False
    assert max_tokens == 1365
    assert any(item.get('step') == 'strict_pre_retrieval_top_k_cap' for item in degradations)
    assert any(item.get('step') == 'strict_pre_retrieval_timeout_cap' for item in degradations)
    assert any(item.get('step') == 'strict_pre_retrieval_disable_reasoning' for item in degradations)
    assert any(item.get('step') == 'strict_pre_retrieval_timeout_aware_max_tokens_cap' for item in degradations)


def test_generation_runtime_strict_pre_retrieval_guard_research_keeps_budget() -> None:
    def _derive(_question: str) -> list[str]:
        return [
            'use the required headings exactly and in the requested order',
            'include heading: ## Scope',
            'include heading: ## Method',
            'include heading: ## Findings by Year',
            'include heading: ## Cross-Year Deltas',
            'include heading: ## Confidence Notes',
            'include heading: ## Next Verification Steps',
        ]

    timeout_seconds, top_k, reasoning_enabled, max_tokens, degradations, strict_ordered_mode = (
        _apply_strict_pre_retrieval_guard(
            question='strict multi-section forensic report',
            query_type='coverage',
            timeout_seconds=900,
            top_k=20,
            reasoning_enabled=True,
            max_tokens=8192,
            applied_degradations=[],
            derive_format_requirements_fn=_derive,
            profile_name='Qwen3-30B-A3B-Q5_K_M',
            response_mode='research',
        )
    )

    assert strict_ordered_mode is True
    assert timeout_seconds == 900
    assert top_k == 20
    assert reasoning_enabled is True
    assert max_tokens == 8192
    assert degradations == []


def test_generation_runtime_preflight_degradation_applies_expected_steps() -> None:
    (
        query_type,
        top_k,
        reasoning_enabled,
        max_tokens,
        timeout_seconds,
        output_constraints,
        degradations,
        projected_seconds,
        ratio,
    ) = _apply_preflight_budget_degradations(
        fit_to_budget_enabled=True,
        policy_soft_top_k_threshold=0.0,
        policy_soft_reasoning_threshold=0.0,
        policy_soft_output_cap_threshold=0.0,
        policy_soft_coverage_to_focused_threshold=999.0,
        profile_name='Qwen3-14B-Q5_K_M',
        question_length=450,
        query_type='coverage',
        timeout_seconds=320,
        top_k=20,
        reasoning_enabled=True,
        max_tokens=1536,
        subtype='aggregate_by_period',
        focused_max_tokens=1280,
        focused_timeout_seconds=240,
        output_constraints={},
        applied_degradations=[],
    )

    assert query_type == 'coverage'
    assert top_k <= 20
    assert reasoning_enabled is False
    assert max_tokens <= 1800
    assert timeout_seconds == 320
    assert output_constraints.get('max_words') == 900
    assert any(item.get('step') == 'reduce_top_k' for item in degradations)
    assert any(item.get('step') == 'disable_reasoning' for item in degradations)
    assert any(item.get('step') == 'cap_output_structure' for item in degradations)
    assert projected_seconds > 0.0
    assert ratio > 0.0


def test_generation_runtime_strict_prompt_controls_caps_context_chars() -> None:
    def _derive(_question: str) -> list[str]:
        return [
            'use the required headings exactly and in the requested order',
            'include heading: Scope',
        ]

    _, _, _, _, chunks, degradations = _apply_strict_format_prompt_controls(
        question='strict report',
        chunks=[{'chunk_text': 'a' * 1500} for _ in range(6)],
        query_type='coverage',
        output_constraints={},
        max_tokens=900,
        reasoning_enabled=False,
        response_mode='balanced',
        derive_format_requirements_fn=_derive,
        applied_degradations=[],
    )

    total_chars = sum(len(str(chunk.get('chunk_text', ''))) for chunk in chunks)
    assert total_chars <= 3200
    assert all(len(str(chunk.get('chunk_text', ''))) <= 900 for chunk in chunks)
    assert any(item.get('step') == 'strict_ordered_context_chars_cap' for item in degradations)


def test_generation_runtime_strict_prompt_controls_use_complex_context_caps() -> None:
    def _derive(_question: str) -> list[str]:
        return [
            'use the required headings exactly and in the requested order',
            'include heading: ## Scope',
            'include heading: ## Method',
            'include heading: ## Findings by Year',
            'include heading: ## Cross-Year Deltas',
            'include heading: ## Confidence Notes',
            'include heading: ## Next Verification Steps',
            'use nested bullet lists with exactly 3 levels where requested',
        ]

    _, constraints, _, _, chunks, degradations = _apply_strict_format_prompt_controls(
        question='strict complex report',
        chunks=[{'chunk_text': 'a' * 1500} for _ in range(8)],
        query_type='coverage',
        output_constraints={},
        max_tokens=900,
        reasoning_enabled=False,
        response_mode='balanced',
        derive_format_requirements_fn=_derive,
        applied_degradations=[],
    )


def test_generation_runtime_strict_pre_retrieval_guard_relaxes_for_analysis_mode() -> None:
    def _derive(_question: str) -> list[str]:
        return [
            'use the required headings exactly and in the requested order',
            'include heading: ## Scope',
            'include heading: ## Method',
            'include heading: ## Findings by Year',
            'include heading: ## Cross-Year Deltas',
            'include heading: ## Confidence Notes',
            'include heading: ## Next Verification Steps',
            'use nested bullet lists with exactly 3 levels where requested',
            'explicitly call out missing evidence by requested group and/or year',
        ]

    timeout_seconds, top_k, reasoning_enabled, max_tokens, _degradations, strict_ordered_mode = (
        _apply_strict_pre_retrieval_guard(
            question='strict multi-section forensic report',
            query_type='coverage',
            timeout_seconds=320,
            top_k=20,
            reasoning_enabled=True,
            max_tokens=2150,
            applied_degradations=[],
            derive_format_requirements_fn=_derive,
            profile_name='Qwen3-14B-Q5_K_M',
            response_mode='analysis',
        )
    )

    assert strict_ordered_mode is True
    assert timeout_seconds > 120
    assert top_k > 10
    assert reasoning_enabled is False
    assert max_tokens > 886


def test_generation_runtime_post_retrieval_degradation_reduces_context() -> None:
    chunks = [{'chunk_text': f'evidence {i}'} for i in range(16)]
    (
        degraded_chunks,
        query_type,
        top_k,
        reasoning_enabled,
        max_tokens,
        timeout_seconds,
        context_chars,
        degradations,
        projected_seconds,
        ratio,
    ) = _apply_post_retrieval_budget_degradations(
        fit_to_budget_enabled=True,
        policy_soft_top_k_threshold=0.0,
        policy_soft_coverage_to_focused_threshold=999.0,
        profile_name='Qwen3-14B-Q5_K_M',
        question_length=450,
        query_type='coverage',
        timeout_seconds=320,
        top_k=16,
        reasoning_enabled=False,
        max_tokens=1200,
        chunks=chunks,
        subtype='aggregate_by_period',
        focused_max_tokens=1280,
        focused_timeout_seconds=240,
        applied_degradations=[],
    )

    assert len(degraded_chunks) < len(chunks)
    assert query_type == 'coverage'
    assert top_k <= 16
    assert reasoning_enabled is False
    assert max_tokens == 1200
    assert timeout_seconds == 320
    assert context_chars > 0
    assert any(item.get('step') == 'reduce_context_chunks' for item in degradations)
    assert projected_seconds > 0.0
    assert ratio > 0.0


def test_generation_runtime_post_retrieval_skips_chunk_reduction_for_diagnostics_depth() -> None:
    chunks = [{'chunk_text': 'x' * 2000} for _ in range(10)]
    (
        effective_chunks,
        _query_type,
        _top_k,
        _reasoning_enabled,
        _max_tokens,
        _timeout_seconds,
        _context_chars,
        degradations,
        _projected_seconds,
        _ratio,
    ) = _apply_post_retrieval_budget_degradations(
        fit_to_budget_enabled=True,
        policy_soft_top_k_threshold=0.2,
        policy_soft_coverage_to_focused_threshold=0.95,
        profile_name='Qwen3-14B-Q5_K_M',
        question_length=220,
        query_type='coverage',
        timeout_seconds=120,
        top_k=10,
        reasoning_enabled=False,
        max_tokens=1800,
        chunks=chunks,
        subtype=None,
        focused_max_tokens=800,
        focused_timeout_seconds=60,
        applied_degradations=[],
        min_output_budget_floor=900,
    )

    assert len(effective_chunks) >= 8
    assert not any(item.get('step') == 'reduce_context_chunks' for item in degradations)
    assert any(item.get('step') == 'diagnostics_depth_constraints_applied' for item in degradations)


def test_generation_runtime_preflight_applies_focused_guard_thresholds() -> None:
    (
        query_type,
        top_k,
        reasoning_enabled,
        max_tokens,
        _timeout_seconds,
        output_constraints,
        degradations,
        _projected_seconds,
        ratio,
    ) = _apply_preflight_budget_degradations(
        fit_to_budget_enabled=True,
        policy_soft_top_k_threshold=0.99,
        policy_soft_reasoning_threshold=0.99,
        policy_soft_output_cap_threshold=0.99,
        policy_soft_coverage_to_focused_threshold=0.99,
        profile_name='Qwen3-14B-Q5_K_M',
        question_length=900,
        query_type='focused',
        timeout_seconds=180,
        top_k=10,
        reasoning_enabled=True,
        max_tokens=2048,
        subtype=None,
        focused_max_tokens=2048,
        focused_timeout_seconds=180,
        output_constraints={},
        applied_degradations=[],
    )

    assert query_type == 'focused'
    assert top_k <= 10
    assert reasoning_enabled is False
    assert max_tokens <= 900
    assert output_constraints.get('max_words') == 420
    assert any(
        item.get('step') == 'disable_reasoning'
        and item.get('reason') == 'preflight_ratio_high_focused_guard'
        for item in degradations
    )
    assert any(
        item.get('step') == 'cap_output_structure'
        and item.get('reason') == 'preflight_ratio_high_focused_guard'
        for item in degradations
    )


def test_generation_runtime_preflight_research_uses_deeper_output_cap() -> None:
    (
        query_type,
        top_k,
        reasoning_enabled,
        max_tokens,
        timeout_seconds,
        output_constraints,
        degradations,
        projected_seconds,
        ratio,
    ) = _apply_preflight_budget_degradations(
        fit_to_budget_enabled=True,
        policy_soft_top_k_threshold=0.0,
        policy_soft_reasoning_threshold=0.0,
        policy_soft_output_cap_threshold=0.0,
        policy_soft_coverage_to_focused_threshold=999.0,
        profile_name='Qwen3-30B-A3B-Q5_K_M',
        question_length=900,
        query_type='coverage',
        timeout_seconds=900,
        top_k=20,
        reasoning_enabled=True,
        max_tokens=8192,
        subtype='aggregate_by_period',
        focused_max_tokens=3072,
        focused_timeout_seconds=450,
        output_constraints={},
        applied_degradations=[],
        response_mode='research',
    )

    assert query_type == 'coverage'
    assert top_k <= 20
    assert reasoning_enabled is False
    assert max_tokens <= 4096
    assert max_tokens > 1200
    assert timeout_seconds == 900
    assert output_constraints.get('max_words') == 2200
    assert any(item.get('step') == 'cap_output_structure' for item in degradations)
    assert projected_seconds > 0.0
    assert ratio > 0.0


def test_generation_runtime_preflight_omits_max_words_cap_for_strict_ordered_mode() -> None:
    (
        query_type,
        top_k,
        reasoning_enabled,
        max_tokens,
        timeout_seconds,
        output_constraints,
        degradations,
        projected_seconds,
        ratio,
    ) = _apply_preflight_budget_degradations(
        fit_to_budget_enabled=True,
        policy_soft_top_k_threshold=0.0,
        policy_soft_reasoning_threshold=0.0,
        policy_soft_output_cap_threshold=0.0,
        policy_soft_coverage_to_focused_threshold=999.0,
        profile_name='Qwen3-14B-Q5_K_M',
        question_length=900,
        query_type='coverage',
        timeout_seconds=320,
        top_k=20,
        reasoning_enabled=True,
        max_tokens=1536,
        subtype='aggregate_by_period',
        focused_max_tokens=1280,
        focused_timeout_seconds=240,
        output_constraints={},
        applied_degradations=[],
        response_mode='analysis',
        strict_ordered_mode=True,
    )

    assert query_type == 'coverage'
    assert top_k <= 20
    assert reasoning_enabled is False
    assert max_tokens <= 1800
    assert timeout_seconds == 320
    assert 'max_words' not in output_constraints
    assert output_constraints.get('max_rows') == 30
    assert any(item.get('step') == 'cap_output_structure' for item in degradations)
    assert projected_seconds > 0.0
    assert ratio > 0.0


def test_generation_runtime_post_retrieval_caps_focused_context_chars() -> None:
    (
        degraded_chunks,
        query_type,
        top_k,
        _reasoning_enabled,
        _max_tokens,
        _timeout_seconds,
        context_chars,
        degradations,
        _projected_seconds,
        _ratio,
    ) = _apply_post_retrieval_budget_degradations(
        fit_to_budget_enabled=True,
        policy_soft_top_k_threshold=0.99,
        policy_soft_coverage_to_focused_threshold=0.99,
        profile_name='Qwen3-14B-Q5_K_M',
        question_length=120,
        query_type='focused',
        timeout_seconds=240,
        top_k=10,
        reasoning_enabled=False,
        max_tokens=900,
        chunks=[{'chunk_text': 'x' * 2200} for _ in range(8)],
        subtype=None,
        focused_max_tokens=1280,
        focused_timeout_seconds=240,
        applied_degradations=[],
    )

    assert query_type == 'focused'
    assert top_k <= 8
    assert context_chars <= 5200
    assert all(len(str(chunk.get('chunk_text', ''))) <= 1200 for chunk in degraded_chunks)
    assert any(item.get('step') == 'focused_context_chars_cap' for item in degradations)


def test_generation_runtime_post_retrieval_caps_coverage_prefill_context_chars() -> None:
    (
        degraded_chunks,
        query_type,
        top_k,
        _reasoning_enabled,
        _max_tokens,
        _timeout_seconds,
        context_chars,
        degradations,
        _projected_seconds,
        _ratio,
    ) = _apply_post_retrieval_budget_degradations(
        fit_to_budget_enabled=True,
        policy_soft_top_k_threshold=0.99,
        policy_soft_coverage_to_focused_threshold=0.99,
        profile_name='Qwen3-14B-Q5_K_M',
        question_length=220,
        query_type='coverage',
        timeout_seconds=320,
        top_k=20,
        reasoning_enabled=False,
        max_tokens=1200,
        chunks=[{'chunk_text': 'x' * 3200} for _ in range(10)],
        subtype=None,
        focused_max_tokens=1280,
        focused_timeout_seconds=240,
        applied_degradations=[],
    )

    assert query_type == 'coverage'
    assert top_k <= 10
    assert context_chars <= 20000
    assert all(len(str(chunk.get('chunk_text', ''))) <= 2200 for chunk in degraded_chunks)
    assert any(item.get('step') == 'coverage_prefill_context_chars_cap' for item in degradations)


def test_generation_runtime_source_scoped_coverage_guard_caps_budget() -> None:
    timeout_seconds, top_k, reasoning_enabled, max_tokens, degradations = (
        _apply_source_scoped_coverage_guard(
            query_type='coverage',
            route_candidate='cross_document_synthesis',
            source_terms=['FormA'],
            timeout_seconds=320,
            top_k=15,
            reasoning_enabled=True,
            max_tokens=1536,
            applied_degradations=[],
        )
    )

    assert timeout_seconds == 220
    assert top_k == 14
    assert reasoning_enabled is True
    assert max_tokens == 1536
    assert any(item.get('step') == 'source_scoped_coverage_top_k_cap' for item in degradations)
    assert any(item.get('step') == 'source_scoped_coverage_timeout_cap' for item in degradations)
    assert not any(item.get('step') == 'source_scoped_coverage_max_tokens_cap' for item in degradations)
    assert not any(item.get('step') == 'source_scoped_coverage_disable_reasoning' for item in degradations)
