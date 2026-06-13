from __future__ import annotations

from dataclasses import dataclass

from informity.api.schemas import SetupTierOption
from informity.llm.model_adapter import (
    MODEL_ID_QWEN_9B,
    MODEL_ID_QWEN_14B,
    MODEL_ID_QWEN_35B_A3B,
    get_model_display_name,
)

_DECIMAL_GB = 1_000_000_000

_MODEL_SIZE_BYTES: dict[str, int] = {
    'Qwen_Qwen3.5-9B-Q4_K_M.gguf': 5_889_811_552,
    'Qwen3-14B-Q5_K_M.gguf': 10_514_569_568,
    'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf': 22_134_528_992,
}

@dataclass(frozen=True)
class ModelReleaseOption:
    tier: str
    model_id: str
    title: str
    display_name: str
    model_filename: str
    model_size_bytes: int
    approx_size_gb: float
    quality: str
    speed: str
    ram_profile: str
    description: str
    release_label: str  # Recommended | Legacy
    downloadable: bool = True


def _build_release(
    *,
    tier: str,
    model_id: str,
    title: str,
    display_name: str,
    model_filename: str,
    model_size_bytes: int,
    quality: str,
    speed: str,
    ram_profile: str,
    description: str,
    release_label: str,
    downloadable: bool = True,
) -> ModelReleaseOption:
    return ModelReleaseOption(
        tier=tier,
        model_id=model_id,
        title=title,
        display_name=display_name,
        model_filename=model_filename,
        model_size_bytes=model_size_bytes,
        approx_size_gb=round(model_size_bytes / _DECIMAL_GB, 2) if model_size_bytes > 0 else 0.0,
        quality=quality,
        speed=speed,
        ram_profile=ram_profile,
        description=description,
        release_label=release_label,
        downloadable=downloadable,
    )

SETUP_TIER_OPTIONS: tuple[SetupTierOption, ...] = (
    SetupTierOption(
        tier='small',
        model_id=MODEL_ID_QWEN_9B,
        title='Small',
        display_name=get_model_display_name('Qwen_Qwen3.5-9B-Q4_K_M.gguf'),
        model_filename='Qwen_Qwen3.5-9B-Q4_K_M.gguf',
        model_size_bytes=_MODEL_SIZE_BYTES['Qwen_Qwen3.5-9B-Q4_K_M.gguf'],
        approx_size_gb=round(_MODEL_SIZE_BYTES['Qwen_Qwen3.5-9B-Q4_K_M.gguf'] / _DECIMAL_GB, 2),
        quality='Good',
        speed='Fast',
        ram_profile='Lower RAM',
        description='Fastest setup with lower memory footprint.',
    ),
    SetupTierOption(
        tier='balanced',
        model_id=MODEL_ID_QWEN_14B,
        title='Balanced',
        display_name=get_model_display_name('Qwen3-14B-Q5_K_M.gguf'),
        model_filename='Qwen3-14B-Q5_K_M.gguf',
        model_size_bytes=_MODEL_SIZE_BYTES['Qwen3-14B-Q5_K_M.gguf'],
        approx_size_gb=round(_MODEL_SIZE_BYTES['Qwen3-14B-Q5_K_M.gguf'] / _DECIMAL_GB, 2),
        quality='High',
        speed='Balanced',
        ram_profile='Medium RAM',
        description='Recommended quality and speed tradeoff.',
    ),
    SetupTierOption(
        tier='quality',
        model_id=MODEL_ID_QWEN_35B_A3B,
        title='Quality',
        display_name=get_model_display_name('Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'),
        model_filename='Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
        model_size_bytes=_MODEL_SIZE_BYTES['Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'],
        approx_size_gb=round(_MODEL_SIZE_BYTES['Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'] / _DECIMAL_GB, 2),
        quality='Highest',
        speed='Slower',
        ram_profile='Higher RAM',
        description='Best answer quality with higher resource usage.',
    ),
)

SETUP_MODEL_RELEASES: tuple[ModelReleaseOption, ...] = (
    _build_release(
        tier='small',
        model_id=MODEL_ID_QWEN_9B,
        title='Small',
        display_name=get_model_display_name('Qwen_Qwen3.5-9B-Q4_K_M.gguf'),
        model_filename='Qwen_Qwen3.5-9B-Q4_K_M.gguf',
        model_size_bytes=_MODEL_SIZE_BYTES['Qwen_Qwen3.5-9B-Q4_K_M.gguf'],
        quality='Good',
        speed='Fast',
        ram_profile='Lower RAM',
        description='Fastest setup with lower memory footprint.',
        release_label='Recommended',
    ),
    _build_release(
        tier='balanced',
        model_id=MODEL_ID_QWEN_14B,
        title='Balanced',
        display_name=get_model_display_name('Qwen3-14B-Q5_K_M.gguf'),
        model_filename='Qwen3-14B-Q5_K_M.gguf',
        model_size_bytes=_MODEL_SIZE_BYTES['Qwen3-14B-Q5_K_M.gguf'],
        quality='High',
        speed='Balanced',
        ram_profile='Medium RAM',
        description='Recommended quality and speed tradeoff.',
        release_label='Recommended',
    ),
    _build_release(
        tier='quality',
        model_id=MODEL_ID_QWEN_35B_A3B,
        title='Quality',
        display_name=get_model_display_name('Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'),
        model_filename='Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
        model_size_bytes=_MODEL_SIZE_BYTES['Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'],
        quality='Highest',
        speed='Slower',
        ram_profile='Higher RAM',
        description='Best answer quality with higher resource usage.',
        release_label='Recommended',
    ),
    _build_release(
        tier='quality',
        model_id=MODEL_ID_QWEN_35B_A3B,
        title='Quality',
        display_name=get_model_display_name('Qwen3.5-35B-A3B-Q4_K_M.gguf'),
        model_filename='Qwen3.5-35B-A3B-Q4_K_M.gguf',
        model_size_bytes=_MODEL_SIZE_BYTES['Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'],
        quality='Highest',
        speed='Slower',
        ram_profile='Higher RAM',
        description='Older quality-tier model; still usable, but superseded by the current quality release.',
        release_label='Legacy',
        downloadable=False,
    ),
)

SETUP_TIER_REPOS: dict[str, str] = {
    'small': 'bartowski/Qwen_Qwen3.5-9B-GGUF',
    'balanced': 'Qwen/Qwen3-14B-GGUF',
    'quality': 'unsloth/Qwen3.6-35B-A3B-GGUF',
}

SETUP_TIER_REVISIONS: dict[str, str] = {
    'small': 'ff13963796ee209598509a81340172bb1c3869fe',
    'balanced': '530227a7d994db8eca5ab5ced2fb692b614357fd',
}

SETUP_MODEL_SHA256: dict[str, str] = {
    'Qwen_Qwen3.5-9B-Q4_K_M.gguf': '9437f5bf0dd0c97800caaf902f41e6a6aa00223ab232f159eda41dcbbb492645',
    'Qwen3-14B-Q5_K_M.gguf': 'e7c9aba1129ca2936be9eca01419d9f86af40e08caa01230d5574b34d08e3e31',
    'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf': 'ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61',
}
