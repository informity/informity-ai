from __future__ import annotations

from informity.file_types import (
    FILE_TYPE_OPTIONS,
    SUPPORTED_EXTENSIONS_CANONICAL_ORDER,
    get_file_type_options,
)


def test_image_file_type_is_present_and_canonical() -> None:
    options = get_file_type_options()
    image_option = next(option for option in options if option['id'] == 'image')

    assert image_option['label'] == 'Images'
    assert image_option['extensions'] == ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp']
    assert '.jpg' in SUPPORTED_EXTENSIONS_CANONICAL_ORDER
    assert any(option['id'] == 'image' for option in FILE_TYPE_OPTIONS)
