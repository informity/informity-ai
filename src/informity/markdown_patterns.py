"""Reusable Markdown pattern regexes."""

from __future__ import annotations

import re

TABLE_DIVIDER_PATTERN = re.compile(r'^\|?(?:\s*:?-{3,}:?\s*\|)+(?:\s*:?-{3,}:?\s*)\|?$')
HORIZONTAL_RULE_PATTERN = re.compile(r'^\s*(?:-{3,}|\*{3,}|_{3,})\s*$')
