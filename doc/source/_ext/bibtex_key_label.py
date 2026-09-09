#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Custom pybtex styles that use the BibTeX key as the label.

This ensures that citations display as [Kouyoumjian74], [TR38901V160100], etc.,
and that labels are inherently unique (since BibTeX keys are unique).
"""

from pybtex.style.formatting.plain import Style as PlainStyle
from pybtex.style.labels import BaseLabelStyle


class KeyLabelStyle(BaseLabelStyle):
    """Label each bibliography entry with its BibTeX key."""

    def format_labels(self, sorted_entries):
        for entry in sorted_entries:
            yield entry.key


class KeyLabelPlainStyle(PlainStyle):
    """Plain formatting style that uses BibTeX keys as labels."""

    default_label_style = "keylabel"
