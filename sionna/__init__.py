#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""This is the Sionna library.
"""

__version__ = '0.15.0'

from . import utils
from .constants import *
from . import nr
from . import fec
from . import mapping
from . import ofdm
from . import mimo
from . import channel
from . import signal
from . import rt
from .config import *

# Instantiate global configuration object
config = Config()
