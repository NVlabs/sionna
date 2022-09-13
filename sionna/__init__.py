#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""This is the Sionna library.
"""

__version__ = '0.11.0'

from . import utils
from .constants import *
from . import fec
from . import mapping
from . import ofdm
from . import mimo
from . import channel
from . import signal
from .config import *

# Instantiate global configuration object
config = Config()
