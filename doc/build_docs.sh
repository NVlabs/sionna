##
## SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
## SPDX-License-Identifier: Apache-2.0
##
set -e
cp -r ../examples source # copy examples as .rst does not allow to index parent dirs
make clean
make html
rm -rf source/examples # remove examples after building docs
