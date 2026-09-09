<!--
SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

-->
# Contributing to Sionna

## Issue tracking
For enhancement, bugfix, or change requests, please open an issue.

## Pull requests
Creating a pull request requires the following steps:

1. Fork the repository and create your branch from `main`.
2. Add new tests if needed in the ``test`` folder.
3. Lint your code.
4. Ensure that all tests pass.
5. Include a license at the topf of new files.
5. Make sure that all commits are "signed-off" as described below.

### Validation and warning contracts

Checks on public arguments and mutable public properties must remain active
under optimized Python (`python -O`). Do not use `assert` for these checks.
Use:

* `TypeError` when an argument has the wrong Python type or is not callable.
* `ValueError` when an argument has an invalid value, shape, rank, dtype, or
  combination with another argument.
* `RuntimeError` when valid public input encounters invalid or unavailable
  internal state.

Bare `assert` is reserved for debug-only invariants that may safely disappear
under optimization. The validation-contract policy test rejects every `assert`
in `src/sionna`, so an exception to this rule requires extending that test.

Tensor-valued validation on a path supported by `torch.compile` needs an
additional carve-out. Python branches such as
`if not torch.all(condition): raise ...`, including `bool(tensor)` and
`tensor.item()`, break the Dynamo graph. Use the private helpers in
`sionna._validation`: they preserve the documented eager exception and lower
the condition to a device assertion while compiling. Compiled failures are
backend-dependent and do not preserve the eager exception class. Do not call
PyTorch's private assertion APIs directly outside that module. Constructor and
configuration checks that run before compilation can remain unconditional. Add
a `fullgraph=True` regression test when changing validation on a compiled path.

Every `warnings.warn()` call must specify a category and a `stacklevel` that
points to the caller that can act on the warning. Use `UserWarning` for
configuration, data, and API-usage notices; `RuntimeWarning` for numerical
fallbacks and unavailable runtime state; and `DeprecationWarning` or
`FutureWarning` for API transitions.

### Signing your contributions

* We require that all contributors "sign-off" on their commits with the [Developer Certificate of Origin (DCO)](https://developercertificate.org). This certifies that the contribution is your original work, or that you have rights to submit it under the same license, or a compatible license.

* Any contribution which contains commits that are not signed-off will not be accepted.

* To sign-off on a commit, you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "New feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* The full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```


