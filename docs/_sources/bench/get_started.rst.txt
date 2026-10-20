Get Started
===========

This guide covers the installation of Sionna Bench, how to display the leaderboard locally,
how to add your own method for comparison, and how to reproduce the leaderboard results.

Although not necessary, we recommend running Sionna in a
`Docker container <https://www.docker.com>`_ and/or `Python virtual
environment <https://docs.python.org/3/library/venv.html>`_.


Installation
------------


Clone the repository and enter its root directory:

.. code-block:: bash

   git clone --recurse-submodules https://github.com/NVlabs/sionna-bench.git
   cd sionna-bench

Install Sionna Bench from the repository root.

.. code-block:: bash

   pip install -e .

The tests require the optional test dependencies:

.. code-block:: bash

   pip install -e ".[test]"

The tuning commands require the optional tuning dependencies:

.. code-block:: bash

   pip install -e ".[tuning]"

Building the documentation requires the optional documentation dependencies:

.. code-block:: bash

   pip install -e ".[docs]"

Confirm that the active environment imports Sionna and Sionna Bench:

.. code-block:: bash

   python - <<'PY'
   from importlib.metadata import version
   import sionna.bench

   print(version("sionna-bench"))
   PY

The command should print the version of Sionna Bench.


.. _get-started-local-leaderboard:

Local Leaderboard
-----------------

The repository includes the numerical results necessary to display the leaderboard locally
without rerunning the benchmark. To display the leaderboard locally, start a
static HTTP server from the repository root:

.. code-block:: bash

   python -m http.server 8000 --directory leaderboard/mimo_ofdm_chest

Open http://localhost:8000 in a browser. Stop the server with :kbd:`Ctrl-C`.


Adding a New Method
-------------------

A new method can be added in an external project or in a new module below
``src/sionna/bench/<benchmark_identifier>/papers``. It only needs to implement
the interface of the applicable Sionna block and be passed to the shared
evaluation pipeline.

For instance, to add your own channel estimator to the MIMO-OFDM Channel
Estimation Benchmark, subclass Sionna's :class:`~sionna.phy.ofdm.BasePilotChannelEstimator`
or :class:`~sionna.phy.ofdm.BaseChannelEstimator`.

``examples/custom_estimator.py`` contains a minimal example of a MIMO-OFDM
channel estimator. It subclasses Sionna's :class:`~sionna.phy.ofdm.BaseChannelEstimator`,
estimates the channel at pilot locations with least squares followed by scalar shrinkage,
and delegates full-grid reconstruction to Sionna's configured interpolator.
``examples/run_custom_benchmark.py`` evaluates it and records the
result through the shared result store.

.. literalinclude:: examples/custom_estimator.py
   :language: python
   :start-at: class ShrinkageLSChannelEstimator(BasePilotChannelEstimator):

The important benchmark integration calls are:

.. literalinclude:: examples/run_custom_benchmark.py
   :language: python
   :start-at: def evaluate_example(
   :end-at: return result

``evaluate_channel_estimator()`` runs the common communication system and
writes the metric arrays. ``record_result()`` then adds the estimator class and
effective algorithmic settings to the scenario-local ``results.json``.
You can run it from the repository root with:

.. code-block:: bash

   python -m examples.run_custom_benchmark

Pass ``--device cpu`` to evaluate the estimator on CPU if you do not have a CUDA-capable GPU.
After the result is recorded, regenerate ``leaderboard/mimo_ofdm_chest/data.json`` by running

.. code-block:: bash

   python leaderboard/mimo_ofdm_chest/generate_data.py

and serve the local dashboard as described above.


Reproducing Results
-------------------

Each implementation in ``src/sionna/bench/mimo_ofdm_chest/papers`` provides a ``run_benchmark.py``
module. Its command-line defaults are the selected estimator and asset-generation settings;
arguments are needed only to choose a method, set the execution device,
select external storage, or deliberately override a default. Full runs are computationally expensive.

The configuration selects ``cuda:0`` by default, but a CUDA-capable
GPU is not required to run an evaluation. Pass ``--device cpu`` to any paper
runner, including asset-preparation commands. For example:

.. code-block:: bash

   python -m sionna.bench.mimo_ofdm_chest.papers.delfeld2025.run_benchmark \
       --device cpu

Displaying the local leaderboard does not require a GPU. Full evaluations,
statistical fitting, and especially neural-network training can take
substantially longer on CPU.


An estimator that requires no persistent assets (such as trained weights or statistics)
can be evaluated directly. For example, this command regenerates the result for
the :class:`~sionna.bench.mimo_ofdm_chest.papers.delfeld2025.RKHSChannelEstimator`:

.. code-block:: bash

   python -m sionna.bench.mimo_ofdm_chest.papers.delfeld2025.run_benchmark

For an implementation with several methods, select one method explicitly. For example, this
command evaluates the low-pass interpolator of the
:class:`~sionna.bench.mimo_ofdm_chest.papers.coleri2002.CombInterpolator`:

.. code-block:: bash

   python -m sionna.bench.mimo_ofdm_chest.papers.coleri2002.run_benchmark \
       --method low-pass

Results are written below ``outputs/mimo_ofdm_chest/<paper>/<scenario>/``. The NPZ file stores
the metric curves, while the neighboring ``config.json`` and ``results.json``
record the system configuration and estimator settings.

For methods that require persistent assets such as covariances, fitted statistical
models, or trained network states, these assets need to be generated explicitly.
To this end, first execute the runner with the ``--prepare-assets`` flag to generate the assets,
and then again without the flag to evaluate the method with the generated assets.
For example, the spatial GMM is fitted and evaluated with:

.. code-block:: bash

   python -m sionna.bench.mimo_ofdm_chest.papers.koller2022.run_benchmark \
       --prepare-assets \
       --asset-root generated-assets \
       --method spatial

   python -m sionna.bench.mimo_ofdm_chest.papers.koller2022.run_benchmark \
       --asset-root generated-assets \
       --method spatial

The preparation command uses the benchmark-matching defaults documented by
the paper module. It writes a manifest beside the payloads and validates the
completed asset through the same strict loader used during evaluation. Each
paper README documents its available methods and any additional generation
arguments.

There is a helper script that invokes every paper runner in a separate Python
process. To regenerate all required assets and then all canonical results,
you can run:

.. code-block:: bash

   python scripts/mimo_ofdm_chest/reproduce_results.py \
       --all \
       --prepare-assets \
       --asset-root generated-assets

Existing compatible assets can be reused by omitting ``--prepare-assets``.
Use one or more ``--paper`` options instead of ``--all`` to run a subset, and
``--dry-run`` to inspect the commands without executing them:

.. code-block:: bash

   python scripts/mimo_ofdm_chest/reproduce_results.py \
       --paper coleri2002 \
       --paper delfeld2025 \
       --dry-run

Pass an ``--output-dir`` to place generated result trees outside of ``outputs`` and
a ``--device`` to override the configured execution device.

Paper modules with tunable hyperparameters provide a separate ``tuning`` command.
That command does not rewrite estimator defaults or canonical results.
For example, the RKHS algorithmic study and its paired validation can be run as:

.. code-block:: bash

   python -m sionna.bench.mimo_ofdm_chest.papers.delfeld2025.tuning \
       --study rkhs \
       --output-dir tuning/mimo_ofdm_chest/delfeld2025/my-rkhs-study

   python -m sionna.bench.mimo_ofdm_chest.papers.delfeld2025.tuning \
       --study rkhs-validation \
       --output-dir tuning/mimo_ofdm_chest/delfeld2025/my-rkhs-study

The corresponding search spaces and tuning budgets are documented in the
paper-local tuning dosctring.

.. _get-started-refresh-leaderboard:

After outputs have changed, rebuild the dashboard data with

.. code-block:: bash

   python leaderboard/mimo_ofdm_chest/generate_data.py
