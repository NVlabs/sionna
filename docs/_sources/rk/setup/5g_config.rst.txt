5G System Configuration
=======================

This section explains how to configure the bandwidth, carrier frequency, and other key parameters of your 5G network.

The 5G network parameters are configured through the ``gnb.XXX.conf`` files. We provide several pre-configured files for common setups, but you can customize them for your needs.

.. note::

    For performance reasons, we recommend using either the configuration file with 24 or 51 PRBs.

The configuration files are located in the ``sionna-rk/config`` directory.
Most configurations are shared between different setups, these can be found in the ``sionna-rk/config/common`` directory:

* ``gnb.***.conf``: GNB configuration file
* ``sys_config.yaml``: OAI CN configuration file
* ``nrue_uicc.conf``: NRUE UICC configuration file (relevant for software-defined UEs)
* ``docker-compose.override.yml.template``: Custom overrides for the docker-compose file (e.g. to enable debugging tools)
* ``oai_db.sql``: database with all registered UEs

See `OpenAirInterface5G documentation <https://openairinterface-docs-5b3d70.eurecom.io/>`_ for further details about the configuration files.

A detailed explanation of the MAC parameters can be found in the `OAI MAC documentation <https://github.com/OPENAIRINTERFACE/openairinterface5g/blob/develop/doc/MAC/mac-usage.md>`_.

Generating Configuration Files
------------------------------

Create configuration files from OAI templates:

.. code-block:: bash

   ./scripts/generate-configs.sh

Configuration files are written to ``config/`` with subdirectories for each setup (e.g., ``rfsim``, ``b200``).

Environment Variables
---------------------

Environment variables can be used to customize the 5G network configuration, e.g., to load plugins or to set the thread pool size.
These variables are stored in the ``.env`` file and includes the following variables:

* ``USRP_SERIAL``: Serial number of the USRP device
* ``USRP_SERIAL_UE``: Serial number of the USRP device for the UE (only for software-defined UE)
* ``GNB_CONFIG``: Path to the GNB configuration file
* ``GNB_EXTRA_OPTIONS``: Extra options for the GNB, e.g., library loader
* ``GNB_THREAD_POOL``: Thread pool assignment for the GNB
* ``UE_EXTRA_OPTIONS``: Extra options for the UE (only for software-defined UE), e.g., library loader
* ``USE_B2XX``: If True, USRP is used (otherwise RF simulator is used)
* ``GNB_RF_OPTIONS``: GNB RF options for USRP or rfsim
* ``UE_RF_OPTIONS``: UE RF options for USRP or rfsim
* ``<XXX>_TAG``: Which docker images to use

Note that the variables must be set before starting the Docker containers.

Additional Resources
--------------------

The carrier frequency is defined using the Absolute Radio Frequency Channel Number (ARFCN). You can use the `online ARFCN calculator <https://5g-tools.com/5g-nr-arfcn-calculator/>`_ to convert between frequency and ARFCN values.

The following could be useful for setting up the 5G network:

* `Practical Guide to 5G RAN Configuration <https://hal.science/hal-04502404v1/document>`_
* `gNB Frequency Setup Guide <https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/doc/gNB_frequency_setup.md>`_
* `OAI Core Network Configuration <https://gitlab.eurecom.fr/oai/cn5g/oai-cn5g-fed/-/blob/master/docs/CONFIGURATION.md>`_


