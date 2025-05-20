5G System Configuration
=======================

This section explains how to configure the bandwidth, carrier frequency, and other key parameters of your 5G network.

The main configuration is done through the ``gnb.XXX.conf`` files. We provide several pre-configured files for common setups, but you can customize them for your needs.

.. note::

    For performance reasons, we recommend using either the configuration file with 24 or 51 PRBs.

The configuration files are located in the ``sionna-rk/config`` directory.
Most configurations are shared between different setups, these can be found in the ``sionna-rk/config/common`` directory:

* ``gnb.***.conf``: GNB configuration file
* ``mini_nonrf_config.yaml``: OAI CN configuration file
* ``nrue_uicc.conf``: NRUE UICC configuration file (relevant for software-defined UEs)
* ``.env.template``: Environment variables for customizing of experiments
* ``docker-compose.override.yml.template``: Custom overrides for the docker-compose file
* ``oai_db.sql``: database with all registered UEs

.. note::

   For testing purposes, we provide x86 configuration files. However, some tutorials will only work on the Jetson platform.

See `OpenAirInterface5G documentation <https://gitlab.eurecom.fr/oai/openairinterface5g/-/tree/develop/doc>`_ for further details about the configuration files.

A detailed explanation of the MAC parameters can be found in the `OAI MAC documentation <https://github.com/OPENAIRINTERFACE/openairinterface5g/blob/develop/doc/MAC/mac-usage.md>`_.

Environment Variables
---------------------

The ``.env`` file is used to set the environment variables for the experiments.
It includes the following variables:

* ``USRP_SERIAL``: Serial number of the USRP device
* ``USRP_SERIAL_UE``: Serial number of the USRP device for the UE (only for software-defined UE)
* ``GNB_CONFIG``: Path to the GNB configuration file
* ``GNB_EXTRA_OPTIONS``: Extra options for the GNB, e.g., library loader or thread-pool assignment
* ``DOCKERTAGS``: Which docker images to use

Note that the variables must be set before starting the Docker containers.

Additional Resources
--------------------

The carrier frequency is defined using the Absolute Radio Frequency Channel Number (ARFCN). You can use the `online ARFCN calculator <https://5g-tools.com/5g-nr-arfcn-calculator/>`_ to convert between frequency and ARFCN values.


The following could be useful for setting up the 5G network:

* `Practical Guide to 5G RAN Configuration <https://hal.science/hal-04502404v1/document>`_
* `gNB Frequency Setup Guide <https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/doc/gNB_frequency_setup.md>`_
* `OAI Core Network Configuration <https://gitlab.eurecom.fr/oai/cn5g/oai-cn5g-fed/-/blob/master/docs/CONFIGURATION.md>`_


