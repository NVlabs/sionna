.. system_upgrade:

System Upgrades
===============

The Sionna Research Kit is constantly evolving to support the latest features and improvements. Keeping your system up to date is important to ensure you are using the latest version of the software and to benefit from the latest bug fixes and improvements. This section provides instructions for upgrading your system to the latest version.

Updating a DGX Spark OS
-----------------------

The recommended way to update a DGX Spark is by using the DGX Dashboard. If that is not available, you can do it manually over the command line by following the steps below.

.. code-block:: bash

    sudo apt update
    sudo apt dist-upgrade
    sudo fwupdmgr refresh
    sudo fwupdmgr upgrade
    sudo reboot

Source: `NVIDIA DGX Spark User Guide <https://docs.nvidia.com/dgx/dgx-spark/os-and-component-update.html>`_

Updating a Jetson Thor OS
-------------------------

Check the `Jetson archive <https://developer.nvidia.com/embedded/jetpack-archive>`_ for the latest Release and Revision for your platform.

.. code-block:: bash

    # Check the current version:
    cat /etc/nv_tegra_release

    # Update the index sources (skip if revision does not change):
    # backup the original file
    sudo cp /etc/apt/sources.list.d/nvidia-l4t-apt-source.list /etc/apt/sources.list.d/nvidia-l4t-apt-source.list.bak
    # update the release and revision
    sudo sed -i 's/r38\.2/r38.4/g' /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
    # check the updated file
    tail -n5 /etc/apt/sources.list.d/nvidia-l4t-apt-source.list

    # update the system
    sudo apt update
    sudo apt dist-upgrade -y
    sudo reboot

Source: `NVIDIA Jetson Linux Developer Guide <https://docs.nvidia.com/jetson/archives/r38.2/DeveloperGuide/SD/SoftwarePackagesAndTheUpdateMechanism.html#updating-a-jetson-device>`_

Updating a AGX Orin and Jetson Orin Nano OS
-------------------------------------------

Check the `Jetson archive <https://developer.nvidia.com/embedded/jetpack-archive>`_ for the latest Release and Revision for your platform.

.. code-block:: bash

    # Check the current version:
    cat /etc/nv_tegra_release

    # Update the index sources (skip if revision does not change):
    # backup the original file
    sudo cp /etc/apt/sources.list.d/nvidia-l4t-apt-source.list /etc/apt/sources.list.d/nvidia-l4t-apt-source.list.bak
    # update the release and revision
    sudo sed -i 's/r34\.2/r36.5/g' /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
    # check the updated file
    tail -n5 /etc/apt/sources.list.d/nvidia-l4t-apt-source.list

    # update the system
    sudo apt update
    sudo apt dist-upgrade -y
    sudo reboot

Source: `NVIDIA Jetson Linux Developer Guide <https://docs.nvidia.com/jetson/archives/r38.2/DeveloperGuide/SD/SoftwarePackagesAndTheUpdateMechanism.html#updating-a-jetson-device>`_

Rebuilding the Sionna Research Kit
----------------------------------

After updating the system, you may need to rebuild the Sionna Research Kit to ensure it is compatible with the latest version. If your system requires a custom kernel, you will need to rebuild the kernel as well.

Rebuilding the kernel
^^^^^^^^^^^^^^^^^^^^^

In some configurations, upgrading the system will override the custom kernel with the standard kernel delivered by the Release. If a custom kernel is needed for the extra modules, you will need to rebuild it. Finish the system upgrade, including a reboot, and then rebuild the custom kernel.

.. code-block:: bash

    cd ~/sionna-rk

    # build the kernel
    ./scripts/build-custom-kernel.sh --clean

    # if compiling for the AGX Orin or Jetson Orin Nano, force the l4t version if needed.
    ./scripts/build-custom-kernel.sh --clean --l4t-version 36.4.4

    # install the kernel
    ./scripts/install-custom-kernel.sh

    # reboot the system
    sudo reboot

Rebuilding the Docker images and the plugins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rebuilding the docker images and the plugins ensures that all components are refreshed and compatible with the latest libraries in the system.

.. code-block:: bash

    cd ~/sionna-rk

    # rebuild the Docker images
    make build-gnb

    # rebuild the plugins (because TRT libraries may have changed)
    ./plugins/common/build_all_plugins.sh --host
    ./plugins/common/build_all_plugins.sh --container    
