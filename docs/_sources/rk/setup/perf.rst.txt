Jetson Performance Tweaks
=========================

This guide covers key performance optimizations for the Jetson platform, including power settings, CPU governors, and real-time priority settings.

Power Management
----------------

Per default the Jetson platform is configured to use the "balanced" power mode.
For best performance, we recommend to use the "max-performance" power mode as described below.

You can view the current settings with:

.. code-block:: bash

   # Check current power mode
   sudo nvpmodel -q

   # View detailed configuration
   sudo nvpmodel -q --verbose

To optimize the power settings, run:

.. code-block:: bash

   # Set maximum performance mode (50W)
   sudo nvpmodel -m 0

   # Verify changes
   sudo nvpmodel -q

The CPU governor controls how the processor scales frequency based on load. Setting it to "performance" ensures maximum processing speed.

.. code-block:: bash

   # on AGX Orin
   sudo cpufreq-set -c 0 -g performance
   sudo cpufreq-set -c 4 -g performance
   sudo cpufreq-set -c 8 -g performance

   # on AGX Thor
   sudo cpufreq-set -c 0 -g performance
   sudo cpufreq-set -c 2 -g performance
   sudo cpufreq-set -c 4 -g performance
   sudo cpufreq-set -c 6 -g performance
   sudo cpufreq-set -c 8 -g performance
   sudo cpufreq-set -c 10 -g performance
   sudo cpufreq-set -c 12 -g performance

To make these changes persistent across reboots, modify ``/etc/nvpmodel.conf`` to set the default power mode, and create ``/etc/default/cpufrequtils`` to set the CPU governor. This can also be done as follows:

.. code-block:: bash

   # on AGX Orin
   sudo sed -i 's|< PM_CONFIG DEFAULT=2 >|< PM_CONFIG DEFAULT=0 >|' /etc/nvpmodel.conf

   # on AGX Thor
   sudo sed -i 's|< PM_CONFIG DEFAULT=1 >|< PM_CONFIG DEFAULT=0 >|' /etc/nvpmodel.conf

   echo 'GOVERNOR="performance"' | tee /etc/default/cpufrequtils

Default System Mode
-------------------

The Ubuntu system in the Jetson ships with booting on graphical mode by default. This consumes resources that can be better used elsewhere. You can switch the default to be multi user command line in order to free these resources.

.. code-block:: bash

   # check current system runlevel target
   systemctl get-default

   # change default to multi-user (non-GUI)
   sudo systemctl set-default multi-user.target

   # change current running system to multi-user
   sudo systemctl isolate multi-user.target

   ### If you want to go back to graphical mode:

   # change current running system back to GUI
   sudo systemctl isolate graphical.target

   # change default back to GUI
   sudo systemctl set-default graphical.target

Note this is mostly relevant for the Jetson Orin Nano Super.

Real-Time Scheduling
--------------------

The signal processing latency is critical for the 5G stack.
We recommend to use the following configuration to further reduce the latency induced by the scheduler.

Create USRP group and add user:

   .. code-block:: bash

      sudo groupadd usrp
      sudo usermod -aG usrp $USER

Add real-time priority limits by creating/editing ``/etc/security/limits.conf``:

   .. code-block:: text

      @usrp - rtprio 99


Configure kernel real-time scheduling:

.. code-block:: bash

   # Check current setting
   cat /proc/sys/kernel/sched_rt_runtime_us

   # Remove runtime limit
   sudo su -
   echo -1 > /proc/sys/kernel/sched_rt_runtime_us

.. note::
   Log out and back in for group changes to take effect. A system reboot may be required for some changes.

Verifying Settings
------------------

.. code-block:: bash

   # Check CPU frequencies
   cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq

   # Check CPU governors
   cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

   # Check real-time limits
   ulimit -r

