Connect & Test Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^

We now verify that the system is working as expected and a connection can be established. For this, we use the `iperf3` tool to test the performance of the 5G connection. Further, the system is monitored using `jtop` to check the performance of the Jetson platform.

Run the following commands to verify the connectivity:

.. code-block:: bash

   # Check connection status
   nmcli connection show

   # Check that ip address is assigned to wwan0
   ip addr show wwan0

   # Test internet connectivity through the 5G tunnel
   ping -I wwan0 google.com

And to monitor the system:

.. code-block:: bash

   # View gNB logs
   docker logs -f oai-gnb

   # Monitor Jetson performance
   pip install jetson-stats
   jtop

The following commands test the performance of the 5G connection:

.. code-block:: bash

   # Install iperf3 on both UE and UPF (if not already installed)
   sudo apt update
   sudo apt install iperf3

   # start iperf3 server in EXT-DN Docker container
   docker exec -d oai-ext-dn iperf3 -s

   # On the client (UE)
   # Downlink test
   iperf3 -u -t 10 -i 1 -b 1M -B 12.1.1.2 -c 192.168.72.135 -R

   # Uplink test
   iperf3 -u -t 10 -i 1 -b 1M -B 12.1.1.2 -c 192.168.72.135

   # change 1M to the desired throughput in Mbit/s

The above command assume that the UE has assigned the IP address 12.1.1.2 and the ext-dn is at 192.168.72.135 as shown in :numref:`fig_5g_stack`. The IP addresses can be changed to your own configuration.

