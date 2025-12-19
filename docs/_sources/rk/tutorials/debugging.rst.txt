.. _debugging:

Debugging & Troubleshooting
===========================

This guide provides a collection of tips and tricks for debugging and troubleshooting the Sionna Research Kit.


Attaching a debugger (`gdb` and `VS code`)
-------------------------------------------

To run a `gdbserver` inside the `gNB` container to which we can attach debuggers to, we override the `docker compose` command line as follows.
You can find the `docker-compose.override.yaml` file in the `config/common/docker-compose.override.yaml` directory of the Sionna Research Kit.

.. code-block:: yaml
   :caption: docker-compose.override.yaml

   services:
     oai-gnb:
       command: ["gdbserver",":7777","/opt/oai-gnb/bin/nr-softmodem","-O","/opt/oai-gnb/etc/gnb.conf"]

Note that we simply prepended `gdbserver :7777` to the pre-existing command line of the `oai-gnb` container, which can be obtained as follows:

.. code-block:: bash

   $ docker inspect --format='{{json .Config.Cmd}}' oai-gnb
   ["/opt/oai-gnb/bin/nr-softmodem","-O","/opt/oai-gnb/etc/gnb.conf"]

We can then attach using `gdb` as follows:

.. code-block:: bash

   $ gdb
   > target remote <CONTAINER_IP>:7777

To get the local IP of a running container, run:

.. code-block:: bash

   $ docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <CONTAINER_NAME, e.g. oai-gnb>

To attach using the `VS code` debugger, the following example `launch.json` configuration can be used:

.. code-block:: json

   {
       "version": "0.2.0",
       "configurations": [
         {
           "name": "C++ Remote Debug",
           "type": "cppdbg",
           "request": "launch",
           "program": "/opt/oai-gnb/bin/nr-softmodem",
           "customLaunchSetupCommands": [
             { "text": "target remote <CONTAINER_IP>:7777", "description": "attach to target", "ignoreFailures": false }
           ],
           "launchCompleteCommand": "None",
           "stopAtEntry": false,
           "cwd": "/",
           "environment": [],
           "externalConsole": false,
           "pipeTransport": { // needed if you are working on a different machine than the container host
               "pipeCwd": "${workspaceRoot}",
               "pipeProgram": "ssh",
               "pipeArgs": [
                   "user@<remote ip ?.?.?.?>"
               ],
               "debuggerPath": "/usr/bin/gdb"
           },
           "sourceFileMap": {
                   "/oai-ran":"${workspaceRoot}"
           },
           "targetArchitecture": "arm",
           "linux": {
             "MIMode": "gdb",
             "miDebuggerServerAddress": "<CONTAINER_IP>:7777",
             "setupCommands": [
               {
                 "description": "Enable pretty-printing for gdb",
                 "text": "-enable-pretty-printing",
                 "ignoreFailures": true
               }
             ]
           }
         }
   }

Inspecting and debugging inside a container interactively
---------------------------------------------------------

First, find out the commands that are usually run inside the container of interest, e.g., for the `oai-gnb` container

.. code-block:: bash

   $ docker inspect --format='{{json .Config.Entrypoint}} {{json .Config.Cmd}}' oai-gnb
   ["/tini","-v","--","/opt/oai-gnb/bin/entrypoint.sh"] ["/opt/oai-gnb/bin/nr-softmodem","-O","/opt/oai-gnb/etc/gnb.conf"]

We can then override the entrypoint to run an interactive session instead of the default launch procedure. This can be done by adding the following to the `docker-compose.override.yaml` file

.. code-block:: yaml
   :caption: docker-compose.override.yaml

   services:
     oai-gnb:
       stdin_open: true # docker run -i
       tty: true        # docker run -t
       entrypoint: /bin/bash

To attach to a running session after running `./start_system.sh` or `docker compose up -d oai-gnb`, and for example start a debug session, run

.. code-block:: bash

   $ docker container attach oai-gnb
   $ gdb --args /tini -v -- /opt/oai-gnb/bin/entrypoint.sh  ./bin/nr-softmodem -O /opt/oai-gnb/etc/gnb.conf

Running memcheck within an interactive Docker compose session
-------------------------------------------------------------

We can use the `compute-sanitizer` tool from the NVIDIA Cuda Toolkit to run a GPU memcheck inside an interactive container session launched as above with the following commands:

.. code-block:: bash

   $ docker container attach oai-gnb
   $ /tini -v -- /opt/oai-gnb/bin/entrypoint.sh compute-sanitizer --require-cuda-init=no --tool memcheck ./bin/nr-softmodem -O /opt/oai-gnb/etc/gnb.conf


Profiling with NVIDIA Nsight Systems
------------------------------------

To enable interactive profiling inside a container, we can mount the `Nsight Systems CLI tools <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>`_ installed on the host. This can be done by adding the following to the `docker-compose.override.yaml` file

.. code-block:: yaml
   :caption: docker-compose.override.yaml

   services:
     oai-gnb:
       stdin_open: true # docker run -i
       tty: true        # docker run -t
       entrypoint: /bin/bash
       cap_add:
        - SYS_ADMIN
       volumes:
        - /opt/nvidia/nsight-systems/:/opt/nvidia/nsight-systems

To collect system usage statistics using NVIDIA Nsight Systems, `nsys` can be run inside the gNB container as follows:

.. code-block:: bash

   $ docker container attach oai-gnb
   $ /tini -v -- /opt/oai-gnb/bin/entrypoint.sh /opt/nvidia/nsight-systems/2024.2.2/bin/nsys profile -t cuda,nvtx,osrt,cudnn,tegra-accelerators -o ./sysprofile ./bin/nr-softmodem -O /opt/oai-gnb/etc/gnb.conf

You can afterwards download the profile data from the Jetson and inspect it locally on your host machine.

Fixing missing linker error messages in `docker build ran-build`
----------------------------------------------------------------

In `./cmake_targets/tools/build_helper`, change the reprinted error context in the `compilations()` function from

.. code-block:: bash

   egrep -A3 "warning:|error:" $dlog/$logfile || true

to

.. code-block:: bash

   egrep -C6 "warning:|error:" $dlog/$logfile || true

This modification shows additional context before and after lines containing 'error', which is particularly helpful for linker (`ld`) errors where the actual error message often appears before the 'error' line in the output.
