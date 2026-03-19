Scheduling
==========

.. figure:: ../figures/scheduling_api.png
   :align: center
   :width: 100%

Since the spectrum is shared among multiple users, resources must be allocated
in a fair and efficient manner. 
On the one hand, it is desirable to allocate resources uniformly across users. 
On the other hand, in the presence of fading, it is crucial to schedule users
when their channel conditions are favorable. 

The proportional fairness (PF) scheduler achieves both objectives by
maximizing the sum of logarithms of the long-term throughputs :math:`T(u)`
across the users :math:`u=1,2,\dots`: 

.. math::

    \max \sum_u \log T(u).

For a usage example of user scheduling in Sionna, refer to the
`Proportional Fairness Scheduler notebook <../tutorials/Scheduling.html>`_.

.. autoclass:: sionna.sys.PFSchedulerSUMIMO
    :members: 
    :exclude-members: call, build

References:

   .. [Jalali00] A\. Jalali, R\. Padovani, R\. Pankaj, "Data
    throughput of CDMA-HDR a high efficiency-high data rate personal
    communication wireless system." VTC2000-Spring. 2000 IEEE 51st Vehicular
    Technology Conference Proceedings. Vol. 3. IEEE, 2000. 
