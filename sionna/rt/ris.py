#
# SPDX-License-Identifier: Apache-2.0
#
"""
Class implementing a Reflective Intelligent Surface.
Note: We support only one RIS in the simulator. Need to refactor code
        to support multiple RIS in future.
"""

import tensorflow as tf
from .radio_device import RadioDevice
from .transmitter import Transmitter
from .receiver import Receiver
from ..channel.utils import cir_to_ofdm_channel
from ..constants import PI

class RIS(RadioDevice):
    # pylint: disable=line-too-long
    r"""
    RIS()

    Implements a Reconfigurable Intelligent Surface (RIS) in the simulator.
    RIS is described as an antenna array with configurable properites like
    Phase, Gain and Polarization control at each antenna element.

    Parameters
    ----------
    name : str
        Name

    position : [3], float
        Position :math:`(x,y,z)` as three-dimensional vector

    ris_array : class:`~sionna.rt.AntennaArray`
        An instance of antenna array that is used to represent the RIS array.
        Need to define each element as a dual polarized antenna

    ris_phases : [array_size] tf.complex64
        Tensor of phasors to be applied at each antenna. Entries in the tensor
        are represented as exp(1j*phase shift) at each antenna. Defaults to tensor
        with 1 as each entry.

    ris_gain : [array_size] tf.complex64
        Tensor of gains to be applied at each antenna.
        Defaults to gain 1 at each antenna.

    element_transfer_matrix: [2,2] tf.complex64
        2D Tensor represents the polarization conversion properties of
        each antenna element. It is defined as following:
        [t_vv t_vh
         t_hv t_hh]
        t_vv : Fraction of the incident vertical polarization signal
                converted to vertical polarization and reflected
        t_vh : Fraction of the incident vertical polarization signal
                converted to horizontal polarization and reflected
        t_hv : Fraction of the incident horizontal polarization signal
                converted to vertical polarization and reflected
        t_hh : Fraction of the incident horizontal polarization signal
                converted to vertical polarization and reflected

        Note: 1 . Each entry could be complex.
        2. If not specified, defaults to an Identity
        matrix indicating no change in polarization upon reflection
        3. All elements should have the same transfer matrix. Consider expanding to have
        different transfer matrices for each element

    orientation : [3], float
        Orientation :math:`(\alpha, \beta, \gamma)` specified
        through three angles corresponding to a 3D rotation
        as defined in :eq:`rotation`.
        This parameter is ignored if ``look_at`` is not `None`.
        Defaults to [0,0,0].

    look_at : [3], float | :class:`~sionna.rt.Transmitter` | :class:`~sionna.rt.Receiver` | :class:`~sionna.rt.Camera` | None
        A position or the instance of a :class:`~sionna.rt.Transmitter`,
        :class:`~sionna.rt.Receiver`, or :class:`~sionna.rt.Camera` to look at.
        If set to `None`, then ``orientation`` is used to orientate the device.

    color : [3], float
        Defines the RGB (red, green, blue) ``color`` parameter for the device as displayed in the previewer and renderer.
        Each RGB component must have a value within the range :math:`\in [0,1]`.

    dtype : tf.complex
        Datatype to be used in internal calculations.
        Defaults to `tf.complex64`.
    """

    def __init__(self,
                 name,
                 position,
                 ris_array,
                 ris_phases=None,
                 ris_gains = None,
                 element_transfer_matrix = None,
                 orientation=(0.,0.,0.),
                 look_at=None,
                 color=(1, 0, 0),
                 dtype=tf.complex64):

        # Initialize the base class Object
        super().__init__(name=name,
                         position=position,
                         orientation=orientation,
                         look_at=look_at,
                         color=color,
                         dtype=dtype)

        self._dtype = dtype
        self.ris_array = ris_array
        self.ris_phases = ris_phases
        self.ris_element_gains = ris_gains
        self.element_transfer_matrix = element_transfer_matrix

        if ris_array is None:
            raise ValueError("`ris_array` cannot be empty. It should be an instance of Antenna array")

        # By default, ris element do not impart any extra phase
        if ris_phases is None:
            self.ris_phases = tf.ones((self.ris_array.array_size),dtype=tf.complex64)
        self.ris_optimal_phases = None

        # By default each element has unity amplification gain
        if ris_gains is None:
            self.ris_element_gains = tf.ones((self.ris_array.array_size),dtype=tf.complex64)

        # By default the polarization conversion should not happen
        if element_transfer_matrix is None:
            self.element_transfer_matrix = tf.eye(2,dtype=tf.complex64) # Identity transfer matrix

    def ris_channel(self,scene,frequencies,tx_to_ris_env=True,rx_to_ris_env=True,phase_optimizer=False,max_depth=1,num_samples=10e6):
        # pylint: disable=line-too-long
        r"""
        Obtain wireless channel between a TX and RX  via a
        Reconfigurable Intelligent Surface.

        Simulating the wireless channel via RIS is done in three steps.

        Step 1: First we simulate the wireless channel between
            Tx and RIS by performing a ray tracing

        Step 2: Next we simulate the wireless channel between
            RIS and Rx by performing another ray tracing

        Step 3: At last, we combine the channels from step 1, step 2 and
            Phase, Gain and Polarization matrices to get the cumulative channel
            through RIS

        Input
        -------
        scene : class:`~sionna.rt.Scene()`
            Scene containing the Transmitters, Receivers and environment

        frequencies : [fft_size], tf.float
            Frequencies at which to compute the channel response

        tx_to_ris_env : bool
            If set to True, the effect of environmental reflections and scattering in the scene are
            considered when computing the channel from Tx to RIS. Defaults to True.
            Note: Diffraction is not considered

        tx_to_ris_env : bool
            If set to True, the effect of environmental reflections and scattering in the scene are
            considered when computing the channel from RIS to Rx. Defaults to True.
            Note: DIffraction is not considered

        phase_optimizer : bool
            If set to True, optimal phases are computed at each antenna element based
            on path length matching and these phases are used for calculating the channel.
            Defaults to False.

        max_depth : float
            Defaults to 1.

        num_samples : float
            Defaults to 10e6.

        Output
        -------
        h : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size], tf.complex
            Channel from Tx to Rx via RIS

        h1 : [batch size, num_ris, num_ris_ant, num_tx, num_tx_ant, num_time_steps, fft_size], tf.complex
            Channel from Tx to RIS

        h2 : [batch size, num_rx, num_rx_ant, num_ris, num_ris_ant, num_time_steps, fft_size], tf.complex
            Channel from RIS to Rx
        """
        # Save the list of transmitters and receivers for using later
        tx_dict = scene.transmitters
        rx_dict = scene.receivers

        # Save the tx and rx array structures for reusing later
        tx_array = scene.tx_array
        rx_array = scene.rx_array

        # Remove rx and make ris act as receiver
        ris_node = Receiver(name=self.name,position=self.position,orientation=self.orientation)
        for rx_label,_ in rx_dict.items():
            scene.remove(rx_label)
        scene.rx_array = self.ris_array
        scene.add(ris_node)
        # Tx to RIS channel
        h1 = self.compute_channel_to_from_ris_v1(scene,frequencies,tx_to_ris_env,max_depth,num_samples)
        scene.remove(ris_node.name)

        # Remove tx and make ris act as transmitter
        ris_node = Transmitter(name=self.name,position=self.position,orientation=self.orientation)
        for tx_label,_ in tx_dict.items():
            scene.remove(tx_label)
        scene.remove("tx")
        scene.tx_array = self.ris_array
        scene.rx_array = rx_array
        scene.add(ris_node)
        for _,rx in rx_dict.items():
            scene.add(rx)
        # RIS to Rx channel
        h2 = self.compute_channel_to_from_ris_v1(scene,frequencies,rx_to_ris_env,max_depth,num_samples)

        # Make the scene look how it is before adding the ris. Hence remove ris
        scene.remove(ris_node.name)
        for _,tx in tx_dict.items():
            scene.add(tx)
        scene.tx_array = tx_array

        if phase_optimizer is True:
            self.ris_optimal_phases = self.find_optimal_phases_based_on_path_length(scene)
            phases_to_use = self.ris_optimal_phases
        else:
            phases_to_use = self.ris_phases

        h = self.combine_channel_matrix_all_pol(h1,h2,phases_to_use)
        return h,h1,h2

    def compute_channel_to_from_ris_v1(self,scene,frequencies,env_flag,max_depth,num_samples):
        # pylint: disable=line-too-long
        r"""
        Obtain channel from a TX / RX to RIS

        Input
        -------
        scene : class:`~sionna.rt.Scene()`
            Scene containing the Transmitters, Receivers and environment

        frequencies : [fft_size], tf.float
            Frequencies at which to compute the channel response

        env_flag : bool
            If set to True, the effect of environmental reflections in the scene are
            considered while ray tracing.

        max_depth : float

        num_samples : float

        Output
        -------
        h : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size], tf.complex
            Channel from Tx to RIS or RIS to Rx
        """
        if env_flag is True:
            paths = scene.compute_paths(max_depth=max_depth,
                                    num_samples=num_samples, reflection=True,scattering=True)
            paths.normalize_delays = False
            a, tau = paths.cir(reflection=True,scattering=True)
        else:
            paths = scene.compute_paths(max_depth=max_depth,
                                    num_samples=num_samples, reflection=False,scattering=False)
            paths.normalize_delays = False
            a, tau = paths.cir(reflection=False,scattering=False)

        h = cir_to_ofdm_channel(frequencies,
                                a,
                                tau,
                                normalize=False) # Non-normalized includes path-loss
        return h

    def combine_channel_matrix_all_pol(self,h1,h2,phases):
        # pylint: disable=line-too-long
        r"""
            Combines the channels from step 1 and step 2 into a cumulative channel from RIS

        Input
        -------
        h1 : [batch size, num_ris, num_ris_ant, num_tx, num_tx_ant, num_time_steps, fft_size], tf.complex64
            Tx to RIS channel

        h2 : [batch size, num_rx, num_rx_ant, num_ris, num_ris_ant, num_time_steps, fft_size], tf.complex64
            RIS to Rx channel

        phases : [array_size] tf.complex64
            Phases to be used at each RIS antenna

        Output
        -------
        h : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size]
            Cumulative channel between Tx to Rx via RIS
        """
        t_vv = self.element_transfer_matrix[0,0]
        t_hv = self.element_transfer_matrix[0,1]
        t_vh = self.element_transfer_matrix[1,0]
        t_hh = self.element_transfer_matrix[1,1]

        h1_v = h1[:,:,0:self.ris_array.array_size,:,:,:,:] # Select the V polarized channels in the RIS
        h1_h = h1[:,:,self.ris_array.array_size:,:,:,:,:] # Select the H polarized channels in the RIS

        h2_v = h2[:,:,:,:,0:self.ris_array.array_size,:,:] # Select the V polarized channels in the RIS
        h2_h = h2[:,:,:,:,self.ris_array.array_size:,:,:] # Select the H polarized channels in the RIS

        h = self.compute_channel_matrix_per_pol(h1_v,h2_v,phases,t_vv) + \
            self.compute_channel_matrix_per_pol(h1_h,h2_v,phases,t_hv) + \
            self.compute_channel_matrix_per_pol(h1_v,h2_h,phases,t_vh) + \
            self.compute_channel_matrix_per_pol(h1_h,h2_h,phases,t_hh)

        return h

    def compute_channel_matrix_per_pol(self,h1,h2,phases,t_xy):
        # pylint: disable=line-too-long
        r"""
        Input
        -------
        h1 : [batch size, num_ris, num_ris_ant, num_tx, num_tx_ant, num_time_steps, fft_size], tf.complex64
            Tx to RIS channel either V/H polarization

        h2 : [batch size, num_rx, num_rx_ant, num_ris, num_ris_ant, num_time_steps, fft_size], tf.complex64
            RIS to Rx channel either V/H polarization

        phases : [array_size] tf.complex64
            Phases to be used at each RIS antenna

        Output
        -------
        h : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size]
            Cumulative channel between Tx to Rx via RIS
        """
        # Use Einstein summation notation to do matrix multiplications.
        h = t_xy*tf.einsum('ijalbno,b,b,ijblcno -> ijalcno',h2,self.ris_element_gains,phases,h1)
        return h

    def find_optimal_phases_based_on_path_length(self,scene):
        # pylint: disable=line-too-long
        r"""
        Use path length difference to compute the optimal phase at each antenna. This maximizes
        the power reflected from the RIS

        Input
        -------
        scene: class:`~sionna.rt.AntennaArray`

        Output
        -------
        optimal_phasors : [array_size], tf.complex
            Phase to maximize the reflected signal power from RIS
        """
        ris_position = self.position
        ris_antenna_rel_positions = self.ris_array.positions
        tx_position = scene.transmitters['tx'].position
        rx_position = scene.receivers['rx'].position

        # Path length vectors from Tx to RIS and Rx to RIS
        tx_ris_vec = (ris_position - tx_position) + ris_antenna_rel_positions
        rx_ris_vec = (ris_position - rx_position) + ris_antenna_rel_positions

        # Path lengths from Tx to RIS and Rx to RIS
        tx_ris_sep = tf.norm(tx_ris_vec, ord=2, axis = 1)
        rx_ris_sep = tf.norm(rx_ris_vec, ord=2, axis = 1)

        k_wave_number = tf.cast(2*PI/scene.wavelength,tf.float32)
        optimal_phases = k_wave_number*(tx_ris_sep+rx_ris_sep)
        optimal_phasors = tf.exp(1j*tf.cast(optimal_phases,tf.complex64))

        return optimal_phasors

    def compute_channel_to_from_ris_v2(self,scene,frequencies,env_flag):
        # pylint: disable=line-too-long
        r"""
        (Still under development)
        This function aims to reduce the compute time in step 2. Typically RIS is a
        very big antenna array compared to Rx array which results in large compute time
        when synthetic array is set to False.

        To reduce the compute time, we use the channel reciprocity to estimate
        the channel from the RIS to Rx by swapping the transmitter and receiver roles.

        Assign the RIS array as a receiver and setting the Rx as a transmitter thus
        reducing the number of ray tracing sources and the compute time.

        """

        rx_before_swap = []
        ris_node_before_swap = []
        rx_array_before_swap = []
        reverse_flag = False

        # Make the Rx a ray tracing source as it has less number of antennas
        # and in effect reduce the ray tracing compute time
        if scene.synthetic_array is False and self.ris_array.array_size > scene.rx_array.array_size:
            rx_before_swap = scene.get("rx")
            ris_node_before_swap = scene.get("ris")
            rx_array_before_swap  = scene.rx_array
            scene.tx_array = scene.rx_array
            scene.rx_array = self.ris_array
            scene.remove("rx")
            scene.remove("ris")

            rx_swapped_as_tx = Transmitter(name = "rx_swapped_as_tx",position=rx_before_swap.position,orientation=rx_before_swap.orientation)
            ris_swapped_as_rx = Receiver(name = "ris_swapped_as_rx",position=ris_node_before_swap.position,orientation=ris_node_before_swap.orientation)

            scene.add(rx_swapped_as_tx)
            scene.add(ris_swapped_as_rx)

            reverse_flag = True # Flag is used to reverse the tx and rx dimensions in the tensor flow array

        if env_flag is True:
            paths = scene.compute_paths(max_depth=1,
                                    num_samples=10e6, reflection=True,scattering=True)
            paths.normalize_delays = False
            paths.reverse_direction = reverse_flag
            a, tau = paths.cir(reflection=True,scattering=True)
        else:
            paths = scene.compute_paths(max_depth=1,
                                    num_samples=10e6, reflection=False,scattering=False)
            paths.normalize_delays = False
            paths.reverse_direction = reverse_flag
            a, tau = paths.cir(reflection=False,scattering=False)

        h = cir_to_ofdm_channel(frequencies,
                                a,
                                tau,
                                normalize=False) # Non-normalized includes path-loss

        # Change the scene back to how it was before swapping the ris and rx
        if scene.synthetic_array is False and self.ris_array.array_size > scene.rx_array.array_size:
            scene.remove("rx_swapped_as_tx")
            scene.remove("ris_swapped_as_rx")
            scene.rx_array = rx_array_before_swap
            scene.tx_array = self.ris_array
            scene.add(rx_before_swap)
            scene.add(ris_node_before_swap)
        return h

