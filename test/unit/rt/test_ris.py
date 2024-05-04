try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("..")
    import sionna
    
import unittest
import tensorflow as tf

from sionna.rt import *
from sionna.channel import subcarrier_frequencies

gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)
        
class TestRIS(unittest.TestCase):
    def test_ris_scene_undisturbed(self):
        """Test adding RIS into the scene and verify that scene objects 
        remain unchanged after simulating RIS channel"""
        
        scene = load_scene()
        tx = Transmitter("tx", [1,2,-3], [0, 0, 0])
        rx = Receiver("rx", [0,0,0], [0, 0, 0]) 
        tx_array = PlanarArray(num_rows=2,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="dipole",
                                    polarization="V")
        rx_array = PlanarArray(num_rows=2,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="dipole",
                                    polarization="V")
        scene.rx_array = rx_array
        scene.tx_array = tx_array
        scene.add(tx)
        scene.add(rx)
        # Introduce RIS
        ris_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="dipole",
                                    polarization="VH")
        ris = RIS(name="ris", position = [-5,0,2.5],ris_array=ris_array)
        frequencies = subcarrier_frequencies(1, 1e3)
        ris.ris_channel(scene,frequencies,tx_to_ris_env=False,
                        rx_to_ris_env=False,phase_optimizer=False)
        
        self.assertTrue(len(scene.transmitters)==1)
        self.assertTrue(len(scene.receivers)==1) 
        
        self.assertTrue(tx_array == scene.tx_array)
        self.assertTrue(rx_array == scene.rx_array)
         
        scene.remove("tx")
        scene.remove("rx")     
        
        self.assertTrue(len(scene.transmitters)==0)
        self.assertTrue(len(scene.receivers)==0)
        