import os
gpu_num = 1 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Colab does currently not support the latest version of ipython.
# Thus, the preview does not work in Colab. However, whenever possible we 
# strongly recommend to use the scene preview mode.
resolution = [480,320] # increase for higher quality of renderings

# Allows to exit cell execution in Jupyter
class ExitCell(Exception):
    def _render_traceback_(self):
        pass

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        tf.config.set_visible_devices(gpus, 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e) 
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

tf.random.set_seed(1) # Set global random seed for reproducibility

# Import Sionna
import sys
print(os.getcwd())
os.chdir('examples')
sys.path.append('../')
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    print('no sionna')
    # os.system("pip install sionna")
    # import sionna

import matplotlib.pyplot as plt
import numpy as np
import sys
    
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RadioMaterial, Camera
from sionna.rt.utils import r_hat
from sionna.constants import PI, SPEED_OF_LIGHT
from sionna.utils import expand_to_rank

if __name__ == "__main__":
    # scene = load_scene(sionna.rt.scene.simple_thickwall) 
    scene = load_scene("/afs/cs.pitt.edu/usr0/zha21/Documents/GitHub/diff-RT-channel-prediction/dataset/scene/princeton-3D/princeton-processed.xml")
    # scene = load_scene()
    # scene = load_scene(sionna.rt.scene.floor_wall)
    # scene = load_scene(sionna.rt.scene.simple_reflector) 
    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="tr38901",
                                polarization="V")

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="dipole",
                                polarization="V")


    # Create transmitter
    tx1 = Transmitter(name="tx1",
                    position=[1, 1, 2])

    tx2 = Transmitter(name="tx2",
                    position=[-1, 0, 2])

    # Add transmitter instance to scene
    scene.add(tx1)
    # scene.add(tx2)

    # Create a receiver
    rx1 = Receiver(name="rx1",
                position=[-1, 1, 1.5],
                orientation=[0,0,0])
    rx2 = Receiver(name="rx2",
                position=[-1, 2, 1.5],
                orientation=[0,0,0])


    # Add receiver instance to scene 
    scene.add(rx1)
    scene.add(rx2)

    scene.receivers['rx1'].position = tf.convert_to_tensor([-1.27, -6.4516,  1.3] )
    scene.transmitters['tx1'].position = [-1.27,  0,   1.3 ]

    tx1.look_at(rx1) # Transmitter points towards receiver
    tx2.look_at(rx1) 
    scene.synthetic_array = True # If set to False, ray tracing will be done per antenna element (slower for large arrays)
    traced_paths = scene.trace_paths(los=True, reflection=True, scattering=True, diffraction=True, refraction=True,
                                    max_depth=6,
                                    num_samples=int(1e6))
    paths = scene.compute_fields(*traced_paths, scat_random_phases=False)

    paths.normalize_delays = False
    # Print path information
    # [num_paths, num_transmitters, num_receivers, num_elements, num_samples]
    a, tau = paths.cir()
    print(a.shape, tau.shape)

