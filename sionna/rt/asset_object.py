"""
Base class for asset objects.
"""

#from abc import abstractmethod
import numpy as np
import mitsuba as mi
from .oriented_object import OrientedObject


class AssetObject(OrientedObject):
    # pylint: disable=line-too-long

    # The convention of Mitsuba for camera is Y as up and look toward Z+.
    # However, Sionna uses Z as up and looks toward X+, for consistency
    # with radio devices.
    # The following transform peforms a rotation to ensure Sionna's
    # convention.
    # Note: Mitsuba uses degrees
    mi_2_sionna = ( mi.ScalarTransform4f.rotate([0,0,1], 90.0)
                @ mi.ScalarTransform4f.rotate([1,0,0], 90.0) )
    

    def __init__(self, name, filename, position=(0.,0.,0.), orientation=(0.,0.,0.), look_at=None):
        #TODO: 
        #- Asset should be associated with as ingle shape + bsdf. check that or provide support for nested asset definition?
        #- Proper implem of position,orientation, etc. setter, getter
        #- Add look at support 
        #- Add Rotation, Orientation modification support 
        #- Set scale parameter transform?
        
        # Keep track of the "to world" transform.
        # Initialized to identity.
        self._to_world = mi.ScalarTransform4f()

        # Initialize shape ids associated with asset
        self._shape_ids = [] #Attributed when added to a scene

        # Asset's XML and meshes sources directory
        self.filename = filename

        # Initialize the base class OrientedObject
        # Position and orientation are set through this call
        super().__init__(name, position, orientation, look_at)

        # self.position = position
        # if look_at is None:
        #     self.orientation = orientation
        # else:
        #     self.look_at(look_at)


        #super().__init__(name)

    def look_at(self, target):
        r"""
        Sets the orientation so that the camera looks at a position, radio
        device, or another camera.

        Given a point :math:`\mathbf{x}\in\mathbb{R}^3` with spherical angles
        :math:`\theta` and :math:`\varphi`, the orientation of the camera
        will be set equal to :math:`(\varphi, \frac{\pi}{2}-\theta, 0.0)`.

        Input
        -----
        target : [3], float | :class:`~sionna.rt.Transmitter` | :class:`~sionna.rt.Receiver` | :class:`~sionna.rt.Camera` | str
            A position or the name or instance of a
            :class:`~sionna.rt.Transmitter`, :class:`~sionna.rt.Receiver`, or
            :class:`~sionna.rt.Camera` in the scene to look at.
        """
        # Get position to look at
        if isinstance(target, str):
            if self.scene is None:
                msg = f"Cannot look for radio device '{target}' as the camera"\
                       " is not part of the scene"
                raise ValueError(msg)
            item = self.scene.get(target)
            if not isinstance(item, OrientedObject):
                msg = f"No radio device or camera named '{target}' found."
                raise ValueError(msg)
            else:
                target = item.position.numpy()
        else:
            target = np.array(target).astype(float)
            if not ( (target.ndim == 1) and (target.shape[0] == 3) ):
                raise ValueError("`x` must be a three-element vector)")

        # If the position and the target are on a line that is parallel to z,
        # then the look-at transform is ill-defined as z is up.
        # In this case, we add a small epsilon to x to avoid this.
        if np.allclose(self.position[:2], target[:2]):
            target[0] = target[0] + 1e-3
        # Look-at transform
        trf = mi.ScalarTransform4f.look_at(self.position, target,
                                           [0.0, 0.0, 1.0]) # Sionna uses Z-up
        # Set the rotation matrix of the Mitsuba sensor
        self._to_world = trf

    @property
    def shape_ids(self):
        return self._shape_ids

    @shape_ids.setter
    def shape_ids(self, shape_ids):
        self._shape_ids = shape_ids
        #if self._scene != None:
        #    self._scene.update(self)

    @property
    def position(self):
        """
        [3], float : Get/set the position :math:`(x,y,z)` as three-dimensional
            vector
        """
        return self._position
        #return AssetObject.world_to_position(self._to_world)

    @position.setter
    def position(self, position):
        self._position = position
        # new_position = np.array(new_position)
        # if not (new_position.ndim == 1 and new_position.shape[0] == 3):
        #     msg = "Position must be shaped as [x,y,z] (rank=1 and shape=[3])"
        #     raise ValueError(msg)
        # # Update transform
        # to_world = self._to_world.matrix.numpy()
        # to_world[:3,3] = new_position
        # self._to_world = mi.ScalarTransform4f(to_world)

    @property
    def orientation(self):
        r"""
        [3], float : Get/set the orientation :math:`(\alpha, \beta, \gamma)`
            specified through three angles corresponding to a 3D rotation
            as defined in :eq:`rotation`.
        """
        return self._orientation
        #return AssetObject.world_to_angles(self._to_world)

    @orientation.setter
    def orientation(self, orientation):
        self._orientation = orientation
    #     new_orientation = np.array(new_orientation)
    #     if not (new_orientation.ndim == 1 and new_orientation.shape[0] == 3):
    #         msg = "Orientation must be shaped as [a,b,c] (rank=1 and shape=[3])"
    #         raise ValueError(msg)

    #     # Mitsuba transform
    #     # Note: Mitsuba uses degrees
    #     new_orientation = new_orientation*180.0/np.pi
    #     rot_x = mi.ScalarTransform4f.rotate([1,0,0], new_orientation[2])
    #     rot_y = mi.ScalarTransform4f.rotate([0,1,0], new_orientation[1])
    #     rot_z = mi.ScalarTransform4f.rotate([0,0,1], new_orientation[0])
    #     rot_mat = rot_z@rot_y@rot_x@AssetObject.mi_2_sionna
    #     # Translation to keep the current position
    #     trs = mi.ScalarTransform4f.translate(self.position)
    #     to_world = trs@rot_mat
    #     # Update in Mitsuba
    #     self._to_world = to_world

    def look_at(self, target):
        r"""
        Sets the orientation so that the AssetObject looks at a position, radio
        device, camera or another AssetObject.

        Given a point :math:`\mathbf{x}\in\mathbb{R}^3` with spherical angles
        :math:`\theta` and :math:`\varphi`, the orientation of the AssetObject
        will be set equal to :math:`(\varphi, \frac{\pi}{2}-\theta, 0.0)`.

        Input
        -----
        target : [3], float | :class:`~sionna.rt.Transmitter` | :class:`~sionna.rt.Receiver` | :class:`~sionna.rt.Camera` | str
            A position or the name or instance of a
            :class:`~sionna.rt.Transmitter`, :class:`~sionna.rt.Receiver`, or
            :class:`~sionna.rt.Camera` in the scene to look at.
        """
        # Get position to look at
        if isinstance(target, str):
            if self.scene is None:
                msg = f"Cannot look for radio device '{target}' as the camera"\
                       " is not part of the scene"
                raise ValueError(msg)
            item = self.scene.get(target)
            if not isinstance(item, OrientedObject):
                msg = f"No radio device, camera or AssetObject named '{target}' found."
                raise ValueError(msg)
            else:
                target = item.position.numpy()
        else:
            target = np.array(target).astype(float)
            if not ( (target.ndim == 1) and (target.shape[0] == 3) ):
                raise ValueError("`x` must be a three-element vector)")

        # If the position and the target are on a line that is parallel to z,
        # then the look-at transform is ill-defined as z is up.
        # In this case, we add a small epsilon to x to avoid this.
        if np.allclose(self.position[:2], target[:2]):
            target[0] = target[0] + 1e-3
        # Look-at transform
        trf = mi.ScalarTransform4f.look_at(self.position, target,
                                           [0.0, 0.0, 1.0]) # Sionna uses Z-up
        # Set the rotation matrix of the Mitsuba sensor
        self._to_world = trf

    ##############################################
    # Internal methods and class functions.
    # Should not be appear in the end user
    # documentation
    ##############################################

    @property
    def world_transform(self):
        return self._to_world

    @staticmethod
    def world_to_angles(to_world):
        """
        Extract the orientation angles corresponding to a ``to_world`` transform

        Input
        ------
        to_world : :class:`~mitsuba.ScalarTransform4f`
            Transform.

        Output
        -------
        : [3], float
            Orientation angles `[a,b,c]`.
        """

        # Undo the rotation to switch from Mitsuba to Sionna convention
        to_world = to_world@AssetObject.mi_2_sionna.inverse()

        # Extract the rotation matrix
        to_world = to_world.matrix.numpy()
        if to_world.ndim == 3:
            to_world = to_world[0]
        r_mat = to_world[:3,:3]

        # Compute angles
        x_ang = np.arctan2(r_mat[2,1], r_mat[2,2])
        y_ang = np.arctan2(-r_mat[2,0],
                        np.sqrt(np.square(r_mat[2,1]) + np.square(r_mat[2,2])))
        z_ang = np.arctan2(r_mat[1,0], r_mat[0,0])

        return np.array([z_ang, y_ang, x_ang])

    @staticmethod
    def world_to_position(to_world):
        """
        Extract the position corresponding to a ``to_world`` transform

        Input
        ------
        to_world : :class:`~mitsuba.ScalarTransform4f`
            Transform.

        Output
        -------
        : [3], float
            Position `[x,y,z]`.
        """

        to_world = to_world.matrix.numpy()
        if to_world.ndim == 3:
            to_world = to_world[0]
        position = to_world[:3,3]
        return position
'''
def load_scene(filename=None, dtype=tf.complex64):
    # pylint: disable=line-too-long
    r"""
    Load a scene from file

    Note that only one scene can be loaded at a time.

    Input
    -----
    filename : str
        Name of a valid scene file. Sionna uses the simple XML-based format
        from `Mitsuba 3 <https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html>`_.
        Defaults to `None` for which an empty scene is created.

    dtype : tf.complex
        Dtype used for all internal computations and outputs.
        Defaults to `tf.complex64`.

    Output
    ------
    scene : :class:`~sionna.rt.Scene`
        Reference to the current scene
    """
    # Create empty scene using the reserved filename "__empty__"
    if filename is None:
        filename = "__empty__"
    return Scene(filename, dtype=dtype)

#
# Module variables for example scene files
#
floor_wall = str(files(scenes).joinpath("floor_wall/floor_wall.xml"))
# pylint: disable=C0301
"""
Example scene containing a ground plane and a vertical wall

.. figure:: ../figures/floor_wall.png
   :align: center
"""
'''