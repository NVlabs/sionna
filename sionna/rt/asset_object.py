#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Abstract Base class for cameras, radios, and scene objects.
"""

from abc import abstractmethod
from .oriented_object import Object


class AssetObject(Object):
    # pylint: disable=line-too-long
    r"""OrientedObject(name, position, orientation=[0.,0.,0.], look_at=None)

    Baseclass for implementing physical objects that are part of the scene and with
    editable localization and orientation.

    Input
    -----
    name : str
        Name of the object

    filename : str
        Name of a valid asset file. Sionna uses the simple XML-based format
        from `Mitsuba 3 <https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html>`_.

    position : [3], float
        Position :math:`(x,y,z)` as three-dimensional vector

    orientation : [3], float
        Orientation :math:`(\alpha, \beta, \gamma)` specified
        through three angles corresponding to a 3D rotation
        as defined in :eq:`rotation`.
        This parameter is ignored if ``look_at`` is not `None`.
        Defaults to [0,0,0].

    look_at : [3], float | :class:`sionna.rt.OrientedObject` | None
        A position or the instance of an :class:`~sionna.rt.OrientedObject` to
        point toward to.
        If set to `None`, then ``orientation`` is used to orientate the object.
    """

    def __init__(self, name, filename, position=(0.,0.,0.), orientation=(0.,0.,0.), look_at=None):
        #TODO: Asset should be associated with as ingle shape + bsdf. check that or provide support for nested asset definition?
        self.position = position
        self._id = None #Attributed when added to a scene?
        if look_at is None:
            self.orientation = orientation
        else:
            self.look_at(look_at)

        self.filename = filename

        super().__init__(name)

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        self._id = id
        #if self._scene != None:
        #    self._scene.update(self)


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