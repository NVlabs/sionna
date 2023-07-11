#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
3D scene and paths viewer
"""

import drjit as dr
import mitsuba as mi
import numpy as np
from ipywidgets.embed import embed_snippet
import pythreejs as p3s
import matplotlib

from .utils import paths_to_segments, scene_scale, rotate
from .renderer import coverage_map_color_mapping


class InteractiveDisplay:
    """
    Lightweight wrapper around the `pythreejs` library.

    Input
    -----
    resolution : [2], int
        Size of the viewer figure.

    fov : float
        Field of view, in degrees.

    background : str
        Background color in hex format prefixed by '#'.
    """

    def __init__(self, scene, resolution, fov, background):

        self._scene = scene
        self._disk_sprite = None

        # List of objects in the scene
        self._objects = []
        # Bounding box of the scene
        self._bbox = mi.ScalarBoundingBox3f()

        ####################################################
        # Setup the viewer
        ####################################################

        # Lighting
        ambient_light = p3s.AmbientLight(intensity=0.80)
        camera_light = p3s.DirectionalLight(
            position=[0, 0, 0], intensity=0.25
        )

        # Camera & controls
        self._camera = p3s.PerspectiveCamera(
            fov=fov, aspect=resolution[0]/resolution[1],
            up=[0, 0, 1], far=10000,
            children=[camera_light],
        )
        self._orbit = p3s.OrbitControls(
            controlling = self._camera
        )

        # Scene & renderer
        self._p3s_scene = p3s.Scene(
            background=background, children=[self._camera, ambient_light]
        )
        self._renderer = p3s.Renderer(
            scene=self._p3s_scene, camera=self._camera, controls=[self._orbit],
            width=resolution[0], height=resolution[1], antialias=True
        )

        ####################################################
        # Plot the scene geometry
        ####################################################
        self.plot_scene()

        # Finally, ensure the camera is looking at the scene
        self.center_view()

    def reset(self):
        """
        Removes objects that are not flagged as persistent, i.e., the paths.
        """
        remaining = []
        for obj, persist in self._objects:
            if persist:
                remaining.append((obj, persist))
            else:
                self._p3s_scene.remove(obj)
        self._objects = remaining

    def center_view(self):
        """
        Automatically place the camera based on the scene's bounding box such
        that it is located at (-1, -1, 1) on the normalized bounding box, and
        oriented toward the center of the scene.
        """
        bbox = self._bbox if self._bbox.valid() else mi.ScalarBoundingBox3f(0.)
        center = bbox.center()

        corner = [bbox.min.x, center.y, 1.5 * bbox.max.z]
        if np.allclose(corner, 0):
            corner = (-1, -1, 1)
        self._camera.position = tuple(corner)

        self._camera.lookAt(center)
        self._orbit.exec_three_obj_method('update')
        self._camera.exec_three_obj_method('updateProjectionMatrix')

    def plot_radio_devices(self, show_orientations=False):
        """
        Plots the radio devices.

        Input
        -----
        show_orientations : bool
            Shows the radio devices' orientations.
            Defaults to `False`.
        """
        scene = self._scene
        sc, tx_positions, rx_positions, _ = scene_scale(scene)
        tr_color = [0.160, 0.502, 0.725]
        rc_color = [0.153, 0.682, 0.375]

        # Radio emitters, shown as points
        p = np.array(list(tx_positions.values()) + list(rx_positions.values()))
        albedo = np.array(
            [tr_color] * len(scene.transmitters)
            + [rc_color] * len(scene.receivers)
        )

        if p.shape[0] > 0:
            # Radio devices are not persistent
            radius = max(0.005 * sc, 1)
            self._plot_points(p, persist=False, colors=albedo,
                              radius=radius)
        if show_orientations:
            line_length = 0.0075 * sc
            head_length = 0.15 * line_length
            zeros = np.zeros((1, 3))

            for devices, color in [(scene.transmitters.values(), tr_color),
                                   (scene.receivers.values(), rc_color)]:
                if len(devices) == 0:
                    continue
                color = f'rgb({", ".join([str(int(v * 255)) for v in color])})'
                starts, ends = [], []
                for rd in devices:
                    # Arrow line
                    starts.append(rd.position)
                    endpoint = rd.position + rotate([line_length, 0., 0.],
                                                    rd.orientation)
                    ends.append(endpoint)

                    geo = p3s.CylinderGeometry(
                        radiusTop=0, radiusBottom=0.3 * head_length,
                        height=head_length, radialSegments=8,
                        heightSegments=0, openEnded=False)
                    mat = p3s.MeshLambertMaterial(color=color)
                    mesh = p3s.Mesh(geo, mat)
                    mesh.position = tuple(endpoint)
                    angles = rd.orientation.numpy()
                    mesh.rotateZ(angles[0] - np.pi/2)
                    mesh.rotateY(angles[2])
                    mesh.rotateX(-angles[1])
                    self._add_child(mesh, zeros, zeros, persist=False)

                self._plot_lines(np.array(starts), np.array(ends),
                                 width=2, color=color)

    def plot_paths(self, paths):
        """
        Plot the ``paths``.

        Input
        -----
        paths : :class:`~sionna.rt.Paths`
            Paths to plot
        """
        starts, ends = paths_to_segments(paths)
        if starts and ends:
            self._plot_lines(np.vstack(starts), np.vstack(ends))

    def plot_scene(self):
        """
        Plots the meshes that make the scene.
        """
        shapes = self._scene.mi_scene.shapes()
        n = len(shapes)
        if n <= 0:
            return

        palette = None
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wi = mi.Vector3f(0, 0, 1)

        # Shapes (e.g. buildings)
        vertices, faces, albedos = [None] * n, [None] * n, [None] * n
        f_offset = 0
        for i, s in enumerate(shapes):
            n_vertices = s.vertex_count()
            v = s.vertex_position(dr.arange(mi.UInt32, n_vertices))
            vertices[i] = v.numpy()
            f = s.face_indices(dr.arange(mi.UInt32, s.face_count()))
            faces[i] = f.numpy() + f_offset
            f_offset += n_vertices

            albedo = s.bsdf().eval_diffuse_reflectance(si).numpy()
            if not np.any(albedo > 0.):
                if palette is None:
                    palette = matplotlib.cm.get_cmap('Pastel1_r')
                albedo[:] = palette((i % palette.N + 0.5) / palette.N)[:3]

            albedos[i] = np.tile(albedo, (n_vertices, 1))

        # Plot all objects as a single PyThreeJS mesh, which is must faster
        # than creating individual mesh objects in large scenes.
        self._plot_mesh(np.concatenate(vertices, axis=0),
                        np.concatenate(faces, axis=0),
                        persist=True, # The scene geometry is persistent
                        colors=np.concatenate(albedos, axis=0))

    def plot_coverage_map(self, coverage_map, tx=0, db_scale=True,
                          vmin=None, vmax=None):
        """
        Plot the coverage map as a textured rectangle in the scene. Regions
        where the coverage map is zero-valued are made transparent.
        """
        to_world = coverage_map.to_world()
        # coverage_map = resample_to_corners(
        #     coverage_map[tx, :, :].numpy()
        # )
        coverage_map = coverage_map[tx, :, :].numpy()

        # Create a rectangle from two triangles
        p00 = to_world.transform_affine([-1, -1, 0])
        p01 = to_world.transform_affine([1, -1, 0])
        p10 = to_world.transform_affine([-1, 1, 0])
        p11 = to_world.transform_affine([1, 1, 0])

        vertices = np.array([p00, p01, p10, p11])
        pmin = np.min(vertices, axis=0)
        pmax = np.max(vertices, axis=0)

        faces = np.array([
            [0, 1, 2],
            [2, 1, 3],
        ], dtype=np.uint32)

        vertex_uvs = np.array([
            [0, 0], [1, 0], [0, 1], [1, 1]
        ], dtype=np.float32)

        geo = p3s.BufferGeometry(
            attributes={
                'position': p3s.BufferAttribute(vertices, normalized=False),
                'index': p3s.BufferAttribute(faces.ravel(), normalized=False),
                'uv': p3s.BufferAttribute(vertex_uvs, normalized=False),
            }
        )

        to_map, normalizer, color_map = coverage_map_color_mapping(
            coverage_map, db_scale=db_scale, vmin=vmin, vmax=vmax)
        texture = color_map(normalizer(to_map)).astype(np.float32)
        texture[:, :, 3] = (coverage_map > 0.).astype(np.float32)
        # Pre-multiply alpha
        texture[:, :, :3] *= texture[:, :, 3, None]

        texture = p3s.DataTexture(
            data=texture,
            format='RGBAFormat',
            type='FloatType',
            magFilter='NearestFilter',
            minFilter='NearestFilter',
        )

        mat = p3s.MeshLambertMaterial(
            side='DoubleSide',
            map=texture, transparent=True,
        )
        mesh = p3s.Mesh(geo, mat)

        self._add_child(mesh, pmin, pmax, persist=False)


    @property
    def camera(self):
        """
        pthreejs.PerspectiveCamera : Get the camera
        """
        return self._camera

    @property
    def orbit(self):
        """
        pthreejs.OrbitControls : Get the orbit
        """
        return self._orbit

    def resolution(self):
        """
        Returns a tuple (width, height) with the rendering resolution.
        """
        return (self._renderer.width, self._renderer.height)

    ##################################################
    # Internal methods
    ##################################################

    def _plot_mesh(self, vertices, faces, persist, colors=None):
        """
        Plots a mesh.

        Input
        ------
        vertices : [n,3], float
            Position of the vertices

        faces : [n,3], int
            Indices of the triangles associated with ``vertices``

        persist : bool
            Flag indicating if the mesh is persistent, i.e., should not be
            erased when ``reset()`` is called.

        colors : [n,3] | [3] | None
            Colors of the vertices. If `None`, black is used.
            Defaults to `None`.
        """
        assert vertices.ndim == 2 and vertices.shape[1] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3
        n_v = vertices.shape[0]
        pmin, pmax = np.min(vertices, axis=0), np.max(vertices, axis=0)

        # Assuming per-vertex colors
        if colors is None:
            # Black is default
            colors = np.zeros((n_v, 3), dtype=np.float32)
        elif colors.ndim == 1:
            colors = np.tile(colors[None, :], (n_v, 1))
        colors = colors.astype(np.float32)
        assert ( (colors.ndim == 2)
             and (colors.shape[1] == 3)
             and (colors.shape[0] == n_v) )

        # Closer match to Mitsuba and Blender
        colors = np.power(colors, 1/1.8)

        geo = p3s.BufferGeometry(
            attributes={
                'index': p3s.BufferAttribute(faces.ravel(), normalized=False),
                'position': p3s.BufferAttribute(vertices, normalized=False),
                'color': p3s.BufferAttribute(colors, normalized=False)
            }
        )

        mat = p3s.MeshStandardMaterial(
            side='DoubleSide', metalness=0., roughness=1.0,
            vertexColors='VertexColors', flatShading=True,
        )
        mesh = p3s.Mesh(geo, mat)
        self._add_child(mesh, pmin, pmax, persist=persist)

    def _plot_points(self, points, persist, colors=None, radius=0.05):
        """
        Plots a set of `n` points.

        Input
        -------
        points : [n, 3], float
            Coordinates of the `n` points.

        persist : bool
            Indicates if the points are persistent, i.e., should not be erased
            when ``reset()`` is called.

        colors : [n, 3], float | [3], float | None
            Colors of the points.

        radius : float
            Radius of the points.
        """
        assert points.ndim == 2 and points.shape[1] == 3
        n = points.shape[0]
        pmin, pmax = np.min(points, axis=0), np.max(points, axis=0)

        # Assuming per-vertex colors
        if colors is None:
            colors = np.zeros((n, 3), dtype=np.float32)
        elif colors.ndim == 1:
            colors = np.tile(colors[None, :], (n, 1))
        colors = colors.astype(np.float32)
        assert ( (colors.ndim == 2)
             and (colors.shape[1] == 3)
             and (colors.shape[0] == n) )

        tex = p3s.DataTexture(data=self._get_disk_sprite(), format="RGBAFormat",
                              type="FloatType")

        points = points.astype(np.float32)
        geo = p3s.BufferGeometry(attributes={
            'position': p3s.BufferAttribute(points, normalized=False),
            'color': p3s.BufferAttribute(colors, normalized=False),
        })
        mat = p3s.PointsMaterial(
            size=2*radius, sizeAttenuation=True, vertexColors='VertexColors',
            map=tex, alphaTest=0.5, transparent=True,
        )
        mesh = p3s.Points(geo, mat)
        self._add_child(mesh, pmin, pmax, persist=persist)

    def _add_child(self, obj, pmin, pmax, persist):
        """
        Adds an object for display

        Input
        ------
        obj : :class:`~pythreejs.Mesh`
            Mesh to display

        pmin : [3], float
            Lowest position for the bounding box

        pmax : [3], float
            Highest position for the bounding box

        persist : bool
            Flag that indicates if the object is persistent, i.e., if it should
            be removed from the display when `reset()` is called.
        """
        self._objects.append((obj, persist))
        self._p3s_scene.add(obj)

        self._bbox.expand(pmin)
        self._bbox.expand(pmax)

    def _plot_lines(self, starts, ends, width=0.5, color='black'):
        """
        Plots a set of `n` lines. This is used to plot the paths.

        Input
        ------
        starts : [n, 3], float
            Coordinates of the lines starting points

        ends : [n, 3], float
            Coordinates of the lines ending points

        width : float
            Width of the lines.
            Defaults to 0.5.

        color : str
            Color of the lines.
            Defaults to 'black'.
        """

        assert starts.ndim == 2 and starts.shape[1] == 3
        assert ends.ndim == 2 and ends.shape[1] == 3
        assert starts.shape[0] == ends.shape[0]

        segments = np.hstack((starts, ends)).astype(np.float32).reshape(-1,2,3)
        pmin = np.min(segments, axis=(0, 1))
        pmax = np.max(segments, axis=(0, 1))

        geo = p3s.LineSegmentsGeometry(positions=segments)
        mat = p3s.LineMaterial(linewidth=width, color=color)
        mesh = p3s.LineSegments2(geo, mat)

        # Lines are not flagged as persistent as they correspond to paths, which
        # can changes from one display to the next.
        self._add_child(mesh, pmin, pmax, persist=False)

    def _get_disk_sprite(self):
        """
        Returns the sprite used to represent transmitters and receivers though
        ``_plot_points()``.

        Output
        ------
        : [n,n,4], float
            Sprite
        """
        if self._disk_sprite is not None:
            return self._disk_sprite

        n = 128
        sprite = np.ones((n, n, 4))
        sprite[:, :, 3] = 0.
        # Draw a disk with an empty circle close to the edge
        ij = np.mgrid[:n, :n]
        ij = ij.reshape(2, -1)

        p = (ij + 0.5) / n - 0.5
        t = np.linalg.norm(p, axis=0).reshape((n, n))
        inside = t < 0.48
        in_band = (t < 0.45) & (t > 0.42)
        sprite[inside & (~in_band), 3] = 1.0

        sprite = sprite.astype(np.float32)
        self._disk_sprite = sprite
        return sprite

    ################################################
    # The following methods are required for
    # integration in Jupyter notebooks
    ################################################

    # pylint: disable=unused-argument
    def _repr_mimebundle_(self, **kwargs):
        # pylint: disable=protected-access,not-callable
        bundle = self._renderer._repr_mimebundle_()
        assert 'text/html' not in bundle
        bundle['text/html'] = self._repr_html_()
        return bundle

    def _repr_html_(self):
        """
        Standalone HTML display, i.e. outside of an interactive Jupyter
        notebook environment.
        """

        html = embed_snippet(self._renderer, requirejs=True)
        return html
