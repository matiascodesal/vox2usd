"""Mesh generating algorithms

This module contains classes for different meshing algorithms.
"""
from vox2usd.constants import VoxelSides
from vox2usd.vox import VoxModel, VoxBaseMaterial, VoxGlassMaterial


class MeshingBase(object):
    @staticmethod
    def generate(voxel_size):
        raise NotImplementedError()


class SimpleMeshing(MeshingBase):
    """An algorithm that involves culling hidden faces of voxel cubes."""
    @staticmethod
    def generate(voxel_size):
        for model in VoxModel.get_all():
            pos_sorted_voxels = {}
            mtl_sorted_voxels = {}
            for voxel in model.voxels:
                position_hash = tuple(voxel[:3])
                mtl_id = voxel[3]
                pos_sorted_voxels[position_hash] = VoxBaseMaterial.get(mtl_id)
                if mtl_id not in mtl_sorted_voxels:
                    mtl_sorted_voxels[mtl_id] = []
                mtl_sorted_voxels[mtl_id].append(voxel)
            for mtl_id, voxels in mtl_sorted_voxels.items():
                model.meshes[mtl_id] = []
                # Because, MV only translates on whole values, we need to round or floor the for odd numbered dimensions.
                model_half_x = int(model.size[0] / 2.0)
                model_half_y = int(model.size[1] / 2.0)
                model_half_z = int(model.size[2] / 2.0)
                working_is_glass = type(VoxBaseMaterial.get(mtl_id)) == VoxGlassMaterial
                for voxel in voxels:
                    front = (voxel[0], voxel[1] - 1, voxel[2])
                    if front not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[front]) == VoxGlassMaterial and not working_is_glass):
                        front_face = [(voxel[0] - model_half_x, voxel[1] - model_half_y, voxel[2] - model_half_z),
                                      (voxel[0] + voxel_size - model_half_x, voxel[1] - model_half_y,
                                       voxel[2] - model_half_z),
                                      (voxel[0] + voxel_size - model_half_x, voxel[1] - model_half_y,
                                       voxel[2] + voxel_size - model_half_z),
                                      (voxel[0] - model_half_x, voxel[1] - model_half_y,
                                       voxel[2] + voxel_size - model_half_z)]

                        model.meshes[mtl_id].extend(front_face)

                    back = (voxel[0], voxel[1] + 1, voxel[2])
                    if back not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[back]) == VoxGlassMaterial and not working_is_glass):
                        back_face = [(voxel[0] + voxel_size - model_half_x,
                                      voxel[1] + voxel_size - model_half_y, voxel[2] - model_half_z),
                                     (voxel[0] - model_half_x, voxel[1] + voxel_size - model_half_y,
                                      voxel[2] - model_half_z),
                                     (voxel[0] - model_half_x, voxel[1] + voxel_size - model_half_y,
                                      voxel[2] + voxel_size - model_half_z),
                                     (voxel[0] + voxel_size - model_half_x,
                                      voxel[1] + voxel_size - model_half_y,
                                      voxel[2] + voxel_size - model_half_z)]
                        model.meshes[mtl_id].extend(back_face)

                    right = (voxel[0] + 1, voxel[1], voxel[2])
                    if right not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[right]) == VoxGlassMaterial and not working_is_glass):
                        right_face = [(voxel[0] + voxel_size - model_half_x, voxel[1] - model_half_y,
                                       voxel[2] - model_half_z),
                                      (voxel[0] + voxel_size - model_half_x,
                                       voxel[1] + voxel_size - model_half_y, voxel[2] - model_half_z),
                                      (voxel[0] + voxel_size - model_half_x,
                                       voxel[1] + voxel_size - model_half_y,
                                       voxel[2] + voxel_size - model_half_z),
                                      (voxel[0] + voxel_size - model_half_x, voxel[1] - model_half_y,
                                       voxel[2] + voxel_size - model_half_z)]
                        model.meshes[mtl_id].extend(right_face)

                    left = (voxel[0] - 1, voxel[1], voxel[2])
                    if left not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[left]) == VoxGlassMaterial and not working_is_glass):
                        left_face = [(voxel[0] - model_half_x, voxel[1] + voxel_size - model_half_y,
                                      voxel[2] - model_half_z),
                                     (voxel[0] - model_half_x, voxel[1] - model_half_y, voxel[2] - model_half_z),
                                     (voxel[0] - model_half_x, voxel[1] - model_half_y,
                                      voxel[2] + voxel_size - model_half_z),
                                     (voxel[0] - model_half_x, voxel[1] + voxel_size - model_half_y,
                                      voxel[2] + voxel_size - model_half_z)]
                        model.meshes[mtl_id].extend(left_face)

                    top = (voxel[0], voxel[1], voxel[2] + 1)
                    if top not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[top]) == VoxGlassMaterial and not working_is_glass):
                        top_face = [(voxel[0] - model_half_x, voxel[1] - model_half_y,
                                     voxel[2] + voxel_size - model_half_z),
                                    (voxel[0] + voxel_size - model_half_x, voxel[1] - model_half_y,
                                     voxel[2] + voxel_size - model_half_z),
                                    (voxel[0] + voxel_size - model_half_x,
                                     voxel[1] + voxel_size - model_half_y,
                                     voxel[2] + voxel_size - model_half_z),
                                    (voxel[0] - model_half_x, voxel[1] + voxel_size - model_half_y,
                                     voxel[2] + voxel_size - model_half_z)]
                        model.meshes[mtl_id].extend(top_face)

                    bottom = (voxel[0], voxel[1], voxel[2] - 1)
                    if bottom not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[bottom]) == VoxGlassMaterial and not working_is_glass):
                        bottom_face = [
                            (voxel[0] + voxel_size - model_half_x, voxel[1] - model_half_y,
                             voxel[2] - model_half_z),
                            (voxel[0] - model_half_x, voxel[1] - model_half_y, voxel[2] - model_half_z),
                            (voxel[0] - model_half_x, voxel[1] + voxel_size - model_half_y,
                             voxel[2] - model_half_z),
                            (voxel[0] + voxel_size - model_half_x, voxel[1] + voxel_size - model_half_y,
                             voxel[2] - model_half_z)]
                        model.meshes[mtl_id].extend(bottom_face)


class GreedyMeshing(MeshingBase):
    """An algorithm that involves merging adjacent faces of adjacent voxels

    Traverses the voxel space a plane at a time in each of the 6 directions. For each plane,
    it will start with the top left voxel face and traverse left to right and top to bottom combining
    adjacent voxel faces of the same material.
    """
    @staticmethod
    def generate(voxel_size):
        for model in VoxModel.get_all():
            pos_sorted_voxels, mtl_sorted_voxels = GreedyMeshing._sort_voxels(model)
            for mtl_id, voxels in mtl_sorted_voxels.items():
                model.meshes[mtl_id] = {}
                working_is_glass = type(VoxBaseMaterial.get(mtl_id)) == VoxGlassMaterial
                for voxel in voxels:
                    for side in VoxelSides.values():
                        GreedyMeshing._set_voxel_face_visibility(voxel, side, model, pos_sorted_voxels,
                                                                 mtl_id, working_is_glass)

                greedy_faces = []
                run_start = None
                run_width = 0
                last_voxel = None

                # Traverse the voxel space from all six sides by scanning a face plane
                # at a time push forward through the voxel space.
                for side in VoxelSides.values():
                    uvw_dimensions = GreedyMeshing._get_uvw_dimensions(model, side)
                    for w in range(uvw_dimensions[2]+1):
                        for v in range(uvw_dimensions[1]+1):
                            for u in range(uvw_dimensions[0]+1):
                                if run_start is None and GreedyMeshing._can_merge_face(model, mtl_id, u, v, w, side):
                                    # Set the starting voxel if we haven't started a run yet.
                                    run_start = (u,v)
                                elif run_start and not GreedyMeshing._can_merge_face(model, mtl_id, u, v, w, side):
                                    # If I can no longer merge faces horizontally, start scanning vertically.
                                    for run_v in range(v + 1, uvw_dimensions[1]+1):
                                        row_complete = True
                                        # Scan a row at a time based on the width of the first scanned row in the run
                                        for run_u in range(run_start[0], run_start[0] + run_width):
                                            if not GreedyMeshing._can_merge_face(model, mtl_id, run_u, run_v, w, side):
                                                row_complete = False
                                                break
                                        if row_complete:
                                            last_voxel = (run_start[0] + run_width - 1, run_v)
                                        else:
                                            merged_face = GreedyMeshing._create_merged_face(model, voxel_size,
                                                                                            run_start, last_voxel,
                                                                                            w, side)
                                            greedy_faces.extend(merged_face)
                                            # pop off any faces that we just merged so that they aren't counted when we
                                            # continue scanning.
                                            for pop_v in range(run_start[1], last_voxel[1] + 1):
                                                for pop_u in range(run_start[0], run_start[0] + run_width):
                                                    GreedyMeshing._remove_voxel_face(model, mtl_id, pop_u,
                                                                                     pop_v, w, side)
                                            run_start = None
                                            run_width = 0
                                            last_voxel = None
                                            break

                                if run_start is not None:
                                    run_width += 1
                                    last_voxel = (u, v)

                model.meshes[mtl_id] = greedy_faces

    @staticmethod
    def _set_voxel_face_visibility(voxel, side, model, pos_sorted_voxels, mtl_id, working_is_glass):
        """Sets whether the voxel face is visible on the model.

        Non-visible faces have an adjacent voxel that is not glass so it is thereby occluded.

        Args:
            voxel: The voxel to set face visibility for.
            side: The voxel face being set.
            model (VoxModel): The source of the model voxel data.
            pos_sorted_voxels (dict): All voxels sorted by position.
            mtl_id: The material id of the current voxel.
            working_is_glass (bool): Whether the current voxel is using a glass material.
        """
        neighbor_pos = GreedyMeshing._get_voxel_neighbor(voxel, side)
        if neighbor_pos not in pos_sorted_voxels or (
                type(pos_sorted_voxels[neighbor_pos]) == VoxGlassMaterial and not working_is_glass):
            front_face = (voxel[0], voxel[1], voxel[2], side)
            model.meshes[mtl_id][front_face] = True

    @staticmethod
    def _get_voxel_neighbor(voxel, side):
        """Get the position of the neighboring voxel on the given side."""
        if side == VoxelSides.FRONT:
            return voxel[0], voxel[1] - 1, voxel[2]
        elif side == VoxelSides.BACK:
            return voxel[0], voxel[1] + 1, voxel[2]
        elif side == VoxelSides.RIGHT:
            return voxel[0] + 1, voxel[1], voxel[2]
        elif side == VoxelSides.LEFT:
            return voxel[0] - 1, voxel[1], voxel[2]
        elif side == VoxelSides.TOP:
            return voxel[0], voxel[1], voxel[2] + 1
        elif side == VoxelSides.BOTTOM:
            return voxel[0], voxel[1], voxel[2] - 1

    @staticmethod
    def _sort_voxels(model):
        """Put voxels into dictionaries sorted by position and mtl_id.

        This allows us to be able to quickly query a voxel by either of those properties. A tuple
        of the voxel position is used for the key of the position sorted dict and the mtl_id is used as the
        key for the material sorted dict.

        Args:
            model (VoxModel): The source of the model voxel data.

        Returns:
            dict, dict: The position sorted voxels and the material sorted voxels.
        """
        #
        #
        pos_sorted_voxels = {}
        mtl_sorted_voxels = {}
        for voxel in model.voxels:
            position_str = tuple(voxel[:3])
            mtl_id = voxel[3]
            pos_sorted_voxels[position_str] = VoxBaseMaterial.get(mtl_id)
            if mtl_id not in mtl_sorted_voxels:
                mtl_sorted_voxels[mtl_id] = []
            mtl_sorted_voxels[mtl_id].append(voxel)

        return pos_sorted_voxels, mtl_sorted_voxels


    @staticmethod
    def _can_merge_face(model, mtl_id, u, v, w, side):
        """Returns whether the voxel face at uvw can be merged to the current mesh run.

        If the voxel face exists and has not be merged already and shares the
        same material as the current mesh run, it can be merged.

        Args:
            model (VoxModel): The source of the model voxel data.
            mtl_id: The material id currently being meshed.
            u: The u position of the face being queried.
            v: The v position of the face being queried.
            w: The w position of the face being queried.
            side: The side being meshed.

        Returns:
            bool: Whether the queried voxel face should be merged with the current mesh run.
        """
        if side == VoxelSides.FRONT:
            return (u, w, v, side) in model.meshes[mtl_id]
        elif side == VoxelSides.BACK:
            return (u, w, v, side) in model.meshes[mtl_id]
        elif side == VoxelSides.RIGHT:
            return (w, u, v, side) in model.meshes[mtl_id]
        elif side == VoxelSides.LEFT:
            return (w, u, v, side) in model.meshes[mtl_id]
        elif side == VoxelSides.TOP:
            return (u, v, w, side) in model.meshes[mtl_id]
        elif side == VoxelSides.BOTTOM:
            return (u, v, w, side) in model.meshes[mtl_id]

    @staticmethod
    def _create_merged_face(model, voxel_size, run_start, run_end, w, side):
        """ Create a single merged face.

        Also centers the model in X and Y. Merged faces are always rectangular.
        You can determine the points of a face based off the bounding rectangle
        created by the starting and ending faces of the mesh run.

        Args:
            model (VoxModel): The source of the model voxel data.
            voxel_size: The XYX dimension of a voxel.
            run_start: The starting voxel face of the meshing run.
            run_end: The ending voxel face of the meshing run.
            w: The depth at which to create the merged face.
            side: The voxel face side being meshed.

        Returns:
            list: The four vertices of the merged face in CCW winding order.
        """
        # MV model pivot is centered, so we need to center the generated mesh
        # by subtracting half the width, height, and depth.
        model_half_x = int(model.size[0] / 2.0)
        model_half_y = int(model.size[1] / 2.0)
        model_half_z = int(model.size[2] / 2.0)
        merged_face = None
        if side == VoxelSides.FRONT:
            merged_face = [
                (run_start[0], w, run_start[1]),
                (run_end[0] + voxel_size, w, run_start[1]),
                (run_end[0] + voxel_size, w, run_end[1] + voxel_size),
                (run_start[0], w, run_end[1] + voxel_size)
            ]
        elif side == VoxelSides.BACK:
            merged_face = [
                (run_end[0] + voxel_size, w + voxel_size, run_start[1]),
                (run_start[0], w + voxel_size, run_start[1]),
                (run_start[0], w + voxel_size, run_end[1] + voxel_size),
                (run_end[0] + voxel_size, w + voxel_size, run_end[1] + voxel_size)
            ]
        elif side == VoxelSides.RIGHT:
            merged_face = [
                (w + voxel_size, run_start[0], run_start[1]),
                (w + voxel_size, run_end[0] + voxel_size, run_start[1]),
                (w + voxel_size, run_end[0] + voxel_size, run_end[1] + voxel_size),
                (w + voxel_size, run_start[0], run_end[1] + voxel_size)
            ]
        elif side == VoxelSides.LEFT:
            merged_face = [
                (w, run_end[0] + voxel_size, run_start[1]),
                (w, run_start[0], run_start[1]),
                (w, run_start[0], run_end[1] + voxel_size),
                (w, run_end[0] + voxel_size, run_end[1] + voxel_size)
            ]
        elif side == VoxelSides.TOP:
            # TOP winding order
            # 1. Bottom Left
            # 2. Bottom Right
            # 3. Top Right
            # 4. Top Left
            merged_face = [
                (run_start[0], run_start[1], w + voxel_size),
                (run_end[0] + voxel_size, run_start[1], w + voxel_size),
                (run_end[0] + voxel_size, run_end[1] + voxel_size, w + voxel_size),
                (run_start[0], run_end[1] + voxel_size, w + voxel_size)
            ]
        elif side == VoxelSides.BOTTOM:
            merged_face = [
                (run_end[0] + voxel_size, run_start[1], w),
                (run_start[0], run_start[1], w),
                (run_start[0], run_end[1] + voxel_size, w),
                (run_end[0] + voxel_size, run_end[1] + voxel_size, w)
            ]

        for vert_id in range(len(merged_face)):
            merged_face[vert_id] = (
                merged_face[vert_id][0] - model_half_x,
                merged_face[vert_id][1] - model_half_y,
                merged_face[vert_id][2] - model_half_z
            )

        return merged_face

    @staticmethod
    def _remove_voxel_face(model, mtl_id, u, v, w, side):
        """Remove a voxel face that has already been merged.

        Args:
            model (VoxModel): The source of the model voxel data.
            mtl_id: The material id of the voxel face.
            u: The u position of the voxel face being removed.
            v: The v position of the voxel face being removed.
            w: The w position of the voxel face being removed.
            side: The side voxel face being removed.
        """
        if side == VoxelSides.FRONT:
            model.meshes[mtl_id].pop((u, w, v, side))
        elif side == VoxelSides.BACK:
            model.meshes[mtl_id].pop((u, w, v, side))
        elif side == VoxelSides.RIGHT:
            model.meshes[mtl_id].pop((w, u, v, side))
        elif side == VoxelSides.LEFT:
            model.meshes[mtl_id].pop((w, u, v, side))
        elif side == VoxelSides.TOP:
            model.meshes[mtl_id].pop((u, v, w, side))
        elif side == VoxelSides.BOTTOM:
            model.meshes[mtl_id].pop((u, v, w, side))

    @staticmethod
    def _get_uvw_dimensions(model, side):
        """ Translates model XYZ dimensions into meshing UVW dimensions.

        To make things simple, I define the meshing plane space as UVW
        so that it is consistent for every meshing direction. While facing
        the plane, U is the horizontal units of the plane, V is the
        vertical units of the plane, and Z is the depth position of the plane.

        Args:
            model: The model data containing dimensions.
            side: The voxel side currently being meshed.

        Returns:
            list: UVW dimensions of the meshing plane.
        """
        if side == VoxelSides.FRONT:
            return [model.size[0], model.size[2], model.size[1]]
        elif side == VoxelSides.BACK:
            return [model.size[0], model.size[2], model.size[1]]
        elif side == VoxelSides.RIGHT:
            return [model.size[1], model.size[2], model.size[0]]
        elif side == VoxelSides.LEFT:
            return [model.size[1], model.size[2], model.size[0]]
        elif side == VoxelSides.TOP:
            return [model.size[0], model.size[1], model.size[2]]
        elif side == VoxelSides.BOTTOM:
            return [model.size[0], model.size[1], model.size[2]]
