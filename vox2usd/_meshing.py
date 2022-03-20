
from vox2usd.constants import VoxelSides
from vox2usd.vox import VoxModel, VoxBaseMaterial, VoxGlassMaterial


class MeshingBase(object):
    @staticmethod
    def generate(voxel_size):
        raise NotImplementedError()


class SimpleMeshing(MeshingBase):

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
    @staticmethod
    def generate(voxel_size):
        for model in VoxModel.get_all():
            pos_sorted_voxels = {}
            mtl_sorted_voxels = {}
            for voxel in model.voxels:
                position_str = tuple(voxel[:3])
                mtl_id = voxel[3]
                pos_sorted_voxels[position_str] = VoxBaseMaterial.get(mtl_id)
                if mtl_id not in mtl_sorted_voxels:
                    mtl_sorted_voxels[mtl_id] = []
                mtl_sorted_voxels[mtl_id].append(voxel)
            for mtl_id, voxels in mtl_sorted_voxels.items():
                model.meshes[mtl_id] = {}
                working_is_glass = type(VoxBaseMaterial.get(mtl_id)) == VoxGlassMaterial
                for voxel in voxels:
                    front = (voxel[0], voxel[1] - 1, voxel[2])
                    if front not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[front]) == VoxGlassMaterial and not working_is_glass):
                        front_face = (voxel[0], voxel[1], voxel[2], VoxelSides.FRONT)
                        model.meshes[mtl_id][front_face] = True

                    back = (voxel[0], voxel[1] + 1, voxel[2])
                    if back not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[back]) == VoxGlassMaterial and not working_is_glass):
                        back_face = (voxel[0], voxel[1], voxel[2], VoxelSides.BACK)
                        model.meshes[mtl_id][back_face] = True

                    right = (voxel[0] + 1, voxel[1], voxel[2])
                    if right not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[right]) == VoxGlassMaterial and not working_is_glass):
                        right_face = (voxel[0], voxel[1], voxel[2], VoxelSides.RIGHT)
                        model.meshes[mtl_id][right_face] = True

                    left = (voxel[0] - 1, voxel[1], voxel[2])
                    if left not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[left]) == VoxGlassMaterial and not working_is_glass):
                        left_face = (voxel[0], voxel[1], voxel[2], VoxelSides.LEFT)
                        model.meshes[mtl_id][left_face] = True

                    top = (voxel[0], voxel[1], voxel[2] + 1)
                    if top not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[top]) == VoxGlassMaterial and not working_is_glass):
                        top_face = (voxel[0], voxel[1], voxel[2], VoxelSides.TOP)
                        model.meshes[mtl_id][top_face] = True

                    bottom = (voxel[0], voxel[1], voxel[2] - 1)
                    if bottom not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[bottom]) == VoxGlassMaterial and not working_is_glass):
                        bottom_face = (voxel[0], voxel[1], voxel[2], VoxelSides.BOTTOM)
                        model.meshes[mtl_id][bottom_face] = True

                greedy_faces = []
                run_start = None
                run_width = 0
                last_voxel = None

                for side in VoxelSides.values():
                    uvw_dimensions = GreedyMeshing.__get_uvw_dimensions(model, side)
                    for w in range(uvw_dimensions[2]+1):
                        for v in range(uvw_dimensions[1]+1):
                            for u in range(uvw_dimensions[0]+1):
                                if run_start is None and GreedyMeshing.__can_merge_face(model, mtl_id, u, v, w, side):
                                    run_start = (u,v)
                                elif run_start and not GreedyMeshing.__can_merge_face(model, mtl_id,u ,v ,w, side):
                                    for run_v in range(v + 1, uvw_dimensions[1]+1):
                                        if run_start is None:
                                            break
                                        row_complete = True
                                        for run_u in range(run_start[0], run_start[0] + run_width):
                                            if not GreedyMeshing.__can_merge_face(model, mtl_id, run_u, run_v, w, side):
                                                row_complete = False
                                                break
                                        if row_complete:
                                            last_voxel = (run_start[0] + run_width - 1, run_v)
                                        else:
                                            merged_face = GreedyMeshing.__create_merged_face(model, voxel_size,
                                                                                             run_start, last_voxel,
                                                                                             w, side)
                                            greedy_faces.extend(merged_face)
                                            # pop off any faces that we just merged so that they aren't counted when we
                                            # continue scanning.
                                            for pop_v in range(run_start[1], last_voxel[1] + 1):
                                                for pop_u in range(run_start[0], run_start[0] + run_width):
                                                    GreedyMeshing.__remove_voxel_face(model, mtl_id, pop_u,
                                                                                      pop_v, w, side)
                                            run_start = None
                                            run_width = 0
                                            last_voxel = None
                                            break

                                if run_start is not None:
                                    run_width += 1
                                    last_voxel = (u,v)

                model.meshes[mtl_id] = greedy_faces

    @staticmethod
    def __can_merge_face(model, mtl_id, u, v, w, side):
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
    def __create_merged_face(model, voxel_size, run_start, run_end, w, side):
        """ Create a face. Also centers the model in X and Y.

        Args:
            run_start:
            run_end:
            w:
            side:

        Returns:

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
    def __remove_voxel_face(model, mtl_id, u, v, w, side):
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
    def __get_uvw_dimensions(model, side):
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
