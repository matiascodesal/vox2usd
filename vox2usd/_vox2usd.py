
# TODO: Figure out MV instancing (Ref)
# TODO: add script args (input path, output path, etc)
# TODO: Should I rotate studs to always point up?
# TODO: Only create relevant prototypes for each instancer
# TODO: Get rid of omni info:id errors
# TODO: Compute bboxes


import os

from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf, Kind, Vt, Ar

from vox2usd.vox import (VoxReader, VoxModel, VoxNode, VoxTransform,
                         VoxGroup, VoxShape, VoxBaseMaterial, VoxGlassMaterial)
from constants import (GEOMETRY_SCOPE_NAME, LOOKS_SCOPE_NAME, DEFAULT_METERS_PER_VOXEL,
                       GeometryVariantSetNames, PointShapeVariantSetNames, ShaderVariantSetNames, VoxelSides, StudMesh)


class Vox2UsdConverter(object):
    def __init__(self, filepath, voxel_spacing=DEFAULT_METERS_PER_VOXEL, voxel_size=DEFAULT_METERS_PER_VOXEL,
                 gamma_correct=True, gamma_value=2.2, join_voxels=False, use_physics=False, flatten=False):
        self.vox_file_path = filepath
        self.voxel_spacing = voxel_spacing
        self.voxel_size = voxel_size
        self.gamma_correct = gamma_correct
        self.gamma_value = gamma_value
        self.join_voxels = join_voxels
        self.use_physics = use_physics
        self.flatten = flatten

    def calculate_simple_meshes(self):

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
                                      (voxel[0] + self.voxel_size - model_half_x, voxel[1] - model_half_y, voxel[2] - model_half_z),
                                      (voxel[0] + self.voxel_size - model_half_x, voxel[1] - model_half_y, voxel[2] + self.voxel_size - model_half_z),
                                      (voxel[0] - model_half_x, voxel[1] - model_half_y, voxel[2] + self.voxel_size - model_half_z)]

                        model.meshes[mtl_id].extend(front_face)

                    back = (voxel[0], voxel[1] + 1, voxel[2])
                    if back not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[back]) == VoxGlassMaterial and not working_is_glass):
                        back_face = [(voxel[0] + self.voxel_size - model_half_x, voxel[1] + self.voxel_size - model_half_y, voxel[2] - model_half_z),
                                     (voxel[0] - model_half_x, voxel[1] + self.voxel_size - model_half_y, voxel[2] - model_half_z),
                                     (voxel[0] - model_half_x, voxel[1] + self.voxel_size - model_half_y, voxel[2] + self.voxel_size - model_half_z),
                                     (voxel[0] + self.voxel_size - model_half_x, voxel[1] + self.voxel_size - model_half_y, voxel[2] + self.voxel_size - model_half_z)]
                        model.meshes[mtl_id].extend(back_face)

                    right = (voxel[0] + 1, voxel[1], voxel[2])
                    if right not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[right]) == VoxGlassMaterial and not working_is_glass):
                        right_face = [(voxel[0] + self.voxel_size - model_half_x, voxel[1] - model_half_y, voxel[2] - model_half_z),
                                      (voxel[0] + self.voxel_size - model_half_x, voxel[1] + self.voxel_size - model_half_y, voxel[2] - model_half_z),
                                      (voxel[0] + self.voxel_size - model_half_x, voxel[1] + self.voxel_size - model_half_y, voxel[2] + self.voxel_size - model_half_z),
                                      (voxel[0] + self.voxel_size - model_half_x, voxel[1] - model_half_y, voxel[2] + self.voxel_size - model_half_z)]
                        model.meshes[mtl_id].extend(right_face)

                    left = (voxel[0] - 1, voxel[1], voxel[2])
                    if left not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[left]) == VoxGlassMaterial and not working_is_glass):
                        left_face = [(voxel[0] - model_half_x, voxel[1] + self.voxel_size - model_half_y, voxel[2] - model_half_z),
                                     (voxel[0] - model_half_x, voxel[1] - model_half_y, voxel[2] - model_half_z),
                                     (voxel[0] - model_half_x, voxel[1] - model_half_y, voxel[2] + self.voxel_size - model_half_z),
                                     (voxel[0] - model_half_x, voxel[1] + self.voxel_size - model_half_y, voxel[2] + self.voxel_size - model_half_z)]
                        model.meshes[mtl_id].extend(left_face)

                    top = (voxel[0], voxel[1], voxel[2] + 1)
                    if top not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[top]) == VoxGlassMaterial and not working_is_glass):
                        top_face = [(voxel[0] - model_half_x, voxel[1] - model_half_y, voxel[2] + self.voxel_size - model_half_z),
                                    (voxel[0] + self.voxel_size - model_half_x, voxel[1] - model_half_y, voxel[2] + self.voxel_size - model_half_z),
                                    (voxel[0] + self.voxel_size - model_half_x, voxel[1] + self.voxel_size - model_half_y, voxel[2] + self.voxel_size - model_half_z),
                                    (voxel[0] - model_half_x, voxel[1] + self.voxel_size - model_half_y, voxel[2] + self.voxel_size - model_half_z)]
                        model.meshes[mtl_id].extend(top_face)

                    bottom = (voxel[0], voxel[1], voxel[2] - 1)
                    if bottom not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[bottom]) == VoxGlassMaterial and not working_is_glass):
                        bottom_face = [
                            (voxel[0] + self.voxel_size - model_half_x, voxel[1] - model_half_y, voxel[2] - model_half_z),
                            (voxel[0] - model_half_x, voxel[1] - model_half_y, voxel[2] - model_half_z),
                            (voxel[0] - model_half_x, voxel[1] + self.voxel_size - model_half_y, voxel[2] - model_half_z),
                            (voxel[0] + self.voxel_size - model_half_x, voxel[1] + self.voxel_size - model_half_y, voxel[2] - model_half_z)]
                        model.meshes[mtl_id].extend(bottom_face)

    def calculate_greedy_meshes(self):

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

                def can_merge_face(u, v, w, side):
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

                def create_merged_face(run_start, run_end, w, side):
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
                            (run_end[0] + self.voxel_size, w, run_start[1]),
                            (run_end[0] + self.voxel_size, w, run_end[1] + self.voxel_size),
                            (run_start[0], w, run_end[1] + self.voxel_size)
                        ]
                    elif side == VoxelSides.BACK:
                        merged_face = [
                            (run_end[0] + self.voxel_size, w + self.voxel_size, run_start[1]),
                            (run_start[0], w + self.voxel_size, run_start[1]),
                            (run_start[0], w + self.voxel_size, run_end[1] + self.voxel_size),
                            (run_end[0] + self.voxel_size, w + self.voxel_size, run_end[1] + self.voxel_size)
                        ]
                    elif side == VoxelSides.RIGHT:
                        merged_face = [
                            (w + self.voxel_size, run_start[0], run_start[1]),
                            (w + self.voxel_size, run_end[0] + self.voxel_size, run_start[1]),
                            (w + self.voxel_size, run_end[0] + self.voxel_size, run_end[1] + self.voxel_size),
                            (w + self.voxel_size, run_start[0], run_end[1] + self.voxel_size)
                        ]
                    elif side == VoxelSides.LEFT:
                        merged_face = [
                            (w, run_end[0] + self.voxel_size, run_start[1]),
                            (w, run_start[0], run_start[1]),
                            (w, run_start[0], run_end[1] + self.voxel_size),
                            (w, run_end[0] + self.voxel_size, run_end[1] + self.voxel_size)
                        ]
                    elif side == VoxelSides.TOP:
                        # TOP winding order
                        # 1. Bottom Left
                        # 2. Bottom Right
                        # 3. Top Right
                        # 4. Top Left
                        merged_face = [
                            (run_start[0], run_start[1], w + self.voxel_size),
                            (run_end[0] + self.voxel_size, run_start[1], w + self.voxel_size),
                            (run_end[0] + self.voxel_size, run_end[1] + self.voxel_size, w + self.voxel_size),
                            (run_start[0], run_end[1] + self.voxel_size, w + self.voxel_size)
                        ]
                    elif side == VoxelSides.BOTTOM:
                        merged_face = [
                            (run_end[0] + self.voxel_size, run_start[1], w),
                            (run_start[0], run_start[1], w),
                            (run_start[0], run_end[1] + self.voxel_size, w),
                            (run_end[0] + self.voxel_size, run_end[1] + self.voxel_size, w)
                        ]

                    for vert_id in range(len(merged_face)):
                        merged_face[vert_id] = (
                            merged_face[vert_id][0] - model_half_x,
                            merged_face[vert_id][1] - model_half_y,
                            merged_face[vert_id][2] - model_half_z
                        )

                    return merged_face

                def remove_voxel_face(u, v, w, side):
                    if side == VoxelSides.FRONT:
                        model.meshes[mtl_id].pop((pop_u, w, pop_v, side))
                    elif side == VoxelSides.BACK:
                        model.meshes[mtl_id].pop((pop_u, w, pop_v, side))
                    elif side == VoxelSides.RIGHT:
                        model.meshes[mtl_id].pop((w, pop_u, pop_v, side))
                    elif side == VoxelSides.LEFT:
                        model.meshes[mtl_id].pop((w, pop_u, pop_v, side))
                    elif side == VoxelSides.TOP:
                        model.meshes[mtl_id].pop((pop_u, pop_v, w, side))
                    elif side == VoxelSides.BOTTOM:
                        model.meshes[mtl_id].pop((pop_u, pop_v, w, side))

                def get_uvw_dimensions(side):
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

                for side in VoxelSides.values():
                    uvw_dimensions = get_uvw_dimensions(side)
                    for w in range(uvw_dimensions[2]+1):
                        for v in range(uvw_dimensions[1]+1):
                            for u in range(uvw_dimensions[0]+1):
                                if run_start is None and can_merge_face(u,v,w,side):
                                    run_start = (u,v)
                                elif run_start and not can_merge_face(u,v,w,side):
                                    for run_v in range(v + 1, uvw_dimensions[1]+1):
                                        if run_start is None:
                                            break
                                        row_complete = True
                                        for run_u in range(run_start[0], run_start[0] + run_width):
                                            if not can_merge_face(run_u,run_v,w,side):
                                                row_complete = False
                                                break
                                        if row_complete:
                                            last_voxel = (run_start[0] + run_width - 1, run_v)
                                        else:
                                            merged_face = create_merged_face(run_start, last_voxel, w, side)
                                            greedy_faces.extend(merged_face)
                                            # pop off any faces that we just merged so that they aren't counted when we
                                            # continue scanning.
                                            for pop_v in range(run_start[1], last_voxel[1] + 1):
                                                for pop_u in range(run_start[0], run_start[0] + run_width):
                                                    remove_voxel_face(pop_u, pop_v, w, side)
                                            run_start = None
                                            run_width = 0
                                            last_voxel = None
                                            break

                                if run_start is not None:
                                    run_width += 1
                                    last_voxel = (u,v)

                model.meshes[mtl_id] = greedy_faces

    def create_material(self, stage, looks_scope, name, vox_mtl):
        mtl = UsdShade.Material.Define(stage, looks_scope.GetPath().AppendPath(name))
        self.shader_varset.SetVariantSelection(ShaderVariantSetNames.PREVIEW)
        with self.shader_varset.GetVariantEditContext():
            prvw_shader = UsdShade.Shader.Define(stage, mtl.GetPath().AppendPath("prvw_shader"))
            prvw_shader = vox_mtl.populate_usd_preview_surface(prvw_shader)
            mtl.CreateSurfaceOutput().ConnectToSource(prvw_shader.ConnectableAPI(), "surface")
        self.shader_varset.SetVariantSelection(ShaderVariantSetNames.OMNIVERSE)
        with self.shader_varset.GetVariantEditContext():
            omni_shader = UsdShade.Shader.Define(stage, mtl.GetPath().AppendPath("omni_shader"))
            omni_shader = vox_mtl.populate_omni_shader(omni_shader)
            mtl.CreateSurfaceOutput().ConnectToSource(omni_shader.ConnectableAPI(), "surface")

        return mtl

    def convert(self):
        print("\nImporting voxel file {}\n".format(self.vox_file_path))

        import time
        time_start = time.time()
        VoxBaseMaterial.initialize(self.gamma_correct, self.gamma_value)
        VoxReader(self.vox_file_path).read()

        self.asset_name = os.path.splitext(os.path.basename(self.vox_file_path))[0]
        self.stage = Usd.Stage.CreateNew(r"C:\temp\test_data\{}.usda".format(self.asset_name))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        asset_geom = UsdGeom.Xform.Define(self.stage, "/" + self.asset_name)
        self.stage.SetDefaultPrim(asset_geom.GetPrim())
        Usd.ModelAPI(asset_geom).SetKind(Kind.Tokens.component)
        self.geometry_scope = UsdGeom.Scope.Define(self.stage, asset_geom.GetPath().AppendPath(GEOMETRY_SCOPE_NAME))
        self.looks_scope = UsdGeom.Scope.Define(self.stage, asset_geom.GetPath().AppendPath(LOOKS_SCOPE_NAME))

        self.geo_varset = asset_geom.GetPrim().GetVariantSets().AddVariantSet("Geometry")
        for variant_name in GeometryVariantSetNames.values():
            self.geo_varset.AddVariant(variant_name)

        self.geo_varset.SetVariantSelection(GeometryVariantSetNames.POINT_INSTANCES)
        with self.geo_varset.GetVariantEditContext():
            self.point_shape_varset = asset_geom.GetPrim().GetVariantSets().AddVariantSet("PointShape")
            for variant_name in PointShapeVariantSetNames.values():
                self.point_shape_varset.AddVariant(variant_name)
        self.shader_varset = asset_geom.GetPrim().GetVariantSets().AddVariantSet("Shader")
        for variant_name in ShaderVariantSetNames.values():
            self.shader_varset.AddVariant(variant_name)


        #self.calculate_simple_meshes()
        self.calculate_greedy_meshes()

        self.used_mtls = {}
        for index in VoxBaseMaterial.used_palette_ids:
            mtl = VoxBaseMaterial.get(index)
            mtl = self.create_material(self.stage, self.looks_scope, "VoxelMtl_{}".format(mtl.get_display_id()), mtl)
            self.used_mtls[index] = {"mtl": mtl,
                                     GeometryVariantSetNames.MERGED_MESHES: [],
                                     GeometryVariantSetNames.POINT_INSTANCES: {
                                         PointShapeVariantSetNames.CUBES: [],
                                         PointShapeVariantSetNames.SPHERES: [],
                                         PointShapeVariantSetNames.STUDS: []
                                     }
                                     }

        print("Start converting")
        self.total_voxels = 0
        self.total_triangles = 0
        print(VoxNode.instances)
        print(VoxNode.get_top_nodes())

        mesh_stage = Usd.Stage.CreateNew(r"C:\temp\test_data\{}.mesh.usdc".format(self.asset_name))
        points_stage = Usd.Stage.CreateNew(r"C:\temp\test_data\{}.points.usdc".format(self.asset_name))

        root_xform_node = VoxTransform.get_top_nodes()[0]
        root_group_node = root_xform_node.get_child()

        self.__convert_node(mesh_stage, root_xform_node, mesh_stage.GetPseudoRoot(), GeometryVariantSetNames.MERGED_MESHES)
        self.__convert_node(points_stage, root_xform_node, points_stage.GetPseudoRoot(), GeometryVariantSetNames.POINT_INSTANCES)

        default_prim = mesh_stage.GetPrimAtPath(mesh_stage.GetPseudoRoot().GetPath().AppendPath("VoxelGroup_{}".format(root_group_node.node_id)))
        mesh_stage.SetDefaultPrim(default_prim)
        default_prim = points_stage.GetPrimAtPath(points_stage.GetPseudoRoot().GetPath().AppendPath("VoxelGroup_{}".format(root_group_node.node_id)))
        points_stage.SetDefaultPrim(default_prim)
        mesh_stage.Save()
        points_stage.Save()

        root_xform_prim = self.stage.OverridePrim(self.geometry_scope.GetPath().AppendPath("VoxelRoot"))

        self.geo_varset.SetVariantSelection(GeometryVariantSetNames.MERGED_MESHES)
        with self.geo_varset.GetVariantEditContext():
            root_xform_prim.GetPayloads().AddPayload(Sdf.Payload("./{}.mesh.usdc".format(self.asset_name)))
        self.geo_varset.SetVariantSelection(GeometryVariantSetNames.POINT_INSTANCES)
        with self.geo_varset.GetVariantEditContext():
            root_xform_prim.GetPayloads().AddPayload(Sdf.Payload("./{}.points.usdc".format(self.asset_name)))

        for mtl_id, item in self.used_mtls.items():
            mtl = item["mtl"]
            merged_mesh_targets = item[GeometryVariantSetNames.MERGED_MESHES]
            proto_targets = item[GeometryVariantSetNames.POINT_INSTANCES]
            voxel_root_path = self.geometry_scope.GetPath().AppendChild("VoxelRoot")
            voxel_root = self.stage.GetPrimAtPath(voxel_root_path)
            self.geo_varset.SetVariantSelection(GeometryVariantSetNames.MERGED_MESHES)
            with self.geo_varset.GetVariantEditContext():
                for target_path in merged_mesh_targets:
                    target_path = target_path.pathString.split("/")
                    target_path = "/".join(target_path[2:])
                    target_path = voxel_root_path.AppendPath(target_path)
                    target_prim = self.stage.GetPrimAtPath(target_path)
                    UsdShade.MaterialBindingAPI(target_prim).Bind(mtl)
            self.geo_varset.SetVariantSelection(GeometryVariantSetNames.POINT_INSTANCES)
            with self.geo_varset.GetVariantEditContext():
                self.point_shape_varset.SetVariantSelection(PointShapeVariantSetNames.CUBES)
                with self.point_shape_varset.GetVariantEditContext():
                    voxel_root.GetVariantSet("PointShape").SetVariantSelection(PointShapeVariantSetNames.CUBES)
                    for target_path in proto_targets[PointShapeVariantSetNames.CUBES]:
                        target_path = target_path.pathString.split("/")
                        target_path = "/".join(target_path[2:])
                        target_path = voxel_root_path.AppendPath(target_path)
                        target_prim = self.stage.GetPrimAtPath(target_path)
                        UsdShade.MaterialBindingAPI(target_prim).Bind(mtl)
                self.point_shape_varset.SetVariantSelection(PointShapeVariantSetNames.SPHERES)
                with self.point_shape_varset.GetVariantEditContext():
                    voxel_root.GetVariantSet("PointShape").SetVariantSelection(PointShapeVariantSetNames.SPHERES)
                    for target_path in proto_targets[PointShapeVariantSetNames.SPHERES]:
                        target_path = target_path.pathString.split("/")
                        target_path = "/".join(target_path[2:])
                        target_path = voxel_root_path.AppendPath(target_path)
                        target_prim = self.stage.GetPrimAtPath(target_path)
                        UsdShade.MaterialBindingAPI(target_prim).Bind(mtl)
                self.point_shape_varset.SetVariantSelection(PointShapeVariantSetNames.STUDS)
                with self.point_shape_varset.GetVariantEditContext():
                    voxel_root.GetVariantSet("PointShape").SetVariantSelection(PointShapeVariantSetNames.STUDS)
                    for target_path in proto_targets[PointShapeVariantSetNames.STUDS]:
                        target_path = target_path.pathString.split("/")
                        target_path = "/".join(target_path[2:])
                        target_path = voxel_root_path.AppendPath(target_path)
                        target_prim = self.stage.GetPrimAtPath(target_path)
                        UsdShade.MaterialBindingAPI(target_prim).Bind(mtl)

        if self.flatten:
            self.geo_varset.SetVariantSelection(GeometryVariantSetNames.MERGED_MESHES)
            self.shader_varset.SetVariantSelection(ShaderVariantSetNames.PREVIEW)
            self.stage.Export(r"C:\temp\test_data\{}.usdc".format(self.asset_name))
        else:
            self.geo_varset.SetVariantSelection(GeometryVariantSetNames.MERGED_MESHES)
            self.shader_varset.SetVariantSelection(ShaderVariantSetNames.OMNIVERSE)
            self.point_shape_varset.SetVariantSelection(PointShapeVariantSetNames.CUBES)
            self.stage.Save()

        print("Converted {} total voxels".format(self.total_voxels))
        print("Converted {} total triangles".format(self.total_triangles))
        print("\nSuccessfully converted {} in {:.3f} sec".format(self.vox_file_path, time.time() - time_start))
        return {'FINISHED'}

    def __convert_node(self, stage, node, parent_prim, geom_type):
        print("node_id", node.node_id)
        if isinstance(node, VoxTransform):
            xform_child = node.get_child()
            if isinstance(xform_child, VoxGroup):
                xform = UsdGeom.Xform.Define(stage, parent_prim.GetPath().AppendPath("VoxelGroup_{}".format(xform_child.node_id)))
                xform.AddTransformOp().Set(Gf.Matrix4d(*node.transform))
                for child in xform_child.children:
                    self.__convert_node(stage, child, xform, geom_type)
            elif isinstance(xform_child, VoxShape):
                if geom_type == GeometryVariantSetNames.MERGED_MESHES:
                    self.__voxels2meshes(stage, node, xform_child, parent_prim)
                elif geom_type == GeometryVariantSetNames.POINT_INSTANCES:
                    self.__voxels2point_instances(stage, node, xform_child, parent_prim)

        else:
            raise RuntimeError("Expected VoxTransform node. Got {}.".format(node.__class__))

    def __set_pivot(self, vox_model, xformable):
        # TODO: This doesn't actually work for rotated models. The pivot doesn't end up at the bottom
        # MV pivot is at the center. I prefer to have it at the bottom.
        # I'm tweaking the translation here to counteract that.
        xform_attr = xformable.GetPrim().GetAttribute("xformOp:transform")
        curr_xform = xform_attr.Get()
        # Need to figure out the local
        up_vector = curr_xform * Gf.Vec4d(0,0,1,1)
        up_vector = Gf.Vec3d(up_vector[0:3]).GetNormalized()
        # copy bottom row of matrix since index operator is read only on Matrix4d
        trans_row = curr_xform[3]
        for index in range(3):
            if up_vector[index] != 0:
                trans_row[index] = trans_row[index] - int(vox_model.size[2] / 2.0)
                break

        curr_xform.SetRow(3, Gf.Vec4d(*trans_row))
        xform_attr.Set(curr_xform)

    def __fill_mesh_and_bind_mtl(self, shape_node, mesh):
        display_colors = []
        opacity = []
        mtl_id, mesh_verts = shape_node.model.meshes.items()[0]
        vox_mtl = VoxBaseMaterial.get(mtl_id)
        mesh.CreatePointsAttr(mesh_verts)
        face_count = int(len(mesh_verts) / 4.0)
        self.total_triangles += face_count * 2
        mesh.CreateFaceVertexCountsAttr([4]*face_count)
        mesh.CreateFaceVertexIndicesAttr(list(range(len(mesh_verts))))
        display_colors.extend([Gf.Vec3f(vox_mtl.color[0:3])] * face_count)
        if isinstance(vox_mtl, VoxGlassMaterial):
            opacity.extend([vox_mtl.get_opacity()] * face_count)
        else:
            opacity.extend([1.0] * face_count)
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(display_colors)
        mesh.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.uniform).Set(opacity)
        self.used_mtls[mtl_id][GeometryVariantSetNames.MERGED_MESHES].append(mesh.GetPath())

    def __create_geom_subset_per_mtl(self, shape_node, mesh):
            # merge meshes to create geomsubsets
            vertices = []
            total_face_count = 0
            display_colors = []
            opacity = []
            subsets = []
            for mtl_id, mesh_verts in shape_node.model.meshes.items():
                start_idx = len(vertices)
                vertices.extend(mesh_verts)
                end_idx = len(vertices)
                start_face_idx = int(start_idx / 4)
                end_face_idx = int(end_idx / 4)
                face_count = int(len(mesh_verts) / 4.0)
                total_face_count += face_count
                vox_mtl = VoxBaseMaterial.get(mtl_id)
                display_colors.extend([Gf.Vec3f(vox_mtl.color[0:3])] * face_count)
                if isinstance(vox_mtl, VoxGlassMaterial):
                    opacity.extend([vox_mtl.get_opacity()] * face_count)
                else:
                    opacity.extend([1.0] * face_count)
                subsets.append({"mtl_id": mtl_id, "start_face_idx": start_face_idx, "end_face_idx": end_face_idx})

            mesh.CreatePointsAttr(vertices)
            mesh.CreateFaceVertexCountsAttr([4] * total_face_count)
            mesh.CreateFaceVertexIndicesAttr(list(range(len(vertices))))
            mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(display_colors)
            mesh.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.uniform).Set(opacity)

            with Usd.EditContext(self.stage):
                mesh_binding_api = UsdShade.MaterialBindingAPI(mesh.GetPrim())
                for item in subsets:
                    subset = mesh_binding_api.CreateMaterialBindSubset("VoxelPart_{}".format(item["mtl_id"]),
                                                                       Vt.IntArray(list(range(item["start_face_idx"], item["end_face_idx"]))))
                    self.used_mtls[item["mtl_id"]][GeometryVariantSetNames.MERGED_MESHES].append(subset.GetPath())

            self.total_triangles += total_face_count * 2

    def __voxels2meshes(self, stage, xform_node, shape_node, parent_prim):
        self.total_voxels += len(shape_node.model.voxels)
        mesh = UsdGeom.Mesh.Define(stage, parent_prim.GetPath().AppendPath("VoxelModel_{}".format(shape_node.node_id)))
        mesh.AddTransformOp().Set(Gf.Matrix4d(*xform_node.transform))
        # self.__set_pivot(shape_node.model, mesh)
        if len(shape_node.model.meshes.keys()) > 1:
            self.__create_geom_subset_per_mtl(shape_node, mesh)
        else:
            self.__fill_mesh_and_bind_mtl(shape_node, mesh)

    def __create_cube_geom(self, stage, instancer, proto_container, mtl_id, mtl):
        mtl_display_id = VoxBaseMaterial.get(mtl_id).get_display_id()
        cube = UsdGeom.Cube.Define(stage,
                                   proto_container.GetPath().AppendPath("VoxelCube_{}".format(mtl_display_id)))
        cube.CreateSizeAttr(self.voxel_size)
        self.used_mtls[mtl_id][GeometryVariantSetNames.POINT_INSTANCES][PointShapeVariantSetNames.CUBES].append(
            cube.GetPath())
        self.__set_common_point_attrs(cube, instancer, mtl_id)

    def __create_sphere_geom(self, stage, instancer, proto_container, mtl_id, mtl):
        mtl_display_id = VoxBaseMaterial.get(mtl_id).get_display_id()
        sphere = UsdGeom.Sphere.Define(stage,
                                   proto_container.GetPath().AppendPath("VoxelSphere_{}".format(mtl_display_id)))
        sphere.CreateRadiusAttr(self.voxel_size / 2.0)
        self.used_mtls[mtl_id][GeometryVariantSetNames.POINT_INSTANCES][PointShapeVariantSetNames.SPHERES].append(
            sphere.GetPath())
        self.__set_common_point_attrs(sphere, instancer, mtl_id)

    def __create_stud_geom(self, stage, instancer, proto_container, mtl_id, mtl):
        mtl_display_id = VoxBaseMaterial.get(mtl_id).get_display_id()
        mesh = UsdGeom.Mesh.Define(stage, proto_container.GetPath().AppendPath("VoxelStud_{}".format(mtl_display_id)))
        mesh.CreatePointsAttr(StudMesh.points)
        mesh.CreateFaceVertexCountsAttr(StudMesh.face_vertex_counts)
        mesh.CreateFaceVertexIndicesAttr(StudMesh.face_vertex_indices)
        mesh.CreateNormalsAttr(StudMesh.normals)
        mesh.SetNormalsInterpolation(StudMesh.normals_interpolation)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        self.used_mtls[mtl_id][GeometryVariantSetNames.POINT_INSTANCES][PointShapeVariantSetNames.STUDS].append(mesh.GetPath())
        self.__set_common_point_attrs(mesh, instancer, mtl_id)

    def __set_common_point_attrs(self, geom, instancer, mtl_id):
        # Move voxel pivot to the front-bottom-left like MV voxels
        geom.AddTranslateOp().Set((0.5, 0.5, 0.5))

        vox_mtl = VoxBaseMaterial.get(mtl_id)
        geom.CreateDisplayColorAttr([vox_mtl.color[0:3]])
        if isinstance(vox_mtl, VoxGlassMaterial):
            geom.CreateDisplayOpacityAttr([vox_mtl.get_opacity()])
        if self.use_physics:
            physics_apis = Sdf.TokenListOp.Create(["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"])
            shape_prim = geom.GetPrim()
            shape_prim.SetMetadata("apiSchemas", physics_apis)
            shape_prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(1)
            shape_prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(0)
            shape_prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(1)
            shape_prim.CreateAttribute("physics:startsAsleep", Sdf.ValueTypeNames.Bool).Set(1)

        instancer.GetPrototypesRel().AddTarget(geom.GetPath())

    def __voxels2point_instances(self, stage, xform_node, shape_node, parent_prim):
        instancer = UsdGeom.PointInstancer.Define(stage, parent_prim.GetPath().AppendPath(
            "VoxModel_{}".format(shape_node.node_id)))
        instancer.AddTransformOp().Set(Gf.Matrix4d(*xform_node.transform))
        # self.__set_pivot(shape_node.model, instancer)
        instancer.CreatePrototypesRel()
        proto_container = stage.OverridePrim(instancer.GetPath().AppendPath("Prototypes"))
        mtl2proto_id = {}
        default_prim = stage.GetPseudoRoot().GetChildren()[0]
        if not default_prim.HasVariantSets():
            pt_shape_varset = default_prim.GetVariantSets().AddVariantSet("PointShape")
            for variant_name in PointShapeVariantSetNames.values():
                pt_shape_varset.AddVariant(variant_name)
        else:
            pt_shape_varset = default_prim.GetVariantSet("PointShape")

        for proto_id, item in enumerate(self.used_mtls.items()):
            mtl_id, mtl = item
            mtl = mtl["mtl"]
            mtl2proto_id[mtl_id] = proto_id

            pt_shape_varset.SetVariantSelection(PointShapeVariantSetNames.CUBES)
            with pt_shape_varset.GetVariantEditContext():
                self.__create_cube_geom(stage, instancer, proto_container, mtl_id, mtl)
            pt_shape_varset.SetVariantSelection(PointShapeVariantSetNames.SPHERES)
            with pt_shape_varset.GetVariantEditContext():
                self.__create_sphere_geom(stage, instancer, proto_container, mtl_id, mtl)
            pt_shape_varset.SetVariantSelection(PointShapeVariantSetNames.STUDS)
            with pt_shape_varset.GetVariantEditContext():
                self.__create_stud_geom(stage, instancer, proto_container, mtl_id, mtl)



        # HACK: Need to reset the specifier because it gets set to "def" after adding children
        # proto_container.GetPrim().SetSpecifier(Sdf.SpecifierOver)

        # TODO: Can I make prototypes a relative path?
        ids = []
        positions = []
        for idx, voxel in enumerate(shape_node.model.voxels):
            #self.total_voxels += 1
            ids.append(mtl2proto_id[voxel[3]])
            position = [float(coord) * self.voxel_spacing for coord in voxel[:3]]
            # XY center the local origin on the model
            position = [position[0] - int(shape_node.model.size[0] / 2.0),
                        position[1] - int(shape_node.model.size[1] / 2.0),
                        position[2] - int(shape_node.model.size[2] / 2.0)]
            positions.append(position)
        instancer.CreateProtoIndicesAttr()
        instancer.CreatePositionsAttr()
        instancer.GetProtoIndicesAttr().Set(ids)
        instancer.GetPositionsAttr().Set(positions)


if __name__ == '__main__':
    Vox2UsdConverter(r"C:\temp\test_data\cop_car.vox", use_physics=False, flatten=True).convert()
