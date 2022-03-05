import os
import struct

from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf, Kind

from vox2usd.vox import VoxReader, VoxModel, VoxNode, VoxBaseMaterial, VoxGlassMaterial

GEOMETRY_SCOPE_NAME = "Geometry"
LOOKS_SCOPE_NAME = "Looks"
DEFAULT_VOXELS_PER_METER = 1
DEFAULT_METERS_PER_VOXEL = 1.0 / DEFAULT_VOXELS_PER_METER


class TransformStack(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.stack = []

    def push(self, x, y, z):
        self.x += x
        self.y += y
        self.z += z
        self.stack.append((x, y, z))

    def pop(self):
        x, y, z = self.stack.pop()
        self.x -= x
        self.y -= y
        self.z -= z


# 1 m = 32 voxels
class Vox2UsdConverter(object):
    def __init__(self, filepath, voxel_spacing=DEFAULT_METERS_PER_VOXEL, voxel_size=DEFAULT_METERS_PER_VOXEL,
                use_palette=True, gamma_correct=True, gamma_value=2.2,
                join_voxels=False, use_physics=False, use_point_instancing=True, use_omni_mtls=False):
        self.vox_file_path = filepath
        self.voxel_spacing = voxel_spacing
        self.voxel_size = voxel_size
        self.use_palette = use_palette
        self.gamma_correct = gamma_correct
        self.gamma_value = gamma_value
        self.join_voxels = join_voxels
        self.use_physics = use_physics
        self.use_point_instancing = use_point_instancing
        self.use_omni_mtls = use_omni_mtls

    def calculate_simple_meshes(self, used_palette_indices):

        for model in VoxModel.get_all():
            pos_sorted_voxels = {}
            mtl_sorted_voxels = {}
            for voxel in model.voxels:
                position_str = "{},{},{}".format(*voxel[:3])
                mtl_id = voxel[3]
                pos_sorted_voxels[position_str] = VoxBaseMaterial.get(mtl_id)
                if mtl_id not in mtl_sorted_voxels:
                    mtl_sorted_voxels[mtl_id] = []
                mtl_sorted_voxels[mtl_id].append(voxel)
                # This is done here, so to avoid adding materials for voxels not in bounds
                used_palette_indices.add(mtl_id)  # record the palette entry is used
            for mtl_id, voxels in mtl_sorted_voxels.items():
                model.meshes[mtl_id] = []
                # TODO: IDK why getting the floor of this works.  Otherwise, I get cracks between models
                model_half_x = int(model.size[0] / 2.0)
                model_half_y = int(model.size[1] / 2.0)
                half = self.voxel_size / 2.0
                working_is_glass = type(VoxBaseMaterial.get(mtl_id)) == VoxGlassMaterial
                for voxel in voxels:
                    # -Y = Front
                    # +Y = Back
                    # -X = Right
                    # +X = Left
                    # +Z = Top
                    # -Z = Bottom
                    neighbors = []
                    front = "{},{},{}".format(voxel[0], voxel[1] - 1, voxel[2])
                    if front not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[front]) == VoxGlassMaterial and not working_is_glass):
                        front_face = [(voxel[0] - half - model_half_x, voxel[1] - half - model_half_y, voxel[2] - half),
                                      (voxel[0] + half - model_half_x, voxel[1] - half - model_half_y, voxel[2] - half),
                                      (voxel[0] + half - model_half_x, voxel[1] - half - model_half_y, voxel[2] + half),
                                      (voxel[0] - half - model_half_x, voxel[1] - half - model_half_y, voxel[2] + half)]

                        model.meshes[mtl_id].extend(front_face)

                    back = "{},{},{}".format(voxel[0], voxel[1] + 1, voxel[2])
                    if back not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[back]) == VoxGlassMaterial and not working_is_glass):
                        back_face = [(voxel[0] + half - model_half_x, voxel[1] + half - model_half_y, voxel[2] - half),
                                     (voxel[0] - half - model_half_x, voxel[1] + half - model_half_y, voxel[2] - half),
                                     (voxel[0] - half - model_half_x, voxel[1] + half - model_half_y, voxel[2] + half),
                                     (voxel[0] + half - model_half_x, voxel[1] + half - model_half_y, voxel[2] + half)]
                        model.meshes[mtl_id].extend(back_face)

                    right = "{},{},{}".format(voxel[0] + 1, voxel[1], voxel[2])
                    if right not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[right]) == VoxGlassMaterial and not working_is_glass):
                        right_face = [(voxel[0] + half - model_half_x, voxel[1] - half - model_half_y, voxel[2] - half),
                                      (voxel[0] + half - model_half_x, voxel[1] + half - model_half_y, voxel[2] - half),
                                      (voxel[0] + half - model_half_x, voxel[1] + half - model_half_y, voxel[2] + half),
                                      (voxel[0] + half - model_half_x, voxel[1] - half - model_half_y, voxel[2] + half)]
                        model.meshes[mtl_id].extend(right_face)

                    left = "{},{},{}".format(voxel[0] - 1, voxel[1], voxel[2])
                    if left not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[left]) == VoxGlassMaterial and not working_is_glass):
                        left_face = [(voxel[0] - half - model_half_x, voxel[1] + half - model_half_y, voxel[2] - half),
                                     (voxel[0] - half - model_half_x, voxel[1] - half - model_half_y, voxel[2] - half),
                                     (voxel[0] - half - model_half_x, voxel[1] - half - model_half_y, voxel[2] + half),
                                     (voxel[0] - half - model_half_x, voxel[1] + half - model_half_y, voxel[2] + half)]
                        model.meshes[mtl_id].extend(left_face)

                    top = "{},{},{}".format(voxel[0], voxel[1], voxel[2] + 1)
                    if top not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[top]) == VoxGlassMaterial and not working_is_glass):
                        top_face = [(voxel[0] - half - model_half_x, voxel[1] - half - model_half_y, voxel[2] + half),
                                    (voxel[0] + half - model_half_x, voxel[1] - half - model_half_y, voxel[2] + half),
                                    (voxel[0] + half - model_half_x, voxel[1] + half - model_half_y, voxel[2] + half),
                                    (voxel[0] - half - model_half_x, voxel[1] + half - model_half_y, voxel[2] + half)]
                        model.meshes[mtl_id].extend(top_face)

                    bottom = "{},{},{}".format(voxel[0], voxel[1], voxel[2] - 1)
                    if bottom not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[bottom]) == VoxGlassMaterial and not working_is_glass):
                        bottom_face = [
                            (voxel[0] + half - model_half_x, voxel[1] - half - model_half_y, voxel[2] - half),
                            (voxel[0] - half - model_half_x, voxel[1] - half - model_half_y, voxel[2] - half),
                            (voxel[0] - half - model_half_x, voxel[1] + half - model_half_y, voxel[2] - half),
                            (voxel[0] + half - model_half_x, voxel[1] + half - model_half_y, voxel[2] - half)]
                        model.meshes[mtl_id].extend(bottom_face)

    def calculate_greedy_meshes(self, used_palette_indices):

        for model in VoxModel.get_all():
            pos_sorted_voxels = {}
            mtl_sorted_voxels = {}
            for voxel in model.voxels:
                position_str = "{},{},{}".format(*voxel[:3])
                mtl_id = voxel[3]
                pos_sorted_voxels[position_str] = VoxBaseMaterial.get(mtl_id)
                if mtl_id not in mtl_sorted_voxels:
                    mtl_sorted_voxels[mtl_id] = []
                mtl_sorted_voxels[mtl_id].append(voxel)
                # This is done here, so to avoid adding materials for voxels not in bounds
                used_palette_indices.add(mtl_id)  # record the palette entry is used
            for mtl_id, voxels in mtl_sorted_voxels.items():
                model.meshes[mtl_id] = []
                # TODO: IDK why getting the floor of this works.  Otherwise, I get cracks between models
                model_half_x = int(model.size[0] / 2.0)
                model_half_y = int(model.size[1] / 2.0)
                half = self.voxel_size / 2.0
                working_is_glass = type(VoxBaseMaterial.get(mtl_id)) == VoxGlassMaterial
                for voxel in voxels:
                    # -Y = Front
                    # +Y = Back
                    # -X = Right
                    # +X = Left
                    # +Z = Top
                    # -Z = Bottom
                    neighbors = []
                    front = "{},{},{}".format(voxel[0], voxel[1] - 1, voxel[2])
                    if front not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[front]) == VoxGlassMaterial and not working_is_glass):
                        front_face = [(voxel[0] - half - model_half_x, voxel[1] - half - model_half_y, voxel[2] - half),
                                      (voxel[0] + half - model_half_x, voxel[1] - half - model_half_y, voxel[2] - half),
                                      (voxel[0] + half - model_half_x, voxel[1] - half - model_half_y, voxel[2] + half),
                                      (voxel[0] - half - model_half_x, voxel[1] - half - model_half_y, voxel[2] + half)]

                        model.meshes[mtl_id].extend(front_face)

                    back = "{},{},{}".format(voxel[0], voxel[1] + 1, voxel[2])
                    if back not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[back]) == VoxGlassMaterial and not working_is_glass):
                        back_face = [(voxel[0] + half - model_half_x, voxel[1] + half - model_half_y, voxel[2] - half),
                                     (voxel[0] - half - model_half_x, voxel[1] + half - model_half_y, voxel[2] - half),
                                     (voxel[0] - half - model_half_x, voxel[1] + half - model_half_y, voxel[2] + half),
                                     (voxel[0] + half - model_half_x, voxel[1] + half - model_half_y, voxel[2] + half)]
                        model.meshes[mtl_id].extend(back_face)

                    right = "{},{},{}".format(voxel[0] + 1, voxel[1], voxel[2])
                    if right not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[right]) == VoxGlassMaterial and not working_is_glass):
                        right_face = [(voxel[0] + half - model_half_x, voxel[1] - half - model_half_y, voxel[2] - half),
                                      (voxel[0] + half - model_half_x, voxel[1] + half - model_half_y, voxel[2] - half),
                                      (voxel[0] + half - model_half_x, voxel[1] + half - model_half_y, voxel[2] + half),
                                      (voxel[0] + half - model_half_x, voxel[1] - half - model_half_y, voxel[2] + half)]
                        model.meshes[mtl_id].extend(right_face)

                    left = "{},{},{}".format(voxel[0] - 1, voxel[1], voxel[2])
                    if left not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[left]) == VoxGlassMaterial and not working_is_glass):
                        left_face = [(voxel[0] - half - model_half_x, voxel[1] + half - model_half_y, voxel[2] - half),
                                     (voxel[0] - half - model_half_x, voxel[1] - half - model_half_y, voxel[2] - half),
                                     (voxel[0] - half - model_half_x, voxel[1] - half - model_half_y, voxel[2] + half),
                                     (voxel[0] - half - model_half_x, voxel[1] + half - model_half_y, voxel[2] + half)]
                        model.meshes[mtl_id].extend(left_face)

                    top = "{},{},{}".format(voxel[0], voxel[1], voxel[2] + 1)
                    if top not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[top]) == VoxGlassMaterial and not working_is_glass):
                        top_face = [(voxel[0] - half - model_half_x, voxel[1] - half - model_half_y, voxel[2] + half),
                                    (voxel[0] + half - model_half_x, voxel[1] - half - model_half_y, voxel[2] + half),
                                    (voxel[0] + half - model_half_x, voxel[1] + half - model_half_y, voxel[2] + half),
                                    (voxel[0] - half - model_half_x, voxel[1] + half - model_half_y, voxel[2] + half)]
                        model.meshes[mtl_id].extend(top_face)

                    bottom = "{},{},{}".format(voxel[0], voxel[1], voxel[2] - 1)
                    if bottom not in pos_sorted_voxels or (
                            type(pos_sorted_voxels[bottom]) == VoxGlassMaterial and not working_is_glass):
                        bottom_face = [
                            (voxel[0] + half - model_half_x, voxel[1] - half - model_half_y, voxel[2] - half),
                            (voxel[0] - half - model_half_x, voxel[1] - half - model_half_y, voxel[2] - half),
                            (voxel[0] - half - model_half_x, voxel[1] + half - model_half_y, voxel[2] - half),
                            (voxel[0] + half - model_half_x, voxel[1] + half - model_half_y, voxel[2] - half)]
                        model.meshes[mtl_id].extend(bottom_face)

    def convert(self):
        print("\nImporting voxel file {}\n".format(self.vox_file_path))

        import time
        time_start = time.time()
        VoxBaseMaterial.initialize(self.gamma_correct, self.gamma_value)
        VoxReader(self.vox_file_path).read()

        self.asset_name = os.path.splitext(os.path.basename(self.vox_file_path))[0]
        self.stage = Usd.Stage.CreateNew(os.path.join(r"C:\temp", "{}.usda".format(self.asset_name)))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        asset_prim = UsdGeom.Xform.Define(self.stage, "/" + self.asset_name)
        Usd.ModelAPI(asset_prim).SetKind(Kind.Tokens.component)
        self.geometry_scope = UsdGeom.Scope.Define(self.stage, asset_prim.GetPath().AppendPath(GEOMETRY_SCOPE_NAME))
        self.looks_scope = UsdGeom.Scope.Define(self.stage, asset_prim.GetPath().AppendPath(LOOKS_SCOPE_NAME))

        used_palette_indices = set()
        self.calculate_simple_meshes(used_palette_indices)

        self.used_mtls = {}
        for index in used_palette_indices:
            mtl = VoxBaseMaterial.get(index)
            if self.use_omni_mtls:
                try:
                    mtl = create_omni_mtl(self.stage, self.looks_scope, "VoxelMtl_{}".format(mtl.get_display_id()), mtl)
                except Exception as e:
                    print(e)
                    mtl = create_usd_preview_surface_mtl(self.stage, self.looks_scope, "VoxelMtl_{}".format(mtl.get_display_id()), mtl)
            else:
                mtl = create_usd_preview_surface_mtl(self.stage, self.looks_scope,
                                                     "VoxelMtl_{}".format(mtl.get_display_id()), mtl)
            self.used_mtls[index] = mtl

        print("Start converting")
        self.total_voxels = 0
        self.total_triangles = 0
        for top_node in VoxNode.top_nodes:
            self.__convert_node(top_node, self.geometry_scope)

        self.stage.GetRootLayer().Save()
        print("Converted {} total voxels".format(self.total_voxels))
        print("Converted {} total triangles".format(self.total_triangles))
        print("\nSuccessfully converted {} in {:.3f} sec".format(self.vox_file_path, time.time() - time_start))
        return {'FINISHED'}

    def __convert_node(self, node, parent_prim):
        print("node_id", node.node_id)
        if node.position is not None:
            xform = UsdGeom.Xform.Define(self.stage,
                                         parent_prim.GetPath().AppendPath("VoxelNode_{}".format(node.node_id)))
            xform.AddTranslateOp().Set(Gf.Vec3f(*node.position))
            parent_prim = xform
        if node.children:
            for child in node.children:
                self.__convert_node(child, parent_prim)
        elif node.model is not None:
            # Undo extra vertical translation that MV adds to all models
            translate_attr = parent_prim.GetPrim().GetAttribute("xformOp:translate")
            curr_trans = list(translate_attr.Get())
            new_trans = [curr_trans[0], curr_trans[1], curr_trans[2] - node.model.size[2] / 2.0]
            translate_attr.Set(Gf.Vec3f(*new_trans))
            if self.use_point_instancing:
                self.__voxels2point_instances(node, parent_prim)
            else:
                # self.__voxels2prims(node)
                self.__voxels2greedy_meshes(node, parent_prim)

    def __voxels2meshes(self, node, parent_prim):
        self.total_voxels += len(node.model.voxels)
        xform = UsdGeom.Xform.Define(self.stage, parent_prim.GetPath().AppendPath("VoxelShape_{}".format(node.node_id)))
        if node.position is not None:
            xform.AddTranslateOp().Set(Gf.Vec3f(*node.position))
        for mtl_id, mesh_verts in node.model.meshes.items():
            mtl_display_id = VoxBaseMaterial.get(mtl_id).get_display_id()
            mesh = UsdGeom.Mesh.Define(self.stage, xform.GetPath().AppendPath("VoxelMesh_{}".format(mtl_display_id)))
            mesh.CreatePointsAttr(mesh_verts)
            face_count = int(len(mesh_verts) / 4.0)
            self.total_triangles += face_count * 2
            mesh.CreateFaceVertexCountsAttr([4]*face_count)
            mesh.CreateFaceVertexIndicesAttr(list(range(len(mesh_verts))))
            if self.use_palette:
                UsdShade.MaterialBindingAPI(mesh).Bind(self.used_mtls[mtl_id])

    def __voxels2greedy_meshes(self, node, parent_prim):
        self.total_voxels += len(node.model.voxels)
        xform = UsdGeom.Xform.Define(self.stage, parent_prim.GetPath().AppendPath("VoxelShape_{}".format(node.node_id)))
        if node.position is not None:
            xform.AddTranslateOp().Set(Gf.Vec3f(*node.position))
        for mtl_id, mesh_verts in node.model.meshes.items():
            mtl_display_id = VoxBaseMaterial.get(mtl_id).get_display_id()
            mesh = UsdGeom.Mesh.Define(self.stage, xform.GetPath().AppendPath("VoxelMesh_{}".format(mtl_display_id)))
            mesh.CreatePointsAttr(mesh_verts)
            face_count = int(len(mesh_verts) / 4.0)
            self.total_triangles += face_count * 2
            mesh.CreateFaceVertexCountsAttr([4]*face_count)
            mesh.CreateFaceVertexIndicesAttr(list(range(len(mesh_verts))))
            if self.use_palette:
                UsdShade.MaterialBindingAPI(mesh).Bind(self.used_mtls[mtl_id])

    def __voxels2point_instances(self, node, parent_prim):
        instancer = UsdGeom.PointInstancer.Define(self.stage, parent_prim.GetPath().AppendPath(
            "VoxModel_{}".format(node.model.model_id)))
        instancer.CreatePrototypesRel()
        proto_container = self.stage.OverridePrim(instancer.GetPath().AppendPath("Prototypes"))
        mtl2proto_id = {}
        for proto_id, item in enumerate(self.used_mtls.items()):
            mtl_id, mtl = item
            mtl2proto_id[mtl_id] = proto_id
            mtl_display_id = VoxBaseMaterial.get(mtl_id).get_display_id()
            cube = UsdGeom.Cube.Define(self.stage,
                                       proto_container.GetPath().AppendPath("Voxel_{}".format(mtl_display_id)))
            cube.CreateSizeAttr(self.voxel_size)
            if self.use_physics:
                physics_apis = Sdf.TokenListOp.Create(["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"])
                cube_prim = cube.GetPrim()
                cube_prim.SetMetadata("apiSchemas", physics_apis)
                cube_prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(1)
                cube_prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(0)
                cube_prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(1)
                cube_prim.CreateAttribute("physics:startsAsleep", Sdf.ValueTypeNames.Bool).Set(1)
            if self.use_palette:
                UsdShade.MaterialBindingAPI(cube).Bind(mtl)
            instancer.GetPrototypesRel().AddTarget(cube.GetPath())
        # HACK: Need to reset the specifier because it gets set to "def" after adding children
        # proto_container.GetPrim().SetSpecifier(Sdf.SpecifierOver)

        # TODO: Can I make prototypes a relative path?
        ids = []
        positions = []
        for idx, voxel in enumerate(node.model.voxels):
            self.total_voxels += 1
            ids.append(mtl2proto_id[voxel[3]])
            position = [float(coord) * self.voxel_spacing for coord in voxel[:3]]
            # XY center the local origin on the model
            position = [position[0] - node.model.size[0] / 2.0,
                        position[1] - node.model.size[1] / 2.0,
                        position[2]]
            positions.append(position)
        instancer.CreateProtoIndicesAttr()
        instancer.CreatePositionsAttr()
        instancer.GetProtoIndicesAttr().Set(ids)
        instancer.GetPositionsAttr().Set(positions)

    def __voxels2prims(self, node, parent_prim):
        if self.join_voxels:
            pass
        voxel_xform = UsdGeom.Xform.Define(self.stage, "/Voxel")
        voxel_mesh = UsdGeom.Cube.Define(self.stage, voxel_xform.GetPath().AppendPath("VoxelMesh"))
        voxel_mesh.CreateSizeAttr(self.voxel_size)
        for idx, voxel in enumerate(node.model.voxels):
            self.total_voxels += 1
            # xform = UsdGeom.Xform.Define(stage, geometry_scope.GetPath().AppendPath("Voxel_{}".format(idx)))
            # TODO: Fix reference to same file
            # xform.GetPrim().GetReferences().AddReference(stage.GetRootLayer().identifier, voxel_xform.GetPath())
            # xform.GetPrim().SetInstanceable(True)
            cube = UsdGeom.Cube.Define(self.stage,
                                       parent_prim.GetPath().AppendPath("Voxel_{}".format(idx)))
            cube.CreateSizeAttr(self.voxel_size)
            position = [float(coord) * self.voxel_spacing for coord in voxel[:3]]
            position = [position[0] - node.model.size[0] / 2.0,
                        position[1] - node.model.size[1] / 2.0,
                        position[2]]
            # xform.AddTranslateOp().Set(Gf.Vec3f(*position))
            cube.AddTranslateOp().Set(Gf.Vec3f(*position))

            if self.use_physics:
                physics_apis = Sdf.TokenListOp.Create(["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"])
                cube_prim = cube.GetPrim()
                cube_prim.SetMetadata("apiSchemas", physics_apis)
                cube_prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(1)
                cube_prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(0)
                cube_prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(1)
                cube_prim.CreateAttribute("physics:startsAsleep", Sdf.ValueTypeNames.Bool).Set(1)
            if self.use_palette:
                # UsdShade.MaterialBindingAPI(xform).Bind(used_mtls[voxel[3]])
                UsdShade.MaterialBindingAPI(cube).Bind(self.used_mtls[voxel[3]])


def create_usd_preview_surface_mtl(stage, looks_scope, name, vox_mtl):
    mtl = UsdShade.Material.Define(stage, looks_scope.GetPath().AppendPath(name))
    prvw_shader = UsdShade.Shader.Define(stage, mtl.GetPath().AppendPath("prvw_shader"))
    prvw_shader = vox_mtl.populate_usd_preview_surface(prvw_shader)
    mtl.CreateSurfaceOutput().ConnectToSource(prvw_shader, "surface")
    return mtl


def create_omni_mtl(stage, looks_scope, name, vox_mtl):
    mtl = UsdShade.Material.Define(stage, looks_scope.GetPath().AppendPath(name))
    omni_shader = UsdShade.Shader.Define(stage, mtl.GetPath().AppendPath("omni_shader"))
    omni_shader = vox_mtl.populate_omni_shader(omni_shader)
    mtl.CreateSurfaceOutput().ConnectToSource(omni_shader, "surface")
    return mtl


if __name__ == '__main__':
    Vox2UsdConverter(r"C:\temp\greedy_simple.vox", use_palette=True, use_physics=False, use_point_instancing=False, use_omni_mtls=False).convert()
