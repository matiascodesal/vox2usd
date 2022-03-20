
# TODO: Figure out MV instancing (Ref)
# TODO: add script args (input path, output path, etc)
# TODO: Should I rotate studs to always point up?
# TODO: Get rid of omni info:id errors
# TODO: Compute bboxes


import os

from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf, Kind, Vt

from vox2usd.constants import (GEOMETRY_SCOPE_NAME, LOOKS_SCOPE_NAME, DEFAULT_METERS_PER_VOXEL,
                               GeometryVariantSetNames, PointShapeVariantSetNames, ShaderVariantSetNames, StudMesh)
from vox2usd._meshing import GreedyMeshing
from vox2usd.vox import (VoxReader, VoxTransform, VoxGroup, VoxShape, VoxBaseMaterial, VoxGlassMaterial)


class Vox2UsdConverter(object):
    def __init__(self, vox_file, voxel_spacing=DEFAULT_METERS_PER_VOXEL, voxel_size=DEFAULT_METERS_PER_VOXEL,
                 gamma_correct=True, gamma_value=2.2, use_physics=False, flatten=False):
        self.vox_file_path = vox_file
        self.voxel_spacing = voxel_spacing
        self.voxel_size = voxel_size
        self.gamma_correct = gamma_correct
        self.gamma_value = gamma_value
        self.use_physics = use_physics
        self.flatten = flatten
        self.total_voxels = 0
        self.total_triangles = 0
        self.used_mtls = {}
        self.asset_name = os.path.splitext(os.path.basename(self.vox_file_path))[0]
        self.output_dir = os.path.dirname(self.vox_file_path)
        self.output_file_name = "{}.usd".format(self.asset_name)
        self.mesh_payload_identifier = os.path.join(self.output_dir, "{}.mesh.usdc".format(self.asset_name))
        self.points_payload_identifier = os.path.join(self.output_dir, "{}.points.usdc".format(self.asset_name))

        VoxBaseMaterial.initialize(self.gamma_correct, self.gamma_value)

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
        VoxReader(self.vox_file_path).read()


        stage_layer = Sdf.Layer.CreateNew(os.path.join(self.output_dir, self.output_file_name), args={"format": "usda"})
        self.stage = Usd.Stage.Open(stage_layer.identifier)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        asset_geom = UsdGeom.Xform.Define(self.stage, "/" + self.asset_name)
        self.stage.SetDefaultPrim(asset_geom.GetPrim())
        model_api = Usd.ModelAPI(asset_geom)
        model_api.SetKind(Kind.Tokens.component)
        model_api.SetAssetName(self.asset_name)
        model_api.SetAssetIdentifier(self.output_file_name)
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

        GreedyMeshing.generate(self.voxel_size)

        for index in VoxBaseMaterial.used_palette_ids:
            mtl = VoxBaseMaterial.get(index)
            mtl = self.create_material(self.stage, self.looks_scope, "VoxelMtl_{}".format(mtl.get_display_id()), mtl)
            self.used_mtls[index] = {
                 "mtl": mtl,
                 GeometryVariantSetNames.MERGED_MESHES: [],
                 GeometryVariantSetNames.POINT_INSTANCES: {
                     PointShapeVariantSetNames.CUBES: [],
                     PointShapeVariantSetNames.SPHERES: [],
                     PointShapeVariantSetNames.STUDS: []
                 }
            }

        print("Start converting")

        mesh_stage = Usd.Stage.CreateNew(self.mesh_payload_identifier)
        points_stage = Usd.Stage.CreateNew(self.points_payload_identifier)

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
                    target_prim = self.__get_target_prim(target_path, voxel_root)
                    UsdShade.MaterialBindingAPI(target_prim).Bind(mtl)
            self.geo_varset.SetVariantSelection(GeometryVariantSetNames.POINT_INSTANCES)
            with self.geo_varset.GetVariantEditContext():
                for pt_shape_variant in PointShapeVariantSetNames.values():
                    self.__bind_mtl_to_prototypes(voxel_root, pt_shape_variant, proto_targets, mtl)

        if self.flatten:
            self.geo_varset.SetVariantSelection(GeometryVariantSetNames.MERGED_MESHES)
            self.shader_varset.SetVariantSelection(ShaderVariantSetNames.PREVIEW)
            self.stage.Export(os.path.join(self.output_dir, self.output_file_name), args={"format": "usdc"})
        else:
            self.geo_varset.SetVariantSelection(GeometryVariantSetNames.MERGED_MESHES)
            self.shader_varset.SetVariantSelection(ShaderVariantSetNames.OMNIVERSE)
            self.point_shape_varset.SetVariantSelection(PointShapeVariantSetNames.CUBES)
            self.stage.Save()

        print("Converted {} total voxels".format(self.total_voxels))
        print("Converted {} total triangles".format(self.total_triangles))
        print("\nSuccessfully converted {} in {:.3f} sec".format(self.vox_file_path, time.time() - time_start))
        return {'FINISHED'}

    def __get_target_prim(self, target_path, voxel_root):
        target_path = target_path.pathString.split("/")
        target_path = "/".join(target_path[2:])
        target_path = voxel_root.GetPath().AppendPath(target_path)
        return self.stage.GetPrimAtPath(target_path)

    def __bind_mtl_to_prototypes(self, voxel_root, pt_shape_variant, proto_targets, mtl):
        self.point_shape_varset.SetVariantSelection(pt_shape_variant)
        with self.point_shape_varset.GetVariantEditContext():
            voxel_root.GetVariantSet("PointShape").SetVariantSelection(pt_shape_variant)
            for target_path in proto_targets[pt_shape_variant]:
                target_prim = self.__get_target_prim(target_path, voxel_root)
                UsdShade.MaterialBindingAPI(target_prim).Bind(mtl)

    def __convert_node(self, stage, node, parent_prim, geom_type):
        if isinstance(node, VoxTransform):
            xform_child = node.get_child()
            if isinstance(xform_child, VoxGroup):
                xform_path = parent_prim.GetPath().AppendPath("VoxelGroup_{}".format(xform_child.node_id))
                xform = UsdGeom.Xform.Define(stage, xform_path)
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

    @staticmethod
    def __set_pivot(vox_model, xformable):
        # TODO: This doesn't actually work for rotated models. The pivot doesn't end up at the bottom
        # MV pivot is at the center. I prefer to have it at the bottom.
        # I'm tweaking the translation here to counteract that.
        xform_attr = xformable.GetPrim().GetAttribute("xformOp:transform")
        curr_xform = xform_attr.Get()
        # Need to figure out the local
        up_vector = curr_xform * Gf.Vec4d(0, 0, 1, 1)
        up_vector = Gf.Vec3d(up_vector[0:3]).GetNormalized()
        # copy bottom row of matrix since index operator is read only on Matrix4d
        trans_row = curr_xform[3]
        for index in range(3):
            if up_vector[index] != 0:
                trans_row[index] = trans_row[index] - int(vox_model.size[2] / 2.0)
                break

        curr_xform.SetRow(3, Gf.Vec4d(*trans_row))
        xform_attr.Set(curr_xform)

    def __add_binding_target(self, mtl_id, target_path, geometry_var, point_shape_var=None):
        if geometry_var == GeometryVariantSetNames.MERGED_MESHES:
            self.used_mtls[mtl_id][geometry_var].append(target_path)
        elif geometry_var == GeometryVariantSetNames.POINT_INSTANCES:
            if point_shape_var is None:
                raise TypeError("You must provide a PointShapeVariantSetNames value for point_shape_var.")
            else:
                self.used_mtls[mtl_id][geometry_var][point_shape_var].append(target_path)

    def __fill_mesh(self, shape_node, mesh):
        display_colors = []
        opacity = []
        mtl_id, mesh_verts = list(shape_node.model.meshes.items())[0]
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
        self.__add_binding_target(mtl_id, mesh.GetPath(), GeometryVariantSetNames.MERGED_MESHES)

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
                self.__add_binding_target(item["mtl_id"], subset.GetPath(), GeometryVariantSetNames.MERGED_MESHES)

        self.total_triangles += total_face_count * 2

    def __voxels2meshes(self, stage, xform_node, shape_node, parent_prim):
        self.total_voxels += len(shape_node.model.voxels)
        mesh = UsdGeom.Mesh.Define(stage, parent_prim.GetPath().AppendPath("VoxelModel_{}".format(shape_node.node_id)))
        mesh.AddTransformOp().Set(Gf.Matrix4d(*xform_node.transform))
        # self.__set_pivot(shape_node.model, mesh)
        if len(shape_node.model.meshes.keys()) > 1:
            self.__create_geom_subset_per_mtl(shape_node, mesh)
        else:
            self.__fill_mesh(shape_node, mesh)

    def __create_cube_geom(self, stage, instancer, proto_container, mtl_id):
        mtl_display_id = VoxBaseMaterial.get(mtl_id).get_display_id()
        cube = UsdGeom.Cube.Define(stage,
                                   proto_container.GetPath().AppendPath("VoxelCube_{}".format(mtl_display_id)))
        cube.CreateSizeAttr(self.voxel_size)
        self.__add_binding_target(mtl_id, cube.GetPath(), GeometryVariantSetNames.POINT_INSTANCES,
                                  point_shape_var=PointShapeVariantSetNames.CUBES)
        self.__set_common_point_attrs(cube, instancer, mtl_id)

    def __create_sphere_geom(self, stage, instancer, proto_container, mtl_id):
        mtl_display_id = VoxBaseMaterial.get(mtl_id).get_display_id()
        sphere_path = proto_container.GetPath().AppendPath("VoxelSphere_{}".format(mtl_display_id))
        sphere = UsdGeom.Sphere.Define(stage, sphere_path)
        sphere.CreateRadiusAttr(self.voxel_size / 2.0)
        self.__add_binding_target(mtl_id, sphere.GetPath(), GeometryVariantSetNames.POINT_INSTANCES,
                                  point_shape_var=PointShapeVariantSetNames.SPHERES)
        self.__set_common_point_attrs(sphere, instancer, mtl_id)

    def __create_stud_geom(self, stage, instancer, proto_container, mtl_id):
        mtl_display_id = VoxBaseMaterial.get(mtl_id).get_display_id()
        mesh = UsdGeom.Mesh.Define(stage, proto_container.GetPath().AppendPath("VoxelStud_{}".format(mtl_display_id)))
        mesh.CreatePointsAttr(StudMesh.points)
        mesh.CreateFaceVertexCountsAttr(StudMesh.face_vertex_counts)
        mesh.CreateFaceVertexIndicesAttr(StudMesh.face_vertex_indices)
        mesh.CreateNormalsAttr(StudMesh.normals)
        mesh.SetNormalsInterpolation(StudMesh.normals_interpolation)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        self.__add_binding_target(mtl_id, mesh.GetPath(), GeometryVariantSetNames.POINT_INSTANCES,
                                  point_shape_var=PointShapeVariantSetNames.STUDS)
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

        ids = []
        positions = []
        for voxel in shape_node.model.voxels:
            mtl_id = voxel[3]
            if mtl_id not in mtl2proto_id:
                mtl2proto_id[mtl_id] = len(proto_container.GetChildren())
                pt_shape_varset.SetVariantSelection(PointShapeVariantSetNames.CUBES)
                with pt_shape_varset.GetVariantEditContext():
                    self.__create_cube_geom(stage, instancer, proto_container, mtl_id)
                pt_shape_varset.SetVariantSelection(PointShapeVariantSetNames.SPHERES)
                with pt_shape_varset.GetVariantEditContext():
                    self.__create_sphere_geom(stage, instancer, proto_container, mtl_id)
                pt_shape_varset.SetVariantSelection(PointShapeVariantSetNames.STUDS)
                with pt_shape_varset.GetVariantEditContext():
                    self.__create_stud_geom(stage, instancer, proto_container, mtl_id)

            ids.append(mtl2proto_id[mtl_id])
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
    Vox2UsdConverter(r"C:\temp\test_data\geomsubsets.vox", use_physics=False, flatten=False).convert()
