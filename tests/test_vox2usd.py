from pathlib import Path
import unittest

from pxr import Sdf,Usd, UsdShade

from vox2usd import Vox2UsdConverter

DATA_PATH = Path(__file__).parent.joinpath("data")


class TestUsdPrim(unittest.TestCase):
    def test_default_prim(self):
        filename = "test_geomsubsets"
        vox2usd = Vox2UsdConverter(DATA_PATH / "{}.vox".format(filename))
        vox2usd.convert()

        stage_file = DATA_PATH / "{}.usd".format(filename)
        stage = Usd.Stage.Open(str(stage_file))
        default_prim = stage.GetDefaultPrim()
        self.assertTrue(default_prim)
        self.assertEqual(default_prim.GetPath(), Sdf.Path("/test_geomsubsets"))

        model_api = Usd.ModelAPI(default_prim)
        self.assertEqual(model_api.GetAssetName(), filename)
        self.assertEqual(model_api.GetAssetIdentifier().path, "{}.usd".format(filename))
        self.assertEqual(model_api.GetKind(), "component")


    def test_variant_sets(self):
        filename = "test_geomsubsets"
        vox2usd = Vox2UsdConverter(DATA_PATH / "{}.vox".format(filename))
        vox2usd.convert()

        stage_file = DATA_PATH / "{}.usd".format(filename)
        stage = Usd.Stage.Open(str(stage_file))
        asset_prim = stage.GetDefaultPrim()
        vsets = asset_prim.GetVariantSets()
        self.assertTrue(vsets.HasVariantSet("Geometry"))
        geom_vset = vsets.GetVariantSet("Geometry")
        self.assertTrue(geom_vset.HasAuthoredVariant("MergedMeshes"))
        self.assertTrue(geom_vset.HasAuthoredVariant("PointInstances"))
        self.assertEqual(geom_vset.GetVariantSelection(), "MergedMeshes")
        self.assertTrue(vsets.HasVariantSet("Shader"))
        shader_vset = vsets.GetVariantSet("Shader")
        self.assertTrue(shader_vset.HasAuthoredVariant("Preview"))
        self.assertTrue(shader_vset.HasAuthoredVariant("Omniverse"))
        self.assertEqual(shader_vset.GetVariantSelection(), "Omniverse")
        self.assertFalse(vsets.HasVariantSet("PointShape"))

        geom_vset.SetVariantSelection("PointInstances")
        self.assertTrue(vsets.HasVariantSet("PointShape"))
        pt_shape_vset = vsets.GetVariantSet("PointShape")
        self.assertTrue(pt_shape_vset.HasAuthoredVariant("Cubes"))
        self.assertTrue(pt_shape_vset.HasAuthoredVariant("Spheres"))
        self.assertTrue(pt_shape_vset.HasAuthoredVariant("Studs"))
        self.assertEqual(pt_shape_vset.GetVariantSelection(), "Cubes")

    def test_multilayer_model(self):
        filename = "test_geomsubsets"
        vox2usd = Vox2UsdConverter(DATA_PATH / "{}.vox".format(filename))
        vox2usd.convert()

        stage_file = DATA_PATH / "{}.usd".format(filename)
        self.assertTrue(stage_file.exists())
        mesh_crate = DATA_PATH / "{}.mesh.usdc".format(filename)
        self.assertTrue(mesh_crate.exists())
        points_crate = DATA_PATH / "{}.points.usdc".format(filename)
        self.assertTrue(points_crate.exists())

        stage = Usd.Stage.Open(str(stage_file))
        mesh_prim = stage.GetPrimAtPath("/test_geomsubsets/Geometry/VoxelRoot/VoxelModel_3")
        self.assertTrue(mesh_prim)
        self.assertEqual(mesh_prim.GetTypeName(), "Mesh")

        asset_prim = stage.GetDefaultPrim()
        vsets = asset_prim.GetVariantSets()
        geom_vset = vsets.GetVariantSet("Geometry")
        result = geom_vset.SetVariantSelection("PointInstances")
        pt_instancer_prim = stage.GetPrimAtPath("/test_geomsubsets/Geometry/VoxelRoot/VoxelModel_3")
        self.assertTrue(pt_instancer_prim)
        self.assertEqual(pt_instancer_prim.GetTypeName(), "PointInstancer")

    def test_geomsubsets(self):
        filename = "test_geomsubsets"
        vox2usd = Vox2UsdConverter(DATA_PATH / "{}.vox".format(filename))
        vox2usd.convert()
        stage_file = DATA_PATH / "{}.usd".format(filename)
        self.assertTrue(stage_file.exists())
        mesh_crate = DATA_PATH / "{}.mesh.usdc".format(filename)
        self.assertTrue(mesh_crate.exists())
        points_crate = DATA_PATH / "{}.points.usdc".format(filename)
        self.assertTrue(points_crate.exists())
        stage = Usd.Stage.Open(str(stage_file))
        self.assertTrue(stage)
        mesh_prim = stage.GetPrimAtPath("/test_geomsubsets/Geometry/VoxelRoot/VoxelModel_3")
        geomsubsets = mesh_prim.GetChildren()
        self.assertEqual(len(geomsubsets), 2)
        for subset in geomsubsets:
            face_indices = subset.GetAttribute("indices").Get()
            self.assertTrue(len(face_indices), 5)
            binding_api = UsdShade.MaterialBindingAPI.Get(stage, subset.GetPath())
            mtl_path = binding_api.GetDirectBindingRel().GetTargets()[0]
            self.assertTrue(mtl_path)
