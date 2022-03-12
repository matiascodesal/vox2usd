import inspect

from pxr import UsdGeom

GEOMETRY_SCOPE_NAME = "Geometry"
LOOKS_SCOPE_NAME = "Looks"
# 1 m = 32 voxels
DEFAULT_VOXELS_PER_METER = 1
DEFAULT_METERS_PER_VOXEL = 1.0 / DEFAULT_VOXELS_PER_METER


class CustomEnum(object):
    @classmethod
    def values(cls):
        return [name[1] for name in inspect.getmembers(cls) if not name[0].startswith('_') and not inspect.ismethod(name[1])]


class GeometryVariantSetNames(CustomEnum):
    MERGED_MESHES = "MergedMeshes"
    POINT_INSTANCES = "PointInstances"


class PointShapeVariantSetNames(CustomEnum):
    CUBES = "Cubes"
    SPHERES = "Spheres"
    STUDS = "Studs"


class ShaderVariantSetNames(CustomEnum):
    PREVIEW = "Preview"
    OMNIVERSE = "Omniverse"


class VoxelSides(CustomEnum):
    FRONT = 0   # -Y = Front
    BACK = 1    # +Y = Back
    RIGHT = 2   # -X = Right
    LEFT = 3    # +X = Left
    BOTTOM = 4  # -Z = Bottom
    TOP = 5     # +Z = Top


class StudMesh(object):
    points = [(0.5, 0.5, 0.5), (0.5, 0.5, -0.5), (0.5, -0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, -0.5, -0.5), (0, 0.35, 0.5), (0, 0.35, 0.7226549), (0.10815595, 0.3328698, 0.5), (0.10815595, 0.3328698, 0.7226549), (0.20572484, 0.28315595, 0.5), (0.20572484, 0.28315595, 0.7226549), (0.28315595, 0.20572484, 0.5), (0.28315595, 0.20572484, 0.7226549), (0.3328698, 0.108155936, 0.5), (0.3328698, 0.108155936, 0.7226549), (0.35, -1.5298985e-8, 0.5), (0.35, -1.5298985e-8, 0.7226549), (0.33286977, -0.10815596, 0.5), (0.33286977, -0.10815596, 0.7226549), (0.28315595, -0.2057248, 0.5), (0.28315595, -0.2057248, 0.7226549), (0.2057248, -0.28315598, 0.5), (0.2057248, -0.28315598, 0.7226549), (0.10815596, -0.33286977, 0.5), (0.10815596, -0.33286977, 0.7226549), (-3.059797e-8, -0.35, 0.5), (-3.059797e-8, -0.35, 0.7226549), (-0.108155936, -0.3328698, 0.5), (-0.108155936, -0.3328698, 0.7226549), (-0.20572488, -0.28315592, 0.5), (-0.20572488, -0.28315592, 0.7226549), (-0.28315598, -0.20572478, 0.5), (-0.28315598, -0.20572478, 0.7226549), (-0.33286977, -0.10815598, 0.5), (-0.33286977, -0.10815598, 0.7226549), (-0.35, 4.173708e-9, 0.5), (-0.35, 4.173708e-9, 0.7226549), (-0.33286977, 0.10815599, 0.5), (-0.33286977, 0.10815599, 0.7226549), (-0.28315598, 0.20572478, 0.5), (-0.28315598, 0.20572478, 0.7226549), (-0.20572485, 0.28315592, 0.5), (-0.20572485, 0.28315592, 0.7226549), (-0.10815593, 0.3328698, 0.5), (-0.10815593, 0.3328698, 0.7226549)]
    normals = [(0, -1, 0), (0, -1, 0), (0, -1, 0), (0, -1, 0), (-1, -0, 0), (-1, -0, 0), (-1, -0, 0), (-1, -0, 0), (0, 0, -1), (0, 0, -1), (0, 0, -1), (0, 0, -1), (1, -0, 0), (1, -0, 0), (1, -0, 0), (1, -0, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 0.6680807, 0.74404126), (0, 0.73912776, 0.673513), (0.22840053, 0.70296335, 0.673513), (0.2064272, 0.63539535, 0.74404126), (0.2064272, 0.63539535, 0.74404126), (0.22840053, 0.70296335, 0.673513), (0.4344615, 0.59797966, 0.673513), (0.39268166, 0.5404828, 0.74404126), (0.39268166, 0.5404828, 0.74404126), (0.4344615, 0.59797966, 0.673513), (0.59797966, 0.4344615, 0.673513), (0.5404828, 0.39268166, 0.74404126), (0.5404828, 0.39268166, 0.74404126), (0.59797966, 0.4344615, 0.673513), (0.70296335, 0.22840053, 0.673513), (0.63539535, 0.2064272, 0.74404126), (0.63539535, 0.2064272, 0.74404126), (0.70296335, 0.22840053, 0.673513), (0.73912776, 0, 0.673513), (0.6680807, 0, 0.74404126), (0.6680807, 0, 0.74404126), (0.73912776, 0, 0.673513), (0.70296335, -0.22840053, 0.673513), (0.63539535, -0.2064272, 0.74404126), (0.63539535, -0.2064272, 0.74404126), (0.70296335, -0.22840053, 0.673513), (0.59797966, -0.4344615, 0.673513), (0.5404828, -0.39268166, 0.74404126), (0.5404828, -0.39268166, 0.74404126), (0.59797966, -0.4344615, 0.673513), (0.4344615, -0.59797966, 0.673513), (0.39268166, -0.5404828, 0.74404126), (0.39268166, -0.5404828, 0.74404126), (0.4344615, -0.59797966, 0.673513), (0.22840053, -0.70296335, 0.673513), (0.2064272, -0.63539535, 0.74404126), (0.2064272, -0.63539535, 0.74404126), (0.22840053, -0.70296335, 0.673513), (0, -0.73912776, 0.673513), (0, -0.6680807, 0.74404126), (0, -0.6680807, 0.74404126), (0, -0.73912776, 0.673513), (-0.22840053, -0.70296335, 0.673513), (-0.2064272, -0.63539535, 0.74404126), (-0.2064272, -0.63539535, 0.74404126), (-0.22840053, -0.70296335, 0.673513), (-0.4344615, -0.59797966, 0.673513), (-0.39268166, -0.5404828, 0.74404126), (-0.39268166, -0.5404828, 0.74404126), (-0.4344615, -0.59797966, 0.673513), (-0.59797966, -0.4344615, 0.673513), (-0.5404828, -0.39268166, 0.74404126), (-0.5404828, -0.39268166, 0.74404126), (-0.59797966, -0.4344615, 0.673513), (-0.70296335, -0.22840053, 0.673513), (-0.63539535, -0.2064272, 0.74404126), (-0.63539535, -0.2064272, 0.74404126), (-0.70296335, -0.22840053, 0.673513), (-0.73912776, 0, 0.673513), (-0.6680807, 0, 0.74404126), (-0.6680807, 0, 0.74404126), (-0.73912776, 0, 0.673513), (-0.70296335, 0.22840053, 0.673513), (-0.63539535, 0.2064272, 0.74404126), (-0.63539535, 0.2064272, 0.74404126), (-0.70296335, 0.22840053, 0.673513), (-0.59797966, 0.4344615, 0.673513), (-0.5404828, 0.39268166, 0.74404126), (-0.5404828, 0.39268166, 0.74404126), (-0.59797966, 0.4344615, 0.673513), (-0.4344615, 0.59797966, 0.673513), (-0.39268166, 0.5404828, 0.74404126), (1.2793372e-7, 0, 1), (1.2793372e-7, 0, 1), (1.2793372e-7, 0, 1), (1.2793372e-7, 0, 1), (1.2793372e-7, 0, 1), (1.2793372e-7, 0, 1), (1.2793372e-7, 0, 1), (1.2793372e-7, 0, 1), (1.2793372e-7, 0, 1), (1.2793372e-7, 0, 1), (1.2793372e-7, 0, 1), (1.2793372e-7, 0, 1), (1.2793372e-7, 0, 1), (1.2793372e-7, 0, 1), (1.2793372e-7, 0, 1), (1.2793372e-7, 0, 1), (1.2793372e-7, 0, 1), (1.2793372e-7, 0, 1), (1.2793372e-7, 0, 1), (1.2793372e-7, 0, 1), (-0.39268166, 0.5404828, 0.74404126), (-0.4344615, 0.59797966, 0.673513), (-0.22840053, 0.70296335, 0.673513), (-0.2064272, 0.63539535, 0.74404126), (-0.2064272, 0.63539535, 0.74404126), (-0.22840053, 0.70296335, 0.673513), (0, 0.73912776, 0.673513), (0, 0.6680807, 0.74404126), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (-0, 0, 1), (-0, 0, 1), (-0, 0, 1), (-0, 0, 1), (0, -0, 1), (0, -0, 1), (0, -0, 1), (0, -0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (-0, -0, 1), (-0, -0, 1), (-0, -0, 1), (-0, -0, 1), (-0, -0, 1), (-0, -0, 1), (-0, -0, 1), (-0, -0, 1), (-0, -0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (-0, 0, 1), (-0, 0, 1), (-0, 0, 1), (-0, 0, 1), (-0, 0, 1), (-0, 0, 1), (-0, 0, 1), (-0, 0, 1), (-0, 0, 1), (0, -0, 1), (0, -0, 1), (0, -0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (-0, 0, 1), (-0, 0, 1), (-0, 0, 1), (0, -0, 1), (0, -0, 1), (0, -0, 1), (0, -0, 1), (0, -0, 1), (0, -0, 1), (0, -0, 1), (0, -0, 1), (0, -0, 1)]
    normals_interpolation = UsdGeom.Tokens.faceVarying
    face_vertex_counts = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 20, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    face_vertex_indices = [3, 2, 6, 7, 7, 6, 4, 5, 5, 1, 3, 7, 1, 0, 2, 3, 5, 4, 0, 1, 8, 9, 11, 10, 10, 11, 13, 12, 12, 13, 15, 14, 14, 15, 17, 16, 16, 17, 19, 18, 18, 19, 21, 20, 20, 21, 23, 22, 22, 23, 25, 24, 24, 25, 27, 26, 26, 27, 29, 28, 28, 29, 31, 30, 30, 31, 33, 32, 32, 33, 35, 34, 34, 35, 37, 36, 36, 37, 39, 38, 38, 39, 41, 40, 40, 41, 43, 42, 42, 43, 45, 44, 11, 9, 47, 45, 43, 41, 39, 37, 35, 33, 31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 44, 45, 47, 46, 46, 47, 9, 8, 6, 2, 26, 28, 2, 0, 16, 18, 0, 4, 46, 8, 4, 6, 36, 38, 46, 4, 44, 44, 4, 42, 42, 4, 40, 38, 40, 4, 8, 10, 0, 12, 0, 10, 14, 0, 12, 16, 0, 14, 18, 20, 2, 22, 2, 20, 24, 2, 22, 26, 2, 24, 28, 30, 6, 32, 6, 30, 34, 6, 32, 36, 6, 34]