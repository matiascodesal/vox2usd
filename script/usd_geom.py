from pxr import Gf, Kind, Sdf, Usd, UsdGeom

stage = Usd.Stage.CreateNew(r"C:\temp\mesh2.usda")
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
model_root = UsdGeom.Xform.Define(stage,"/Model")
Usd.ModelAPI(model_root).SetKind(Kind.Tokens.component)
mesh = UsdGeom.Mesh.Define(stage, "/Model/mesh")

mesh.CreatePointsAttr([(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1),
                      (1, 1, -1), (1, 1, 1)])

mesh.CreateFaceVertexCountsAttr([4, 4])
mesh.CreateFaceVertexIndicesAttr([0,1,2,3,1,4,5,2])

# mesh.CreatePointsAttr([(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1),
#                       (1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)])
#
# mesh.CreateFaceVertexCountsAttr([4, 4])
# mesh.CreateFaceVertexIndicesAttr([0,1,2,3,4,5,6,7])

stage.Save()