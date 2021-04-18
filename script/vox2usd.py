import numpy as np
# from skimage.measure import marching_cubes_lewiner
# x = [0,0,0,0,0,1,-1]
# y = [0,0,0,1,-1,0,0]
# z = [0,1,-1,0,0,0,0]
# #X, Y, Z = np.meshgrid(x,y,z)
# X, Y, Z = np.mgrid[:30, :30, :30]
# u = (X-15)**2 + (Y-15)**2 + (Z-15)**2 - 8**2
# verts, faces, normals, values = marching_cubes_lewiner(u, level=30)
#
# thefile = open(r'C:\temp\test.obj', 'w')
#
# for v in verts:
#   thefile.write("v {} {} {}\n".format(*v))
#
# for f in faces:
#   thefile.write("f {} {} {}\n".format(*(f + 1)))
#
# thefile.close()


import mcubes

# Create a data volume (30 x 30 x 30)
X, Y, Z = np.mgrid[:30, :30, :30]
u = (X - 15) ** 2 + (Y - 15) ** 2 + (Z - 15) ** 2 - 8 ** 2

# Extract the 0-isosurface
vertices, triangles = mcubes.marching_cubes(u, 0.5)

# Or export to an OBJ file
mcubes.export_obj(vertices, triangles, r'C:\temp\sphere.obj')