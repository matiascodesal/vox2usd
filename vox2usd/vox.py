import struct

from pxr import Usd, UsdShade, Gf, Sdf

VOX_MAX_DICT_KEY_VALUE_PAIRS = 256

OMNI_ROUGHNESS_SCALAR = 1
OMNI_EMISSIVE_INTENSITY_SCALAR = 20000


class VoxReader(object):
    def __init__(self, vox_file_path):
        self.vox_file_path = vox_file_path
        self.vox_file = None

    def read(self):
        with open(self.vox_file_path, 'rb') as self.vox_file:
            VoxNode.initialize()
            VoxModel.initialize()

            # assert is VOX 150 file
            assert (struct.unpack('<4ci', self.vox_file.read(8)) == (b'V', b'O', b'X', b' ', 150))

            # MAIN chunk
            assert (struct.unpack('<4c', self.vox_file.read(4)) == (b'M', b'A', b'I', b'N'))
            N, M = struct.unpack('<ii', self.vox_file.read(8))
            assert (N == 0)  # MAIN chunk should have no content

            # M is remaining # of bytes in file
            model_num = 0
            while True:
                try:
                    data = struct.unpack('<4cii', self.vox_file.read(12))
                    name = data[:-2]
                    s_self = data[-2]
                    s_child = data[-1]
                    assert (s_child == 0)  # sanity check
                    name = b''.join(name).decode('utf-8')  # unsure of encoding..
                except struct.error:
                    # end of file
                    break

                if name == 'PACK':
                    # number of models
                    num_models, = struct.unpack('<i', self.vox_file.read(4))
                    # clamp load_frame to total number of frames
                    load_frame = min(load_frame, num_models)
                elif name == 'SIZE':
                    # model size
                    size = list(struct.unpack('<3i', self.vox_file.read(12)))
                    VoxModel(model_num, size)
                elif name == 'XYZI':
                    # voxel data
                    num_voxels, = struct.unpack('<i', self.vox_file.read(4))
                    voxels = []
                    for voxel in range(num_voxels):
                        voxel_data = struct.unpack('<4B', self.vox_file.read(4))
                        voxels.append(voxel_data)
                    VoxModel.get(model_num).add_voxels(voxels)
                    model_num += 1
                elif name == 'nTRN':
                    node_id, = struct.unpack('<i', self.vox_file.read(4))
                    # node name and hidden
                    node_dict = self.__read_vox_dict()
                    if node_dict is not None:
                        pass
                    child_node_id, = struct.unpack('<i', self.vox_file.read(4))
                    reserved_id, = struct.unpack('<i', self.vox_file.read(4))
                    layer_id, = struct.unpack('<i', self.vox_file.read(4))
                    num_frames, = struct.unpack('<i', self.vox_file.read(4))
                    node = VoxNode.get_or_create_node(node_id, register_new_as_top=True)
                    child = VoxNode.get_or_create_node(child_node_id)
                    node.children.append(child)

                    # TODO: assert (reserved_id == UINT32_MAX & & num_frames == 1); // must be these values according to the spec
                    frame_dict = self.__read_vox_dict()
                    if frame_dict and "_t" in frame_dict:
                        node.position = [float(component) for component in frame_dict["_t"].split()]
                    print("xform", node_id, node.position)
                    # TODO: Rotation
                    print("frame_dict", frame_dict)
                    if frame_dict is not None:
                        pass
                elif name == 'nGRP':
                    node_id, = struct.unpack('<i', self.vox_file.read(4))
                    node_dict = self.__read_vox_dict()
                    print("node_dict", node_dict)
                    # node dict is unused
                    assert (node_dict is None)
                    num_children, = struct.unpack('<i', self.vox_file.read(4))
                    child_ids = struct.unpack('<{}i'.format(num_children), self.vox_file.read(4 * num_children))
                    node = VoxNode.get_or_create_node(node_id, register_new_as_top=True)
                    for child_id in child_ids:
                        child = VoxNode.get_or_create_node(child_id)
                        print(node)
                        node.children.append(child)
                elif name == 'nSHP':
                    node_id, = struct.unpack('<i', self.vox_file.read(4))
                    node_dict = self.__read_vox_dict()
                    # node dict is unused
                    assert (node_dict is None)
                    num_models, = struct.unpack('<i', self.vox_file.read(4))
                    assert (num_models == 1)  # must be 1 according to spec
                    model_id, = struct.unpack('<i', self.vox_file.read(4))

                    print("node_id", node_id)
                    print("model_id", model_id)
                    model_dict = self.__read_vox_dict()
                    shape = VoxNode.get_or_create_node(node_id)
                    shape.model = VoxModel.get(model_id)
                    # model dict is unused
                    assert (model_dict is None)
                elif name == 'LAYR':
                    layer_id, = struct.unpack('<i', self.vox_file.read(4))
                    layer_dict = self.__read_vox_dict()
                    reserved_id, = struct.unpack('<i', self.vox_file.read(4))
                    assert (reserved_id == -1)
                elif name == 'RGBA':
                    # palette
                    for col in range(256):
                        palette_id = col + 1
                        color = list(struct.unpack('<4B', self.vox_file.read(4)))
                        gamma_corrected = [pow(col / 255.0, VoxBaseMaterial.gamma_value) for col in color[:3]]
                        gamma_corrected.append(color[3] / 255.0)
                        VoxBaseMaterial(palette_id, gamma_corrected)
                elif name == 'MATT':
                    # material
                    matt_id, mat_type, weight = struct.unpack('<iif', self.vox_file.read(12))
                    prop_bits, = struct.unpack('<i', self.vox_file.read(4))
                    binary = bin(prop_bits)
                    # Need to read property values, but this gets fiddly
                    # TODO: finish implementation
                    # We have read 16 bytes of this chunk so far, ignoring remainder
                    self.vox_file.read(s_self - 16)
                elif name == 'MATL':
                    palette_id, = struct.unpack('<i', self.vox_file.read(4))
                    mtl = VoxBaseMaterial.get(palette_id)
                    mtl_dict = self.__read_vox_dict()
                    VoxBaseMaterial.create_subclass(palette_id, mtl.color, mtl_dict, display_id=mtl.display_id)
                elif name == 'rOBJ':
                    # skip
                    self.vox_file.read(s_self)
                elif name == 'rCAM':
                    # skip
                    self.vox_file.read(s_self)
                elif name == 'NOTE':
                    # skip
                    self.vox_file.read(s_self)
                elif name == 'IMAP':
                    for col in range(256):
                        palette_id = col + 1
                        remapped_id, = struct.unpack('<B', self.vox_file.read(1))
                        VoxBaseMaterial.get(palette_id).display_id = remapped_id
                else:
                    # Any other chunk, we don't know how to handle
                    # This puts us out-of-step
                    raise RuntimeError("Unknown Chunk id {}".format(name))

    def __read_vox_dict(self):
        data = {}
        num_items, = struct.unpack('<i', self.vox_file.read(4))
        assert (num_items <= VOX_MAX_DICT_KEY_VALUE_PAIRS)
        for x in range(num_items):
            key_str_size, = struct.unpack('<i', self.vox_file.read(4))
            key, = struct.unpack('{}s'.format(key_str_size), self.vox_file.read(key_str_size))
            value_str_size, = struct.unpack('<i', self.vox_file.read(4))
            value, = struct.unpack('{}s'.format(value_str_size), self.vox_file.read(value_str_size))
            data[key] = value

        return data or None


"""
Value Ranges
_ior = 0-2 (UI: 1-3)
_ldr = 0-1
_rough = 0-1
_metal = 0-1
_emit = 0-1
_sp = 1-2
_alpha 0-1
_flux = 0-4
_d
_trans= 0-1
"""


class Voxel(object):
    def __init__(self, x, y, z, mtl_id):
        self.x = x
        self.y = y
        self.z = z
        self.mtl_id = mtl_id
        self.position = [x, y, z]


class VoxNode(object):
    instances = {}
    top_nodes = []

    def __init__(self, node_id):
        self.node_id = node_id
        self.children = []
        self.model = None
        self.position = None
        VoxNode.instances[node_id] = self

    @staticmethod
    def initialize():
        VoxNode.instances = {}
        VoxNode.top_level_nodes = []

    @staticmethod
    def get(node_id):
        return VoxNode.instances[node_id]

    @staticmethod
    def get_or_create_node(node_id, register_new_as_top=False):
        if node_id in VoxNode.instances:
            return VoxNode.instances[node_id]
        else:
            node = VoxNode(node_id)
            if register_new_as_top:
                VoxNode.top_nodes.append(node)
            return node


class VoxModel(object):
    instances = {}

    def __init__(self, model_id, size):
        self.model_id = model_id
        self.size = size
        self.voxels = []
        self.shape = None
        self.meshes = {}
        VoxModel.instances[model_id] = self

    @staticmethod
    def initialize():
        VoxModel.instances = {}

    @staticmethod
    def get(model_id):
        return VoxModel.instances[model_id]

    @staticmethod
    def get_all():
        return VoxModel.instances.values()

    def add_voxels(self, voxels):
        self.voxels.extend(voxels)


class VoxBaseMaterial(object):
    instances = {}
    gamma_correct = False
    gamma_value = 2.2

    def __init__(self, palette_id, color):
        self.palette_id = palette_id
        self.display_id = None
        self.color = color
        VoxBaseMaterial.instances[palette_id] = self

    @staticmethod
    def initialize(gamma_correct, gamma_value):
        VoxBaseMaterial.gamma_correct = gamma_correct
        if not gamma_correct:
            gamma_value = 1.0
        VoxBaseMaterial.gamma_value = gamma_value
        for col in range(len(DEFAULT_PALETTE)):
            VoxBaseMaterial.instances[col + 1] = struct.unpack('<4B', struct.pack('<I', DEFAULT_PALETTE[col]))

    @staticmethod
    def get(palette_id):
        return VoxBaseMaterial.instances[palette_id]

    def get_display_id(self):
        return self.display_id or self.palette_id

    @classmethod
    def create_subclass(cls, palette_id, color, mtl_dict, display_id=None):
        mtl = None
        if "_type" not in mtl_dict:
            mtl = VoxDiffuseMaterial(palette_id, color, mtl_dict)
        elif mtl_dict["_type"] == "_glass":
            mtl = VoxGlassMaterial(palette_id, color, mtl_dict)
        elif mtl_dict["_type"] == "_emit":
            mtl = VoxEmitMaterial(palette_id, color, mtl_dict)
        elif mtl_dict["_type"] == "_metal":
            mtl = VoxMetalMaterial(palette_id, color, mtl_dict)
        elif mtl_dict["_type"] == "_blend":
            mtl = VoxBlendMaterial(palette_id, color, mtl_dict)
        elif mtl_dict["_type"] == "_media":
            # TODO: Implement Media type
            mtl = VoxDiffuseMaterial(palette_id, color, mtl_dict)

        mtl.display_id = display_id

        return mtl


class VoxDiffuseMaterial(VoxBaseMaterial):
    _rough = 1.0

    def __init__(self, palette_id, color, mtl_dict):
        super(VoxDiffuseMaterial, self).__init__(palette_id, color)
        self._rough = float(mtl_dict.get("_rough", VoxDiffuseMaterial._rough))
        self._ior = float(mtl_dict.get("_ior", 0.3))

    def populate_usd_preview_surface(self, shader):
        shader.CreateIdAttr("UsdPreviewSurface")
        # We store the track from MV, but we use 1.0 for Diffuse materials
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(VoxDiffuseMaterial._rough)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1 + self._ior)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*self.color[0:3]))
        return shader

    def populate_omni_shader(self, shader):
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        try:
            shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        except AttributeError:
            sub_id = shader.GetPrim().CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token,
                                                      custom=False, variability=Sdf.VariabilityUniform)
            sub_id.Set("OmniPBR")
        # We track the roughness from MV, but we use 1.0 for Diffuse materials
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(VoxDiffuseMaterial._rough)
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*self.color[0:3]))
        return shader


class VoxGlassMaterial(VoxBaseMaterial):
    _rough = 0.0

    def __init__(self, palette_id, color, mtl_dict):
        super(VoxGlassMaterial, self).__init__(palette_id, color)
        self._rough = float(mtl_dict.get("_rough", VoxGlassMaterial._rough))
        self._ior = float(mtl_dict.get("_ior", 0.3))
        self._trans = float(mtl_dict.get("_trans", 0.0))

    def populate_usd_preview_surface(self, shader):
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(self._rough)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1 + self._ior)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1 - self._trans)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*self.color[0:3]))
        return shader

    def populate_omni_shader(self, shader):
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
        try:
            shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
        except AttributeError:
            sub_id = shader.GetPrim().CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token,
                                                      custom=False, variability=Sdf.VariabilityUniform)
            sub_id.Set("OmniGlass")
        shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(self._rough * OMNI_ROUGHNESS_SCALAR)
        shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1 + self._ior)
        # Not using opacity for glass because you lose refraction.  You roughness for more opaque glass
        # if self._trans > 0.0:
        #     shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
        #     shader.CreateInput("cutout_opacity", Sdf.ValueTypeNames.Float).Set(1 - self._trans)
        glass_color = [col + self._trans for col in self.color[:3]]
        shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*glass_color[0:3]))
        return shader


class VoxEmitMaterial(VoxBaseMaterial):
    # Give it some non-zero default because even emit=0 in MV glows
    _emit = 0.1

    def __init__(self, palette_id, color, mtl_dict):
        super(VoxEmitMaterial, self).__init__(palette_id, color)
        self._emit = float(mtl_dict.get("_emit", VoxEmitMaterial._emit))
        self._flux = float(mtl_dict.get("_flux", 0.0))
        self._ldr = float(mtl_dict.get("_ldr", 0.0))

    def populate_usd_preview_surface(self, shader):
        shader.CreateIdAttr("UsdPreviewSurface")
        emissive_color = [self._emit * component for component in self.color[0:3]]
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive_color))
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*self.color[0:3]))
        return shader

    def populate_omni_shader(self, shader):
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        try:
            shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        except AttributeError:
            sub_id = shader.GetPrim().CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token,
                                                      custom=False, variability=Sdf.VariabilityUniform)
            sub_id.Set("OmniPBR")
        shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
        emissive_color = [self._emit * component for component in self.color[0:3]]
        shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive_color))
        emissive_intensity = self._flux * OMNI_EMISSIVE_INTENSITY_SCALAR + OMNI_EMISSIVE_INTENSITY_SCALAR
        shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(emissive_intensity)
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*self.color[0:3]))
        return shader


class VoxMetalMaterial(VoxBaseMaterial):
    _rough = 0.0 # different from Diffuse

    def __init__(self, palette_id, color, mtl_dict):
        super(VoxMetalMaterial, self).__init__(palette_id, color)
        self._rough = float(mtl_dict.get("_rough", VoxMetalMaterial._rough))
        self._ior = float(mtl_dict.get("_ior", 0.3))
        self._metal = float(mtl_dict.get("_metal", 0.0))
        self._specular = float(mtl_dict.get("_specular", 1.0))

    def populate_usd_preview_surface(self, shader):
        shader.CreateIdAttr("UsdPreviewSurface")
        # Need to divide roughness by 100 to be better results in omniverse
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(self._rough)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1 + self._ior)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(self._metal)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*self.color[0:3]))
        return shader

    def populate_omni_shader(self, shader):
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        try:
            shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        except AttributeError:
            sub_id = shader.GetPrim().CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token,
                                                      custom=False, variability=Sdf.VariabilityUniform)
            sub_id.Set("OmniPBR")
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(self._rough * OMNI_ROUGHNESS_SCALAR)
        shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(self._metal)
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*self.color[0:3]))
        return shader


class VoxBlendMaterial(VoxBaseMaterial):
    _rough = 0.0  # different from Diffuse

    def __init__(self, palette_id, color, mtl_dict):
        super(VoxBlendMaterial, self).__init__(palette_id, color)
        self._rough = float(mtl_dict.get("_rough", VoxBlendMaterial._rough))
        self._ior = float(mtl_dict.get("_ior", 0.3))
        self._metal = float(mtl_dict.get("_metal", 0.0))
        self._specular = float(mtl_dict.get("_specular", 1.0))
        self._trans = float(mtl_dict.get("_trans", 0.0))

    def populate_usd_preview_surface(self, shader):
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(self._rough)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1 + self._ior)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(self._metal)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1 - self._trans)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*self.color[0:3]))
        return shader

    def populate_omni_shader(self, shader):
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        try:
            shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        except AttributeError:
            sub_id = shader.GetPrim().CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token,
                                                      custom=False, variability=Sdf.VariabilityUniform)
            sub_id.Set("OmniPBR")
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(self._rough * OMNI_ROUGHNESS_SCALAR)
        shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(self._metal)
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*self.color[0:3]))
        if self._trans > 0.0:
            shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
            shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(1 - self._trans)
        return shader


class VoxMediaMaterial(VoxBaseMaterial):
    def __init__(self, palette_id, color, mtl_dict):
        super(VoxMediaMaterial, self).__init__(palette_id, color)
        self._rough = float(mtl_dict.get("_rough", 0.1))
        self._ior = float(mtl_dict.get("_ior", 0.3))
        self._density = float(mtl_dict.get("_density", 0.0))


# Default palette, given in .vox 150 specification format
DEFAULT_PALETTE = [0x00000000, 0xffffffff, 0xffccffff, 0xff99ffff, 0xff66ffff, 0xff33ffff, 0xff00ffff, 0xffffccff,
                   0xffccccff, 0xff99ccff, 0xff66ccff, 0xff33ccff, 0xff00ccff, 0xffff99ff, 0xffcc99ff, 0xff9999ff,
                   0xff6699ff, 0xff3399ff, 0xff0099ff, 0xffff66ff, 0xffcc66ff, 0xff9966ff, 0xff6666ff, 0xff3366ff,
                   0xff0066ff, 0xffff33ff, 0xffcc33ff, 0xff9933ff, 0xff6633ff, 0xff3333ff, 0xff0033ff, 0xffff00ff,
                   0xffcc00ff, 0xff9900ff, 0xff6600ff, 0xff3300ff, 0xff0000ff, 0xffffffcc, 0xffccffcc, 0xff99ffcc,
                   0xff66ffcc, 0xff33ffcc, 0xff00ffcc, 0xffffcccc, 0xffcccccc, 0xff99cccc, 0xff66cccc, 0xff33cccc,
                   0xff00cccc, 0xffff99cc, 0xffcc99cc, 0xff9999cc, 0xff6699cc, 0xff3399cc, 0xff0099cc, 0xffff66cc,
                   0xffcc66cc, 0xff9966cc, 0xff6666cc, 0xff3366cc, 0xff0066cc, 0xffff33cc, 0xffcc33cc, 0xff9933cc,
                   0xff6633cc, 0xff3333cc, 0xff0033cc, 0xffff00cc, 0xffcc00cc, 0xff9900cc, 0xff6600cc, 0xff3300cc,
                   0xff0000cc, 0xffffff99, 0xffccff99, 0xff99ff99, 0xff66ff99, 0xff33ff99, 0xff00ff99, 0xffffcc99,
                   0xffcccc99, 0xff99cc99, 0xff66cc99, 0xff33cc99, 0xff00cc99, 0xffff9999, 0xffcc9999, 0xff999999,
                   0xff669999, 0xff339999, 0xff009999, 0xffff6699, 0xffcc6699, 0xff996699, 0xff666699, 0xff336699,
                   0xff006699, 0xffff3399, 0xffcc3399, 0xff993399, 0xff663399, 0xff333399, 0xff003399, 0xffff0099,
                   0xffcc0099, 0xff990099, 0xff660099, 0xff330099, 0xff000099, 0xffffff66, 0xffccff66, 0xff99ff66,
                   0xff66ff66, 0xff33ff66, 0xff00ff66, 0xffffcc66, 0xffcccc66, 0xff99cc66, 0xff66cc66, 0xff33cc66,
                   0xff00cc66, 0xffff9966, 0xffcc9966, 0xff999966, 0xff669966, 0xff339966, 0xff009966, 0xffff6666,
                   0xffcc6666, 0xff996666, 0xff666666, 0xff336666, 0xff006666, 0xffff3366, 0xffcc3366, 0xff993366,
                   0xff663366, 0xff333366, 0xff003366, 0xffff0066, 0xffcc0066, 0xff990066, 0xff660066, 0xff330066,
                   0xff000066, 0xffffff33, 0xffccff33, 0xff99ff33, 0xff66ff33, 0xff33ff33, 0xff00ff33, 0xffffcc33,
                   0xffcccc33, 0xff99cc33, 0xff66cc33, 0xff33cc33, 0xff00cc33, 0xffff9933, 0xffcc9933, 0xff999933,
                   0xff669933, 0xff339933, 0xff009933, 0xffff6633, 0xffcc6633, 0xff996633, 0xff666633, 0xff336633,
                   0xff006633, 0xffff3333, 0xffcc3333, 0xff993333, 0xff663333, 0xff333333, 0xff003333, 0xffff0033,
                   0xffcc0033, 0xff990033, 0xff660033, 0xff330033, 0xff000033, 0xffffff00, 0xffccff00, 0xff99ff00,
                   0xff66ff00, 0xff33ff00, 0xff00ff00, 0xffffcc00, 0xffcccc00, 0xff99cc00, 0xff66cc00, 0xff33cc00,
                   0xff00cc00, 0xffff9900, 0xffcc9900, 0xff999900, 0xff669900, 0xff339900, 0xff009900, 0xffff6600,
                   0xffcc6600, 0xff996600, 0xff666600, 0xff336600, 0xff006600, 0xffff3300, 0xffcc3300, 0xff993300,
                   0xff663300, 0xff333300, 0xff003300, 0xffff0000, 0xffcc0000, 0xff990000, 0xff660000, 0xff330000,
                   0xff0000ee, 0xff0000dd, 0xff0000bb, 0xff0000aa, 0xff000088, 0xff000077, 0xff000055, 0xff000044,
                   0xff000022, 0xff000011, 0xff00ee00, 0xff00dd00, 0xff00bb00, 0xff00aa00, 0xff008800, 0xff007700,
                   0xff005500, 0xff004400, 0xff002200, 0xff001100, 0xffee0000, 0xffdd0000, 0xffbb0000, 0xffaa0000,
                   0xff880000, 0xff770000, 0xff550000, 0xff440000, 0xff220000, 0xff110000, 0xffeeeeee, 0xffdddddd,
                   0xffbbbbbb, 0xffaaaaaa, 0xff888888, 0xff777777, 0xff555555, 0xff444444, 0xff222222, 0xff111111]

