from pathlib import Path

from OpenGL import GL

from entities.components.visuals.assets import AssetStatus
from entities.components.visuals.material import Material
from visuals.shader import Shader
from visuals.src_utils import read_source_file


class GGXHammonShader(Shader):
    def __init__(self, vertex_source: str, fragment_source: str):
        super().__init__(vertex_source, fragment_source)

        self.u_albedo = GL.glGetUniformLocation(self.program, "u_Albedo")
        self.u_roughness = GL.glGetUniformLocation(self.program, "u_Roughness")
        self.u_metallic = GL.glGetUniformLocation(self.program, "u_Metallic")
        self.u_reflectance = GL.glGetUniformLocation(self.program, "u_Reflectance")
        self.u_ao = GL.glGetUniformLocation(self.program, "u_AO")

        self.u_albedo_map = GL.glGetUniformLocation(self.program, "u_AlbedoMap")
        self.u_normal_map = GL.glGetUniformLocation(self.program, "u_NormalMap")
        self.u_roughness_map = GL.glGetUniformLocation(self.program, "u_RoughnessMap")
        self.u_metallic_map = GL.glGetUniformLocation(self.program, "u_MetallicMap")

        self.u_use_albedo_map = GL.glGetUniformLocation(self.program, "u_UseAlbedoMap")
        self.u_use_normal_map = GL.glGetUniformLocation(self.program, "u_UseNormalMap")
        self.u_use_roughness_map = GL.glGetUniformLocation(self.program, "u_UseRoughnessMap")
        self.u_use_metallic_map = GL.glGetUniformLocation(self.program, "u_UseMetallicMap")

        GL.glUseProgram(self.program)
        GL.glUniform1i(self.u_albedo_map, 0)
        GL.glUniform1i(self.u_normal_map, 1)
        GL.glUniform1i(self.u_roughness_map, 2)
        GL.glUniform1i(self.u_metallic_map, 3)

    def set_material(self, material: Material, default_texture_id: int):
        GL.glUniform3fv(self.u_albedo, 1, material.albedo)
        GL.glUniform1f(self.u_roughness, float(material.roughness))
        GL.glUniform1f(self.u_metallic, float(material.metallic))
        GL.glUniform1f(self.u_reflectance, float(material.reflectance))
        GL.glUniform1f(self.u_ao, float(material.ao))

        self._bind_and_update(material.albedo_map,    self.u_use_albedo_map,    0, default_texture_id)
        self._bind_and_update(material.normal_map,    self.u_use_normal_map,    1, default_texture_id)
        self._bind_and_update(material.roughness_map, self.u_use_roughness_map, 2, default_texture_id)
        self._bind_and_update(material.metallic_map,  self.u_use_metallic_map,  3, default_texture_id)

    def _bind_and_update(self, tex_asset, flag_loc, unit, default_id):
        GL.glActiveTexture(GL.GL_TEXTURE0 + unit)

        if tex_asset and tex_asset.status == AssetStatus.Ready and tex_asset.gl_id:
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex_asset.gl_id)
            GL.glUniform1i(flag_loc, 1)
        else:
            GL.glBindTexture(GL.GL_TEXTURE_2D, default_id)
            GL.glUniform1i(flag_loc, 0)


def make_shader() -> GGXHammonShader:
    p = Path(__file__).parent.absolute()
    return GGXHammonShader(
        read_source_file(p / "tf2_ggx_hammon/vert.glsl"),
        read_source_file(p / "tf2_ggx_hammon/frag.glsl")
    )
