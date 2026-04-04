from queue import Queue
import queue

from OpenGL import GL
import numpy as np
from PIL import Image
import threading

from entities.components.textures_state import Texture, TextureStatus, TexturesState
from entities.registry import Registry


def background_load_task(filepath: str, upload_queue: Queue):
    try:
        with Image.open(filepath) as img:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA")
            format_ext = img.mode
            data = np.array(img)

        upload_queue.put((filepath, data, format_ext))
    except Exception as e:
        upload_queue.put((filepath, e, "ERROR"))


class TextureSystem:
    @staticmethod
    def update(registry: Registry):
        r_textures = registry.get_singleton(TexturesState)
        if r_textures is None:
            return

        _, (textures_state, ) = r_textures

        while True:
            try:
                filepath, data, format_info = textures_state.upload_queue.get_nowait()
            except queue.Empty:
                break

            if filepath not in textures_state.textures:
                continue

            tex_obj = textures_state.textures[filepath]

            if format_info == "ERROR":
                tex_obj.status = TextureStatus.Failed
                print(f"TextureSystem error loading '{filepath}': {data}")
                continue

            tex_id = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)

            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

            gl_format = GL.GL_RGB if format_info == "RGB" else GL.GL_RGBA

            height, width = data.shape[:2]

            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                gl_format,
                width,
                height,
                0,
                gl_format,
                GL.GL_UNSIGNED_BYTE,
                data
            )

            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

            tex_obj.gl_id = tex_id
            tex_obj.width = width
            tex_obj.height = height
            tex_obj.status = TextureStatus.Ready

    @staticmethod
    def request_texture(textures_state: TexturesState, filepath: str) -> Texture:
        if filepath in textures_state.textures:
            return textures_state.textures[filepath]

        tex = Texture(filepath=filepath, status=TextureStatus.Loading)
        textures_state.textures[filepath] = tex

        thread = threading.Thread(
            target=background_load_task, 
            args=(filepath, textures_state.upload_queue),
            daemon=True
        )
        thread.start()

        return tex
