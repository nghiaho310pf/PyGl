from queue import Queue
import queue

from OpenGL import GL
import cv2
import threading

from entities.components.textures_state import Texture, TextureStatus, TexturesState
from entities.registry import Registry


def background_load_task(filepath: str, upload_queue: Queue):
    try:
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not load {filepath}")

        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            format_ext = "RGB"
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            format_ext = "RGBA"

        upload_queue.put((filepath, image, format_ext))
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
