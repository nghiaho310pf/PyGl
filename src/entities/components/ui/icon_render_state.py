from dataclasses import dataclass


@dataclass(slots=True, eq=False)
class IconRenderState:
    draw_light_icons: bool = True
    draw_camera_icons: bool = True
