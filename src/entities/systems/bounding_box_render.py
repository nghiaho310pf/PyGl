from imgui_bundle import imgui
from entities.components.render_state import RenderState
from entities.registry import Registry


class BoundingBoxRenderSystem:
    @staticmethod
    def update(registry: Registry):
        r_state = registry.get_singleton(RenderState)
        if r_state is None:
            return
        _, (render_state, ) = r_state

        if not render_state.show_bounding_boxes:
            return

        draw_list = imgui.get_background_draw_list()
        viewport = imgui.get_main_viewport()

        v_pos = viewport.pos
        v_size = viewport.size

        for bbox in render_state.bounding_boxes:
            p_min = (v_pos.x + bbox.min_x * v_size.x, v_pos.y + bbox.min_y * v_size.y)
            p_max = (v_pos.x + bbox.max_x * v_size.x, v_pos.y + bbox.max_y * v_size.y)

            color = imgui.get_color_u32((0.0, 1.0, 1.0, 1.0))
            draw_list.add_rect(imgui.ImVec2(*p_min), imgui.ImVec2(*p_max), color, thickness=2.0)

            label = bbox.classification_name
            draw_list.add_text(imgui.ImVec2(p_min[0], p_min[1] - 20), color, label)
