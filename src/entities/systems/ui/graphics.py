from imgui_bundle import imgui

from entities.components.render_state import GlobalDrawMode, RenderState
from entities.components.ui.icon_render_state import IconRenderState


def draw_graphics_section(
    render_state: RenderState,
    icon_render_state: IconRenderState,
):
    if imgui.collapsing_header("Graphics", imgui.TreeNodeFlags_.default_open):
        if imgui.button("Capture this frame"):
            render_state.is_capture = True

        if imgui.begin_table("graphics_render_mode", 2):
            imgui.table_setup_column("Label", imgui.TableColumnFlags_.width_fixed)
            imgui.table_setup_column("Controls", imgui.TableColumnFlags_.width_stretch)

            imgui.table_next_row()

            imgui.table_next_column()
            imgui.align_text_to_frame_padding()
            imgui.text("Draw")

            imgui.table_next_column()
            if imgui.radio_button("normally", render_state.global_draw_mode == GlobalDrawMode.Normal):
                render_state.global_draw_mode = GlobalDrawMode.Normal
            imgui.same_line()
            if imgui.radio_button("wireframe", render_state.global_draw_mode == GlobalDrawMode.Wireframe):
                render_state.global_draw_mode = GlobalDrawMode.Wireframe
            imgui.same_line()
            if imgui.radio_button("depth", render_state.global_draw_mode == GlobalDrawMode.DepthOnly):
                render_state.global_draw_mode = GlobalDrawMode.DepthOnly

            imgui.end_table()

        imgui.push_item_width(imgui.get_window_width() * 0.5)
        changed_li, new_li = imgui.checkbox("Light icons", icon_render_state.draw_light_icons)
        if changed_li:
            icon_render_state.draw_light_icons = new_li
        imgui.same_line()
        changed_ci, new_ci = imgui.checkbox("Camera icons", icon_render_state.draw_camera_icons)
        if changed_ci:
            icon_render_state.draw_camera_icons = new_ci
        imgui.same_line()
        changed_bb, new_bb = imgui.checkbox("Show bounding boxes", render_state.show_bounding_boxes)
        if changed_bb:
            render_state.show_bounding_boxes = new_bb

        imgui.pop_item_width()
