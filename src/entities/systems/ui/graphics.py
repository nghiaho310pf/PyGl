from imgui_bundle import imgui
from engine.application import Application
from entities.components.render_state import GlobalDrawMode, RenderState
from entities.components.ui.icon_render_state import IconRenderState
from math_utils import float1


def draw_graphics_section(
    render_state: RenderState,
    icon_render_state: IconRenderState,
):
    if imgui.collapsing_header("Graphics", imgui.TreeNodeFlags_.default_open):
        if imgui.button("Capture this frame"):
            render_state.is_capture = True

        if not Application.has_broken_opengl:
            imgui.same_line()
            imgui.text(f"FPS: {render_state.fps:.1f} ({render_state.render_time_ms:.2f}ms, max: {render_state.theoretical_max_fps:.1f})")

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

        if imgui.tree_node_ex("Shadow blurring"):
            changed_depth_sensitivity, new_depth_sensitivity = imgui.drag_float(
                "Depth sensitivity", float(render_state.shadow_blur_depth_sensitivity), 0.01, 0.0, 200.0)
            if changed_depth_sensitivity:
                render_state.shadow_blur_depth_sensitivity = float1(new_depth_sensitivity)
            changed_norm_thres, new_norm_thres = imgui.drag_float(
                "Normal threshold", float(render_state.shadow_blur_normal_threshold), 0.001, 0.0, 1.0)
            if changed_norm_thres:
                render_state.shadow_blur_normal_threshold = float1(new_norm_thres)

            imgui.tree_pop()

        for name, preset in [
            ("Viewport settings", render_state.viewport_graphics_settings),
            ("Capture settings", render_state.capture_graphics_settings)
        ]:
            if imgui.tree_node_ex(name):
                changed_smaa, new_smaa = imgui.checkbox("SMAA", preset.enable_smaa)
                if changed_smaa:
                    preset.enable_smaa = new_smaa

                imgui.text_disabled("Shadows")

                changed_p_pcf, new_p_pcf = imgui.drag_int(
                    "Point samples", preset.point_shadow_samples, 1, 1, 128)
                if changed_p_pcf:
                    preset.point_shadow_samples = new_p_pcf

                changed_d_pcf, new_d_pcf = imgui.drag_int(
                    "Directional samples", preset.directional_shadow_samples, 1, 1, 128)
                if changed_d_pcf:
                    preset.directional_shadow_samples = new_d_pcf

                imgui.tree_pop()

        imgui.pop_item_width()
