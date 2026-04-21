from imgui_bundle import imgui, icons_fontawesome_6

from entities.components.camera import Camera
from entities.components.directional_light import DirectionalLight
from entities.components.point_light import PointLight
from entities.components.transform import Transform
from entities.components.ui.ui_state import UiState
from entities.components.visuals.visuals import Visuals
from entities.components.entity_flags import EntityFlags
from entities.registry import Registry


def draw_entity_list_section(
    registry: Registry,
    ui_state: UiState,
    selected_entity: int | None,
):
    if imgui.collapsing_header("Entities", imgui.TreeNodeFlags_.default_open):
        content_max_x = imgui.get_cursor_pos().x + imgui.get_content_region_avail().x
        ICON_PRIORITY = {
            Camera: icons_fontawesome_6.ICON_FA_CAMERA,
            DirectionalLight: icons_fontawesome_6.ICON_FA_SUN,
            PointLight: icons_fontawesome_6.ICON_FA_LIGHTBULB,
            Visuals: icons_fontawesome_6.ICON_FA_CUBE,
            Transform: icons_fontawesome_6.ICON_FA_ARROWS_UP_DOWN_LEFT_RIGHT,
        }
        FIXED_ICON_WIDTH = 26

        for entity_id, components in registry.view_all():
            entity_flags: EntityFlags | None = components.get(EntityFlags)
            if entity_flags is not None and entity_flags.is_internal:
                continue

            display_name = entity_flags.name if (
                entity_flags and entity_flags.name) else "Entity"

            display_icon = None
            for comp_type, icon in ICON_PRIORITY.items():
                if comp_type in components:
                    display_icon = icon
                    break

            start_x = imgui.get_cursor_pos_x()

            if display_icon:
                icon_width = imgui.calc_text_size(display_icon).x
                center_offset = (FIXED_ICON_WIDTH - icon_width) / 2
                imgui.set_cursor_pos_x(start_x + center_offset)
                imgui.text_disabled(display_icon)
                imgui.same_line(start_x + FIXED_ICON_WIDTH)
            else:
                imgui.set_cursor_pos_x(start_x + FIXED_ICON_WIDTH)

            id_str = f"#{entity_id}"
            selectable_label = f"{display_name}###entity_{entity_id}"

            clicked, _ = imgui.selectable(selectable_label, selected_entity == entity_id)
            if clicked:
                registry.set_parent(ui_state.selection_child_entity, entity_id)

            id_width = imgui.calc_text_size(id_str).x
            align_x = content_max_x - id_width

            imgui.same_line(int(align_x))
            imgui.text_disabled(id_str)
