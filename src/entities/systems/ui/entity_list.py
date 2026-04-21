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

        entity_info = {}
        for entity_id, components in registry.view_all():
            entity_flags: EntityFlags | None = components.get(EntityFlags)
            if entity_flags is not None and entity_flags.is_internal:
                continue

            display_name = entity_flags.name if (entity_flags and entity_flags.name) else "Entity"
            display_icon = None
            for comp_type, icon in ICON_PRIORITY.items():
                if comp_type in components:
                    display_icon = icon
                    break
            entity_info[entity_id] = (display_name, display_icon)

        def render_entity_node(entity_id: int):
            display_name, display_icon = entity_info[entity_id]
            children = registry.get_children(entity_id)
            visible_children = sorted([c for c in children if c in entity_info])

            is_leaf = len(visible_children) == 0

            flags = (
                imgui.TreeNodeFlags_.open_on_arrow |
                imgui.TreeNodeFlags_.open_on_double_click |
                imgui.TreeNodeFlags_.span_full_width |
                imgui.TreeNodeFlags_.default_open
            )
            if is_leaf:
                flags |= imgui.TreeNodeFlags_.leaf
            if selected_entity == entity_id:
                flags |= imgui.TreeNodeFlags_.selected

            node_open = imgui.tree_node_ex(f"###node_{entity_id}", flags)

            if imgui.is_item_clicked() and not imgui.is_item_toggled_open():
                registry.set_parent(ui_state.selection_child_entity, entity_id)

            imgui.same_line()
            icon_start_x = imgui.get_cursor_pos_x()

            if display_icon:
                icon_width = imgui.calc_text_size(display_icon).x
                center_offset = (FIXED_ICON_WIDTH - icon_width) / 2
                imgui.set_cursor_pos_x(icon_start_x + center_offset)
                imgui.text_disabled(display_icon)
                imgui.same_line(icon_start_x + FIXED_ICON_WIDTH)
            else:
                imgui.set_cursor_pos_x(icon_start_x + FIXED_ICON_WIDTH)

            imgui.text(display_name)

            id_str = f"#{entity_id}"
            id_width = imgui.calc_text_size(id_str).x
            imgui.same_line()
            imgui.set_cursor_pos_x(content_max_x - id_width)
            imgui.text_disabled(id_str)

            if node_open:
                for child_id in visible_children:
                    render_entity_node(child_id)
                imgui.tree_pop()

        roots = []
        for entity_id in entity_info:
            parent = registry.get_parent(entity_id)
            if parent is None or parent not in entity_info:
                roots.append(entity_id)
        roots.sort()

        for root_id in roots:
            render_entity_node(root_id)
