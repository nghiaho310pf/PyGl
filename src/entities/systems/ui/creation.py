import copy
import dataclasses
from imgui_bundle import imgui, portable_file_dialogs as pfd

from entities.components.camera import Camera
from entities.components.camera_state import CameraState
from entities.components.directional_light import DirectionalLight
from entities.components.point_light import PointLight
from entities.components.spawner_state import SpawnerState
from entities.components.surface_function import SurfaceFunction
from entities.components.gd.surface import GradientDescentSurface
from entities.components.transform import Transform
from entities.components.ui.ui_state import AddType, UiState
from entities.components.visuals.visuals import Visuals
from entities.components.entity_flags import EntityFlags
from entities.components.visuals.assets import AssetsState
from entities.systems.assets import AssetSystem
from entities.registry import Registry
from entities.systems.spawner import SpawnerSystem
from meshes.surfaces.arrow import generate_arrow
from meshes.surfaces.circle import generate_circle
from meshes.surfaces.ellipse import generate_ellipse
from meshes.surfaces.hexagon import generate_hexagon
from meshes.surfaces.pentagon import generate_pentagon
from meshes.surfaces.plane import generate_plane
from meshes.surfaces.star import generate_star
from meshes.surfaces.trapezoid import generate_trapezoid
from meshes.surfaces.triangle import generate_triangle
from meshes.volumes.cone import generate_cone
from meshes.volumes.cube import generate_cube
from meshes.volumes.cylinder import generate_cylinder
from meshes.volumes.prism import generate_prism
from meshes.volumes.subdivided_spheres import generate_icosphere, generate_tetrasphere
from meshes.volumes.tetrahedron import generate_tetrahedron
from meshes.volumes.torus import generate_torus
from meshes.volumes.uv_sphere import generate_uv_sphere
from math_utils import vec3, quaternion_identity


def draw_creation_section(
    registry: Registry,
    ui_state: UiState,
    assets_state: AssetsState,
    spawner_state: SpawnerState,
    camera_state: CameraState,
    preview_transform: Transform,
    preview_visuals: Visuals,
):
    if ui_state.should_close_add_menu:
        imgui.set_next_item_open(False)
        ui_state.should_close_add_menu = False
    add_menu_expanded = imgui.collapsing_header("+ Add")

    if not add_menu_expanded:
        preview_visuals.enabled = False
    else:
        if imgui.button("Load model..."):
            dialog = pfd.open_file(
                title=f"Select model file",
                default_path="",
                filters=["3D Model Files", "*.glb *.gltf *.obj *.ply", "All Files", "*"]
            )
            result = dialog.result()

            if result and len(result) > 0:
                filepath = result[0]
                model_asset = AssetSystem.request_model(assets_state, filepath)
                SpawnerSystem.load_and_spawn_one(
                    spawner_state, model_asset,
                    Transform(scale=vec3(0.04, 0.04, 0.04)),
                )
                ui_state.should_close_add_menu = True

        preview_transform.local.position = vec3(*camera_state.focal_point)
        preview_transform.local.rotation = quaternion_identity()
        preview_transform.local.scale = vec3(1.0, 1.0, 1.0)

        changed_type = False

        if imgui.radio_button("Triangle", ui_state.add_mesh_type == AddType.Triangle):
            ui_state.add_mesh_type = AddType.Triangle
            changed_type = True
        imgui.same_line()
        if imgui.radio_button("Plane", ui_state.add_mesh_type == AddType.Plane):
            ui_state.add_mesh_type = AddType.Plane
            changed_type = True
        imgui.same_line()
        if imgui.radio_button("Pentagon", ui_state.add_mesh_type == AddType.Pentagon):
            ui_state.add_mesh_type = AddType.Pentagon
            changed_type = True
        imgui.same_line()
        if imgui.radio_button("Hexagon", ui_state.add_mesh_type == AddType.Hexagon):
            ui_state.add_mesh_type = AddType.Hexagon
            changed_type = True
        if imgui.radio_button("Circle", ui_state.add_mesh_type == AddType.Circle):
            ui_state.add_mesh_type = AddType.Circle
            changed_type = True
        imgui.same_line()
        if imgui.radio_button("Ellipse", ui_state.add_mesh_type == AddType.Ellipse):
            ui_state.add_mesh_type = AddType.Ellipse
            changed_type = True
        if imgui.radio_button("Trapezoid", ui_state.add_mesh_type == AddType.Trapezoid):
            ui_state.add_mesh_type = AddType.Trapezoid
            changed_type = True
        imgui.same_line()
        if imgui.radio_button("Star", ui_state.add_mesh_type == AddType.Star):
            ui_state.add_mesh_type = AddType.Star
            changed_type = True
        imgui.same_line()
        if imgui.radio_button("Arrow", ui_state.add_mesh_type == AddType.Arrow):
            ui_state.add_mesh_type = AddType.Arrow
            changed_type = True

        if imgui.radio_button("Cube", ui_state.add_mesh_type == AddType.Cube):
            ui_state.add_mesh_type = AddType.Cube
            changed_type = True
        imgui.same_line()
        if imgui.radio_button("Tetrahedron", ui_state.add_mesh_type == AddType.Tetrahedron):
            ui_state.add_mesh_type = AddType.Tetrahedron
            changed_type = True

        if imgui.radio_button("Function surface", ui_state.add_mesh_type == AddType.FunctionSurface):
            ui_state.add_mesh_type = AddType.FunctionSurface
            changed_type = True
        imgui.same_line()
        if imgui.radio_button("Gradient descent surface", ui_state.add_mesh_type == AddType.GradientDescentSurface):
            ui_state.add_mesh_type = AddType.GradientDescentSurface
            changed_type = True

        if imgui.radio_button("Prism", ui_state.add_mesh_type == AddType.Prism):
            ui_state.add_mesh_type = AddType.Prism
            changed_type = True
        imgui.same_line()
        if imgui.radio_button("Cone", ui_state.add_mesh_type == AddType.Cone):
            ui_state.add_mesh_type = AddType.Cone
            changed_type = True
        imgui.same_line()
        if imgui.radio_button("Cylinder", ui_state.add_mesh_type == AddType.Cylinder):
            ui_state.add_mesh_type = AddType.Cylinder
            changed_type = True

        if imgui.radio_button("UV sphere", ui_state.add_mesh_type == AddType.UVSphere):
            ui_state.add_mesh_type = AddType.UVSphere
            changed_type = True
        imgui.same_line()
        if imgui.radio_button("Tetrasphere", ui_state.add_mesh_type == AddType.Tetrasphere):
            ui_state.add_mesh_type = AddType.Tetrasphere
            changed_type = True
        imgui.same_line()
        if imgui.radio_button("Icosphere", ui_state.add_mesh_type == AddType.Icosphere):
            ui_state.add_mesh_type = AddType.Icosphere
            changed_type = True

        if imgui.radio_button("Torus", ui_state.add_mesh_type == AddType.Torus):
            ui_state.add_mesh_type = AddType.Torus
            changed_type = True

        if imgui.radio_button("Directional light", ui_state.add_mesh_type == AddType.DirectionalLight):
            ui_state.add_mesh_type = AddType.DirectionalLight
            changed_type = True
        imgui.same_line()
        if imgui.radio_button("Point light", ui_state.add_mesh_type == AddType.PointLight):
            ui_state.add_mesh_type = AddType.PointLight
            changed_type = True
        imgui.same_line()
        if imgui.radio_button("Camera", ui_state.add_mesh_type == AddType.Camera):
            ui_state.add_mesh_type = AddType.Camera
            changed_type = True

        mesh_changed = not ui_state.preview_visual_initialized or changed_type

        if ui_state.add_mesh_type == AddType.Triangle:
            changed_s, ui_state.general_mesh_size = imgui.drag_float("Size", ui_state.general_mesh_size, 0.01, 0.01, 10.0)
            mesh_changed = mesh_changed or changed_s
        elif ui_state.add_mesh_type == AddType.Plane:
            changed_s, ui_state.general_mesh_size = imgui.drag_float("Size", ui_state.general_mesh_size, 0.01, 0.01, 10.0)
            mesh_changed = mesh_changed or changed_s
        elif ui_state.add_mesh_type == AddType.Pentagon:
            changed_s, ui_state.general_mesh_size = imgui.drag_float("Size", ui_state.general_mesh_size, 0.01, 0.01, 10.0)
            mesh_changed = mesh_changed or changed_s
        elif ui_state.add_mesh_type == AddType.Hexagon:
            changed_s, ui_state.general_mesh_size = imgui.drag_float("Size", ui_state.general_mesh_size, 0.01, 0.01, 10.0)
            mesh_changed = mesh_changed or changed_s
        elif ui_state.add_mesh_type == AddType.Circle:
            changed_s, ui_state.general_mesh_size = imgui.drag_float("Size", ui_state.general_mesh_size, 0.01, 0.01, 10.0)
            changed_se, ui_state.round_surface_sides = imgui.slider_int("Sides", ui_state.round_surface_sides, 3, 50)
            mesh_changed = mesh_changed or changed_s or changed_se
        elif ui_state.add_mesh_type == AddType.Ellipse:
            changed_rx, ui_state.ellipse_radius_x = imgui.drag_float("X radius", ui_state.ellipse_radius_x, 0.01, 0.01, 10.0)
            changed_rz, ui_state.ellipse_radius_z = imgui.drag_float("Z radius", ui_state.ellipse_radius_z, 0.01, 0.01, 10.0)
            changed_se, ui_state.round_surface_sides = imgui.slider_int("Sides", ui_state.round_surface_sides, 3, 50)
            mesh_changed = mesh_changed or changed_rx or changed_rz or changed_se
        elif ui_state.add_mesh_type == AddType.Trapezoid:
            changed_tw, ui_state.trapezoid_top_width = imgui.drag_float("Top width", ui_state.trapezoid_top_width, 0.01, 0.01, 10.0)
            changed_bw, ui_state.trapezoid_bottom_width = imgui.drag_float("Bottom width", ui_state.trapezoid_bottom_width, 0.01, 0.01, 10.0)
            changed_h, ui_state.trapezoid_height = imgui.drag_float("Height", ui_state.trapezoid_height, 0.01, 0.01, 10.0)
            mesh_changed = mesh_changed or changed_tw or changed_bw or changed_h
        elif ui_state.add_mesh_type == AddType.Star:
            changed_ir, ui_state.star_inner_radius = imgui.drag_float("Inner radius", ui_state.star_inner_radius, 0.01, 0.01, 10.0)
            changed_or, ui_state.star_outer_radius = imgui.drag_float("Outer radius", ui_state.star_outer_radius, 0.01, 0.01, 10.0)
            changed_p, ui_state.star_points = imgui.slider_int("Points", ui_state.star_points, 2, 20)
            mesh_changed = mesh_changed or changed_ir or changed_or or changed_p
        elif ui_state.add_mesh_type == AddType.Arrow:
            changed_s, ui_state.general_mesh_size = imgui.drag_float("Size", ui_state.general_mesh_size, 0.01, 0.01, 10.0)
            changed_tl, ui_state.arrow_tail_length = imgui.drag_float("Tail length", ui_state.arrow_tail_length, 0.01, 0.01, 10.0)
            mesh_changed = mesh_changed or changed_s or changed_tl
        elif ui_state.add_mesh_type == AddType.Cube:
            changed_s, ui_state.general_mesh_size = imgui.drag_float("Size", ui_state.general_mesh_size, 0.01, 0.01, 10.0)
            mesh_changed = mesh_changed or changed_s
        elif ui_state.add_mesh_type == AddType.Tetrahedron:
            changed_s, ui_state.general_mesh_size = imgui.drag_float("Size", ui_state.general_mesh_size, 0.01, 0.01, 10.0)
            mesh_changed = mesh_changed or changed_s
        elif ui_state.add_mesh_type == AddType.Prism:
            changed_rb, ui_state.column_radius_bottom = imgui.drag_float("Radius", ui_state.column_radius_bottom, 0.01, 0.01, 10.0)
            changed_h, ui_state.column_height = imgui.drag_float("Height", ui_state.column_height, 0.01, 0.01, 10.0)
            changed_se, ui_state.column_sectors = imgui.slider_int("Sectors", ui_state.column_sectors, 3, 50)
            mesh_changed = mesh_changed or changed_rb or changed_h or changed_se
        elif ui_state.add_mesh_type == AddType.Cone:
            changed_rb, ui_state.column_radius_bottom = imgui.drag_float("Bottom radius", ui_state.column_radius_bottom, 0.01, 0.01, 10.0)
            changed_h, ui_state.column_height = imgui.drag_float("Height", ui_state.column_height, 0.01, 0.01, 10.0)
            changed_se, ui_state.column_sectors = imgui.slider_int("Sectors", ui_state.column_sectors, 3, 50)
            mesh_changed = mesh_changed or changed_rb or changed_h or changed_se
        elif ui_state.add_mesh_type == AddType.Cylinder:
            changed_rb, ui_state.column_radius_bottom = imgui.drag_float("Bottom radius", ui_state.column_radius_bottom, 0.01, 0.01, 10.0)
            changed_rt, ui_state.cylinder_radius_top = imgui.drag_float("Top radius", ui_state.cylinder_radius_top, 0.01, 0.01, 10.0)
            changed_h, ui_state.column_height = imgui.drag_float("Height", ui_state.column_height, 0.01, 0.01, 10.0)
            changed_se, ui_state.column_sectors = imgui.slider_int("Sectors", ui_state.column_sectors, 3, 50)
            mesh_changed = mesh_changed or changed_rb or changed_rt or changed_h or changed_se
        elif ui_state.add_mesh_type == AddType.UVSphere:
            changed_r, ui_state.sphere_radius = imgui.drag_float("Radius", ui_state.sphere_radius, 0.01, 0.01, 10.0)
            changed_st, ui_state.uv_sphere_stacks = imgui.slider_int("Stacks", ui_state.uv_sphere_stacks, 3, 50)
            changed_se, ui_state.uv_sphere_sectors = imgui.slider_int("Sectors", ui_state.uv_sphere_sectors, 3, 50)
            mesh_changed = mesh_changed or changed_r or changed_st or changed_se
        elif ui_state.add_mesh_type == AddType.Tetrasphere:
            changed_r, ui_state.sphere_radius = imgui.drag_float("Radius", ui_state.sphere_radius, 0.01, 0.01, 10.0)
            changed_sss, ui_state.subdiv_sphere_subdivisions = imgui.slider_int("Subdivisions", ui_state.subdiv_sphere_subdivisions, 1, 6)
            mesh_changed = mesh_changed or changed_r or changed_sss
        elif ui_state.add_mesh_type == AddType.Icosphere:
            changed_r, ui_state.sphere_radius = imgui.drag_float("Radius", ui_state.sphere_radius, 0.01, 0.01, 10.0)
            changed_sss, ui_state.subdiv_sphere_subdivisions = imgui.slider_int("Subdivisions", ui_state.subdiv_sphere_subdivisions, 1, 6)
            mesh_changed = mesh_changed or changed_r or changed_sss
        elif ui_state.add_mesh_type == AddType.Torus:
            changed_mr, ui_state.torus_main_radius = imgui.drag_float("Main radius", ui_state.torus_main_radius, 0.01, 0.01, 10.0)
            changed_tr, ui_state.torus_tube_radius = imgui.drag_float("Tube radius", ui_state.torus_tube_radius, 0.01, 0.01, 10.0)
            changed_mse, ui_state.torus_main_sectors = imgui.slider_int("Main sectors", ui_state.torus_main_sectors, 3, 50)
            changed_tse, ui_state.torus_tube_sectors = imgui.slider_int("Tube sectors", ui_state.torus_tube_sectors, 3, 50)
            mesh_changed = mesh_changed or changed_mr or changed_tr or changed_mse or changed_tse

        if ui_state.add_mesh_type in (AddType.DirectionalLight, AddType.PointLight, AddType.Camera, AddType.FunctionSurface, AddType.GradientDescentSurface):
            preview_visuals.enabled = False
        else:
            preview_visuals.enabled = True
            if mesh_changed:
                vi = None
                if ui_state.add_mesh_type == AddType.Triangle:
                    vi = generate_triangle(ui_state.general_mesh_size)
                elif ui_state.add_mesh_type == AddType.Plane:
                    vi = generate_plane(ui_state.general_mesh_size)
                elif ui_state.add_mesh_type == AddType.Pentagon:
                    vi = generate_pentagon(ui_state.general_mesh_size)
                elif ui_state.add_mesh_type == AddType.Hexagon:
                    vi = generate_hexagon(ui_state.general_mesh_size)
                elif ui_state.add_mesh_type == AddType.Circle:
                    vi = generate_circle(ui_state.general_mesh_size, ui_state.round_surface_sides)
                elif ui_state.add_mesh_type == AddType.Ellipse:
                    vi = generate_ellipse(ui_state.ellipse_radius_x, ui_state.ellipse_radius_z, ui_state.round_surface_sides)
                elif ui_state.add_mesh_type == AddType.Trapezoid:
                    vi = generate_trapezoid(ui_state.trapezoid_top_width, ui_state.trapezoid_bottom_width, ui_state.trapezoid_height)
                elif ui_state.add_mesh_type == AddType.Star:
                    vi = generate_star(ui_state.star_outer_radius, ui_state.star_inner_radius, ui_state.star_points)
                elif ui_state.add_mesh_type == AddType.Arrow:
                    vi = generate_arrow(ui_state.general_mesh_size, ui_state.arrow_tail_length)
                elif ui_state.add_mesh_type == AddType.Cube:
                    vi = generate_cube(ui_state.general_mesh_size)
                elif ui_state.add_mesh_type == AddType.Tetrahedron:
                    vi = generate_tetrahedron(ui_state.general_mesh_size)
                elif ui_state.add_mesh_type == AddType.Prism:
                    vi = generate_prism(ui_state.column_sectors, ui_state.column_radius_bottom, ui_state.column_height)
                elif ui_state.add_mesh_type == AddType.Cone:
                    vi = generate_cone(ui_state.column_radius_bottom, ui_state.column_height, ui_state.column_sectors)
                elif ui_state.add_mesh_type == AddType.Cylinder:
                    vi = generate_cylinder(ui_state.column_radius_bottom, ui_state.cylinder_radius_top, ui_state.column_height, ui_state.column_sectors)
                elif ui_state.add_mesh_type == AddType.UVSphere:
                    vi = generate_uv_sphere(ui_state.sphere_radius, ui_state.uv_sphere_stacks, ui_state.uv_sphere_sectors)
                elif ui_state.add_mesh_type == AddType.Tetrasphere:
                    vi = generate_tetrasphere(ui_state.sphere_radius, ui_state.subdiv_sphere_subdivisions)
                elif ui_state.add_mesh_type == AddType.Icosphere:
                    vi = generate_icosphere(ui_state.sphere_radius, ui_state.subdiv_sphere_subdivisions)
                elif ui_state.add_mesh_type == AddType.Torus:
                    vi = generate_torus(ui_state.torus_main_radius, ui_state.torus_tube_radius, ui_state.torus_main_sectors, ui_state.torus_tube_sectors)

                if vi is not None:
                    preview_visuals.mesh = AssetSystem.create_immediate_mesh(assets_state, *vi)
                ui_state.preview_visual_initialized = True

        imgui.separator()
        if imgui.button("Add to scene"):
            new_entity = registry.create_entity()

            if ui_state.add_mesh_type in (
                AddType.Triangle, AddType.Plane, AddType.Pentagon, AddType.Hexagon,
                AddType.Circle, AddType.Ellipse, AddType.Trapezoid, AddType.Star, AddType.Arrow,
                AddType.Cube, AddType.Tetrahedron,
                AddType.Prism, AddType.Cone, AddType.Cylinder,
                AddType.UVSphere, AddType.Tetrasphere, AddType.Icosphere, AddType.Torus
            ):
                vi = None
                if ui_state.add_mesh_type == AddType.Triangle:
                    vi = generate_triangle(ui_state.general_mesh_size)
                elif ui_state.add_mesh_type == AddType.Plane:
                    vi = generate_plane(ui_state.general_mesh_size)
                elif ui_state.add_mesh_type == AddType.Pentagon:
                    vi = generate_pentagon(ui_state.general_mesh_size)
                elif ui_state.add_mesh_type == AddType.Hexagon:
                    vi = generate_hexagon(ui_state.general_mesh_size)
                elif ui_state.add_mesh_type == AddType.Circle:
                    vi = generate_circle(ui_state.general_mesh_size, ui_state.round_surface_sides)
                elif ui_state.add_mesh_type == AddType.Ellipse:
                    vi = generate_ellipse(ui_state.ellipse_radius_x, ui_state.ellipse_radius_z, ui_state.round_surface_sides)
                elif ui_state.add_mesh_type == AddType.Trapezoid:
                    vi = generate_trapezoid(ui_state.trapezoid_top_width, ui_state.trapezoid_bottom_width, ui_state.trapezoid_height)
                elif ui_state.add_mesh_type == AddType.Star:
                    vi = generate_star(ui_state.star_outer_radius, ui_state.star_inner_radius, ui_state.star_points)
                elif ui_state.add_mesh_type == AddType.Arrow:
                    vi = generate_arrow(ui_state.general_mesh_size, ui_state.arrow_tail_length)
                elif ui_state.add_mesh_type == AddType.Cube:
                    vi = generate_cube(ui_state.general_mesh_size)
                elif ui_state.add_mesh_type == AddType.Tetrahedron:
                    vi = generate_tetrahedron(ui_state.general_mesh_size)
                elif ui_state.add_mesh_type == AddType.Prism:
                    vi = generate_prism(ui_state.column_sectors, ui_state.column_radius_bottom, ui_state.column_height)
                elif ui_state.add_mesh_type == AddType.Cone:
                    vi = generate_cone(ui_state.column_radius_bottom, ui_state.column_height, ui_state.column_sectors)
                elif ui_state.add_mesh_type == AddType.Cylinder:
                    vi = generate_cylinder(ui_state.column_radius_bottom, ui_state.cylinder_radius_top, ui_state.column_height, ui_state.column_sectors)
                elif ui_state.add_mesh_type == AddType.UVSphere:
                    vi = generate_uv_sphere(ui_state.sphere_radius, ui_state.uv_sphere_stacks, ui_state.uv_sphere_sectors)
                elif ui_state.add_mesh_type == AddType.Tetrasphere:
                    vi = generate_tetrasphere(ui_state.sphere_radius, ui_state.subdiv_sphere_subdivisions)
                elif ui_state.add_mesh_type == AddType.Icosphere:
                    vi = generate_icosphere(ui_state.sphere_radius, ui_state.subdiv_sphere_subdivisions)
                elif ui_state.add_mesh_type == AddType.Torus:
                    vi = generate_torus(ui_state.torus_main_radius, ui_state.torus_tube_radius, ui_state.torus_main_sectors, ui_state.torus_tube_sectors)

                if vi is not None:
                    new_material = copy.copy(ui_state.default_material)

                    registry.add_components(
                        new_entity,
                        EntityFlags(name=f"{ui_state.add_mesh_type.name}"),
                        dataclasses.replace(preview_transform),
                        Visuals(AssetSystem.create_immediate_mesh(assets_state, *vi), new_material)
                    )
            elif ui_state.add_mesh_type == AddType.FunctionSurface:
                new_material = copy.copy(ui_state.default_material)
                vi = generate_plane(10.0)

                registry.add_components(
                    new_entity,
                    EntityFlags(name="Function surface"),
                    Transform(position=vec3(*camera_state.focal_point)),
                    Visuals(AssetSystem.create_immediate_mesh(assets_state, *vi), new_material, cull_back_faces=False),
                    SurfaceFunction()
                )
            elif ui_state.add_mesh_type == AddType.GradientDescentSurface:
                new_material = copy.copy(ui_state.default_material)
                vi = generate_plane(10.0)

                registry.add_components(
                    new_entity,
                    EntityFlags(name="Gradient descent surface"),

                    Transform(position=vec3(0.0, 0.0, 0.0), scale=vec3(1.0, 1.0, 1.0)),
                    GradientDescentSurface(),
                    Visuals(AssetSystem.create_immediate_mesh(assets_state, *vi), new_material, cull_back_faces=False),
                )
            elif ui_state.add_mesh_type == AddType.DirectionalLight:
                registry.add_components(
                    new_entity,
                    EntityFlags(name="Directional light"),
                    DirectionalLight()
                )
            elif ui_state.add_mesh_type == AddType.PointLight:
                registry.add_components(
                    new_entity,
                    EntityFlags(name="Point light"),
                    Transform(position=vec3(*camera_state.focal_point)),
                    PointLight()
                )
            elif ui_state.add_mesh_type == AddType.Camera:
                registry.add_components(
                    new_entity,
                    EntityFlags(name="Camera"),
                    Transform(position=vec3(*camera_state.focal_point)),
                    Camera()
                )

            registry.set_parent(ui_state.selection_child_entity, new_entity)
            ui_state.should_close_add_menu = True
