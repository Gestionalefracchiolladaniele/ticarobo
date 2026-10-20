"""Script eseguito DENTRO Blender: converte e semplifica una mesh.

Non importarlo dal progetto: usa ``bpy``, che esiste solo dentro Blender.
Si invoca tramite ``robotica.assets.blender.convert_mesh``.

    blender --background --python _convert_mesh.py -- <src> <dest> <max_faces>
"""

import sys

import bpy


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(argv) < 2:
        raise SystemExit("uso: -- <sorgente> <destinazione> [max_faces]")
    src, dest = argv[0], argv[1]
    max_faces = int(argv[2]) if len(argv) > 2 else 0

    bpy.ops.wm.read_factory_settings(use_empty=True)
    _import_any(src)

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit(f"nessuna mesh trovata in {src}")

    if max_faces > 0:
        for obj in meshes:
            _decimate(obj, max_faces)

    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]

    _export_any(dest)
    total = sum(len(o.data.polygons) for o in meshes)
    print(f"OK: {len(meshes)} mesh, {total} facce -> {dest}")


def _import_any(path: str) -> None:
    """Importa in base all'estensione."""
    ext = path.lower().rsplit(".", 1)[-1]
    importers = {
        "obj": bpy.ops.wm.obj_import,
        "stl": bpy.ops.wm.stl_import,
        "fbx": bpy.ops.import_scene.fbx,
        "dae": bpy.ops.wm.collada_import,
        "ply": bpy.ops.wm.ply_import,
        "glb": bpy.ops.import_scene.gltf,
        "gltf": bpy.ops.import_scene.gltf,
    }
    if ext not in importers:
        raise SystemExit(f"formato non supportato in ingresso: .{ext}")
    importers[ext](filepath=path)


def _export_any(path: str) -> None:
    """Esporta in base all'estensione, solo la selezione."""
    ext = path.lower().rsplit(".", 1)[-1]
    if ext == "stl":
        bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True)
    elif ext == "obj":
        bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True)
    elif ext == "ply":
        bpy.ops.wm.ply_export(filepath=path, export_selected_objects=True)
    else:
        raise SystemExit(f"formato non supportato in uscita: .{ext}")


def _decimate(obj: bpy.types.Object, max_faces: int) -> None:
    """Riduce il numero di facce sotto la soglia.

    Una mesh CAD ha spesso 100k+ triangoli; per la collisione in MuJoCo ne
    bastano poche migliaia, e la simulazione va molto piu' veloce.
    """
    current = len(obj.data.polygons)
    if current <= max_faces:
        return
    mod = obj.modifiers.new(name="decimate", type="DECIMATE")
    mod.ratio = max_faces / current
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)


if __name__ == "__main__":
    main()
