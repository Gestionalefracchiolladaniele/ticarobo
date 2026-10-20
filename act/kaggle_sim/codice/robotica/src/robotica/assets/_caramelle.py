"""Script eseguito DENTRO Blender: le caramelle e il bicchiere per la scena ACT.

Non importarlo dal progetto: usa ``bpy``. Si invoca tramite
``robotica.assets.blender.crea_caramelle``.

    blender --background --python _caramelle.py -- <misure.json> <cartella>

Le misure arrivano da ``robotica.sim.so101_caramelle`` (metri): la forma che si
vede (qui) e quella che si tocca (le primitive di MuJoCo) vengono dagli stessi
numeri. Ogni oggetto e' centrato nell'origine con gli assi di simmetria su
x, y, z: MuJoCo sposta una mesh sul suo baricentro, e cosi' non c'e' niente
da spostare.
"""

import json
import math
import sys
from pathlib import Path

import bmesh
import bpy


def _unisci(oggetti: list, nome: str):
    for o in bpy.context.scene.objects:
        o.select_set(False)
    for o in oggetti:
        o.select_set(True)
    bpy.context.view_layer.objects.active = oggetti[0]
    if len(oggetti) > 1:
        bpy.ops.object.join()
    o = bpy.context.view_layer.objects.active
    o.name = nome
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return o


def _ala(
    segno: float,
    x0: float,
    r_corpo: float,
    r_strozzo: float,
    r_ventaglio: float,
    lungo: float,
    schiaccia: float,
) -> list:
    """L'estremita' attorcigliata dell'incarto: uno strozzo, poi un ventaglio piatto."""
    giro = (0, segno * math.pi / 2, 0)
    strozzo = lungo * 0.4
    bpy.ops.mesh.primitive_cone_add(
        vertices=16,
        radius1=r_corpo,
        radius2=r_strozzo,
        depth=strozzo,
        location=(segno * (x0 + strozzo / 2), 0, 0),
        rotation=giro,
    )
    a = bpy.context.active_object
    ventaglio = lungo - strozzo
    bpy.ops.mesh.primitive_cone_add(
        vertices=16,
        radius1=r_strozzo,
        radius2=r_ventaglio,
        depth=ventaglio,
        location=(segno * (x0 + strozzo + ventaglio / 2), 0, 0),
        rotation=giro,
    )
    b = bpy.context.active_object
    # Il ventaglio e' piatto. Prima si applica la rotazione, cosi' gli assi
    # dell'oggetto sono quelli del mondo: schiacciare la z "propria" del cono
    # ruotato vorrebbe dire accorciarne l'asse, non lo spessore.
    for x in bpy.context.scene.objects:
        x.select_set(False)
    b.select_set(True)
    bpy.context.view_layer.objects.active = b
    bpy.ops.object.transform_apply(rotation=True)
    b.scale = (1, 1, schiaccia)
    return [a, b]


def gialla(m: dict):
    lx, ly, lz = m["corpo"]
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1)
    corpo = bpy.context.active_object
    corpo.scale = (lx / 2, ly / 2, lz / 2)
    pezzi = [corpo]
    for s in (1, -1):
        pezzi += _ala(s, lx / 2 - 0.0015, ly * 0.32, 0.0012, ly * 0.45, m["ala"], 0.35)
    return _unisci(pezzi, "gialla")


def rossana(m: dict):
    lungo, diametro = m["corpo"]
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=24, radius=diametro / 2, depth=lungo, rotation=(0, math.pi / 2, 0)
    )
    pezzi = [bpy.context.active_object]
    for s in (1, -1):
        pezzi += _ala(s, lungo / 2, diametro / 2, 0.0013, diametro * 0.5, m["ala"], 0.3)
    return _unisci(pezzi, "rossana")


def galatine(m: dict):
    lx, ly, lz = m["corpo"]
    bpy.ops.mesh.primitive_cube_add(size=1)
    corpo = bpy.context.active_object
    corpo.scale = (lx, ly, lz)
    bpy.ops.object.transform_apply(scale=True)
    smusso = corpo.modifiers.new("cuscinetto", "BEVEL")
    smusso.width = min(ly, lz) * 0.45
    smusso.segments = 4
    bpy.ops.object.modifier_apply(modifier="cuscinetto")
    pezzi = [corpo]
    for s in (1, -1):  # le due saldature dell'incarto, sui lati corti
        bpy.ops.mesh.primitive_cube_add(size=1, location=(s * (lx / 2 + m["saldatura"] / 2), 0, 0))
        b = bpy.context.active_object
        b.scale = (m["saldatura"], ly * 0.95, 0.0006)
        pezzi.append(b)
    return _unisci(pezzi, "galatine")


def bicchiere(m: dict):
    """Per rotazione di un profilo (raggio, quota): parete esterna, bordo, parete
    interna, fondo. Aperto sopra, spessore vero: e' una mesh solo da vedere."""
    rb, rt, h, sp = m["raggio_fondo"], m["raggio_bordo"], m["altezza"], m["spessore"]
    profilo = [(0.0, 0.0), (rb, 0.0), (rt, h), (rt - sp, h), (rb - sp, sp), (0.0, sp)]
    n = 48
    bm = bmesh.new()
    anelli = []
    for r, z in profilo:
        if r == 0.0:
            anelli.append([bm.verts.new((0, 0, z))])
        else:
            anelli.append(
                [
                    bm.verts.new(
                        (r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), z)
                    )
                    for k in range(n)
                ]
            )
    for a, b in zip(anelli[:-1], anelli[1:], strict=True):
        for k in range(n):
            k2 = (k + 1) % n
            if len(a) == 1:
                bm.faces.new((a[0], b[k], b[k2]))
            elif len(b) == 1:
                bm.faces.new((a[k], b[0], a[k2]))
            else:
                bm.faces.new((a[k], b[k], b[k2], a[k2]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    # Centrato in altezza: l'origine e' a meta' bicchiere.
    bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, -h / 2))
    me = bpy.data.meshes.new("bicchiere")
    bm.to_mesh(me)
    o = bpy.data.objects.new("bicchiere", me)
    bpy.context.collection.objects.link(o)
    return o


def _esporta(o, cartella: Path) -> None:
    for x in bpy.context.scene.objects:
        x.select_set(False)
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    dest = cartella / f"{o.name}.obj"
    bpy.ops.wm.obj_export(
        filepath=str(dest),
        export_selected_objects=True,
        export_uv=True,
        export_normals=True,
        export_materials=False,
        export_triangulated_mesh=True,
        forward_axis="Y",
        up_axis="Z",
    )
    print(
        f"OK: {o.name}, {len(o.data.vertices)} vertici, "
        f"dimensioni {tuple(round(v, 4) for v in o.dimensions)} -> {dest}"
    )


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :]
    misure = json.loads(Path(argv[0]).read_text())
    cartella = Path(argv[1])
    cartella.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for nome, costruttore in (
        ("gialla", gialla),
        ("rossana", rossana),
        ("galatine", galatine),
        ("bicchiere", bicchiere),
    ):
        _esporta(costruttore(misure[nome]), cartella)


main()
