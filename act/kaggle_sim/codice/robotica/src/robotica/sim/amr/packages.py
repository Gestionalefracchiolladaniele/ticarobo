"""I pacchi alla banchina: compaiono, si guardano, si riciclano.

**Perche' esistono.** Senza, l'AMR *esegue* ordini scritti in una lista:
gli si dice "porta da banchina a spedizione" e lui ci va. Sul nastro (D17)
il colore veniva da una camera e la decisione nasceva da li'; qui no, e si
vede — sembra uno script fisso. Un pacco che **compare** e di cui si deve
**scoprire** il colore e' cio' che trasforma l'esecuzione in una decisione.

**Il colore sta nell'oggetto, non nel nome** (D15). Un pacco si chiama
``pacco_0``, non ``pacco_rosso_1``: il nome viene dal tempo (D19), il colore
si percepisce. Chi ordina non sa dove andra' il pacco — lo sapra' il robot,
dopo averlo guardato.

**Uno per volta, e si riciclano.** In MuJoCo il numero di corpi e' fissato
alla compilazione: non si aggiungono oggetti a scena viva. Quindi non
esistono venti pacchi -- ne esistono ``POOL_SIZE``, e un pacco consegnato
torna al **parcheggio** fuori dal magazzino per rinascere con un altro
colore. E' la stessa tecnica dei cassoni di ``warehouse.py``
(``OBSTACLE_PARK``), ed evita che il magazzino si riempia di roba: venti
pacchi lasciati in giro diventerebbero venti ostacoli, e la scena
smetterebbe di dimostrare cio' per cui e' fatta.

**La quota si legge, non si deriva.** Il pacco e' un cubo di lato
``PACKAGE_SIZE`` appoggiato sul pavimento: il centro sta a meta' lato, e
quel numero viene da qui -- non da "circa cinque centimetri".

**Cosa trasferisce.** Niente: questo file **e' la scena**, come
``sim/conveyor.py`` per il nastro. Su un magazzino vero i pacchi arrivano
da soli e nessuno li fa comparire. Cio' che trasferisce e' il layer che li
**guarda** (``perception/``) e quello che **decide** (``navigation/``) --
e infatti nessuno dei due importa questo modulo.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import mujoco
import numpy as np

__all__ = [
    "PACKAGE_SIZE",
    "POOL_SIZE",
    "PACKAGE_COLORS",
    "Package",
    "PackageSpawner",
    "add_packages_to_spec",
]

# Lato del cubo-pacco, in metri. Un pacco di magazzino non e' un cubetto da
# tavolo: dev'essere visibile da 1-2 m, che e' dove un AMR si ferma davanti
# a una stazione. Misurato: a 1.20 m occupa 1734 px su 640x480.
PACKAGE_SIZE = 0.12

# Quanti pacchi esistono davvero nella scena. Bastano pochi: uno in vista
# alla banchina, gli altri al parcheggio pronti a rinascere. Tre permette
# una piccola coda senza gonfiare la scena (il magazzino ha 5 corpi in
# tutto, e ogni corpo in piu' e' fisica in piu' su 2 core).
POOL_SIZE = 3

# I colori nominabili e il loro RGB. **Sono gli stessi di
# ``perception/detect.py``**, e non per comodita': quel modulo li usa come
# riferimenti da confrontare, questo li usa per dipingere. Se divergessero,
# il robot vedrebbe colori che non sa nominare -- e sarebbe un guasto
# silenzioso, perche' ``classify_color`` direbbe solo ``None``.
PACKAGE_COLORS: dict[str, tuple[float, float, float]] = {
    "rosso": (0.85, 0.2, 0.2),
    "blu": (0.2, 0.35, 0.85),
    "verde": (0.2, 0.7, 0.3),
    "giallo": (0.9, 0.75, 0.15),
}

# Dove aspettano i pacchi non ancora comparsi: **fuori dal magazzino**, come
# i cassoni. Sotto il pavimento sarebbe piu' nascosto ma il LiDAR e la
# fisica li troverebbero comunque; fuori dai muri no.
PACKAGE_PARK = (0.0, 6.0)

# Distanza fra un pacco e l'altro al parcheggio, per non impilarli.
_PARK_SPACING = 0.5


@dataclass
class Package:
    """Un pacco: dove sta, di che colore e', e se e' stato guardato.

    Attributes:
        name: ``pacco_0``, ``pacco_1``... Il nome viene dall'**ordine di
            creazione**, non dal colore (D15, D19).
        color: il colore vero, quello con cui e' dipinto. **Chi decide non
            deve leggerlo**: e' l'oracolo con cui si verifica la percezione,
            come ``observer.perceive`` in D19.
        station: la stazione dove si trova, o ``None`` se e' al parcheggio.
        seen_as: il colore **percepito** dalla camera, ``None`` finche'
            nessuno l'ha guardato. Che sia separato da ``color`` e' il
            punto: la differenza fra i due e' l'errore di percezione, e un
            campo che non esiste non si puo' misurare.
    """

    name: str
    color: str
    station: str | None = None
    seen_as: str | None = None
    delivered_to: str | None = None

    @property
    def perceived(self) -> bool:
        """E' stato guardato? Diverso da 'riconosciuto'."""
        return self.seen_as is not None

    def __str__(self) -> str:
        visto = self.seen_as or "?"
        return f"{self.name} (visto: {visto})"


def add_packages_to_spec(spec: mujoco.MjSpec, n: int = POOL_SIZE) -> None:
    """Aggiunge ``n`` pacchi alla scena, tutti al parcheggio.

    Si usa ``MjSpec`` -- l'albero del modello ancora modificabile -- invece
    di incollare stringhe nell'XML: e' l'API ufficiale, e ``sim/scene.py``
    gia' monta cosi' le camere del braccio.

    Ogni pacco ha un ``freejoint``, quindi lo si sposta scrivendo ``qpos``
    come fanno i cassoni. Senza giunto sarebbe geometria fissa, e comparire
    e' precisamente muoversi.
    """
    h = PACKAGE_SIZE / 2
    for i in range(n):
        body = spec.worldbody.add_body(
            name=f"pacco_{i}",
            pos=[PACKAGE_PARK[0] + i * _PARK_SPACING, PACKAGE_PARK[1], h],
        )
        body.add_freejoint(name=f"pacco_{i}_free")
        body.add_geom(
            name=f"pacco_{i}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[h, h, h],
            rgba=[0.5, 0.5, 0.5, 1.0],  # il colore vero si dipinge a runtime
            mass=2.0,
        )


class PackageSpawner:
    """Fa comparire i pacchi, uno per volta, e li ricicla.

    **Non decide dove vanno.** Sa solo far comparire un pacco di un certo
    colore in una certa stazione, e toglierlo quando e' consegnato. Che il
    rosso vada a spedizione e il verde si lasci li' e' una regola di
    ``navigation/``, non della scena: se stesse qui, la scena saprebbe la
    risposta che il robot deve **scoprire**.

    Args:
        model, data: la scena.
        seed: il generatore dei colori. Fissarlo rende una sequenza
            ripetibile -- serve ai test; lasciarlo a ``None`` rende ogni
            lancio diverso, che e' cio' che si vuole guardando.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        seed: int | None = None,
        colors: dict[str, tuple[float, float, float]] | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.rng = random.Random(seed)
        self.colors = dict(colors or PACKAGE_COLORS)
        self._next_index = 0
        self.packages: dict[str, Package] = {}
        self._free: list[int] = []

        # quali slot esistono davvero nella scena? Si legge dal modello.
        self.slots: list[int] = []
        i = 0
        while mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"pacco_{i}") >= 0:
            self.slots.append(i)
            self._free.append(i)
            i += 1
        self._slot_of: dict[str, int] = {}

    # -- interrogare -----------------------------------------------------

    @property
    def active(self) -> list[Package]:
        """I pacchi in questo momento dentro il magazzino."""
        return [p for p in self.packages.values() if p.station is not None]

    def at_station(self, station: str) -> list[Package]:
        """I pacchi che stanno in una stazione. Puo' essere vuota."""
        return [p for p in self.active if p.station == station]

    # -- agire -----------------------------------------------------------

    def spawn(self, station: str, xy: np.ndarray, color: str | None = None) -> Package | None:
        """Fa comparire un pacco in una stazione. ``None`` se non ci sono slot.

        Che possa tornare ``None`` non e' un dettaglio: gli slot sono finiti,
        e un sistema che esaurisce una risorsa deve **dirlo** invece di
        fallire piu' avanti per una ragione che sembra scollegata (D14).
        """
        if not self._free:
            return None
        slot = self._free.pop(0)
        colore = color or self.rng.choice(list(self.colors))
        nome = f"pacco_{self._next_index}"
        self._next_index += 1

        pkg = Package(name=nome, color=colore, station=station)
        self.packages[nome] = pkg
        self._slot_of[nome] = slot

        self._move_slot(slot, float(xy[0]), float(xy[1]), PACKAGE_SIZE / 2)
        self._paint_slot(slot, colore)
        return pkg

    def remove(self, pkg: Package, delivered_to: str | None = None) -> None:
        """Toglie un pacco dalla scena e libera il suo slot.

        Il corpo torna al parcheggio invece di sparire, perche' in MuJoCo
        non si puo' cancellare: si sposta dove non da' fastidio.
        """
        slot = self._slot_of.get(pkg.name)
        if slot is None:
            return
        pkg.station = None
        pkg.delivered_to = delivered_to
        self._move_slot(
            slot,
            PACKAGE_PARK[0] + slot * _PARK_SPACING,
            PACKAGE_PARK[1],
            PACKAGE_SIZE / 2,
        )
        self._free.append(slot)
        del self._slot_of[pkg.name]

    # -- la scena --------------------------------------------------------

    def _move_slot(self, slot: int, x: float, y: float, z: float) -> None:
        """Sposta il corpo scrivendo ``qpos``, e **azzera la velocita'**.

        Azzerare ``qvel`` non e' cerimoniale: un corpo teletrasportato che
        conserva la velocita' di prima parte scivolando, e un pacco che
        striscia via dalla banchina sarebbe un guasto difficile da leggere.
        """
        jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, f"pacco_{slot}_free"
        )
        adr = self.model.jnt_qposadr[jid]
        self.data.qpos[adr : adr + 3] = [x, y, z]
        self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        vadr = self.model.jnt_dofadr[jid]
        self.data.qvel[vadr : vadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _paint_slot(self, slot: int, color: str) -> None:
        """Dipinge il cubo. ``geom_rgba`` si scrive a scena viva."""
        gid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, f"pacco_{slot}_geom"
        )
        r, g, b = self.colors[color]
        self.model.geom_rgba[gid] = [r, g, b, 1.0]
