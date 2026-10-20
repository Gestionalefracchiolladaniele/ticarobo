"""Rilevare gli urti invece di impedirli, e perche' e' la scelta giusta.

**Il problema, misurato.** Il modello ufficiale TidyBot comanda i giunti della
base con attuatori di posizione a guadagno altissimo (``kp=1000000``). Contro
un muro sviluppano **122.426 N** su un corpo di 60 kg: il contatto esiste e
MuJoCo lo rileva -- 12 contatti attivi -- ma perde. Il robot penetra di **94
mm** e ci resta dentro.

Non e' un bug del contatto: e' ``forza = kp x scarto``. Per una forza
ragionevole (500 N) lo scarto dovrebbe essere mezzo millimetro, e con mezzo
millimetro il robot non accelererebbe mai. Attuatore rigido e contatto cedevole
non possono coesistere.

**Cosa fanno gli autori del robot vero.** Il loro ``mujoco_env.py`` comanda
posizioni generate da Ruckig e **non tocca ne' solref ne' solimp**: nessuna
taratura del contatto. Il loro robot in simulazione attraverserebbe i muri
esattamente come questo. Per il loro compito -- manipolazione in ambiente
aperto -- non importa.

**La domanda giusta non era "come rendo i muri solidi".** Un AMR non deve mai
toccare un muro: se lo tocca, la navigazione ha gia' fallito, e questo vale
identico sul robot vero. Serve quindi **sapere che ha toccato**, non impedirlo
con una fisica ritarata a occhio.

Un rilevatore che misura le distanze e' anche **piu' severo** del contatto:

- prende lo **sfioramento**, che la fisica non registrerebbe come contatto ma
  che su un robot vero e' gia' un incidente;
- ha un **margine dichiarato**, quindi si puo' chiedere "quanto vicino e'
  passato" e non solo "ha urtato si/no";
- **non dipende dalla taratura**, quindi il numero che produce e' confrontabile
  con quello di un robot diverso.

E' la stessa mossa di ``safety/checks.py`` (D2): il valore non e' impedire, e'
verificare. Con la differenza che qui la verifica guarda **dove il sintomo si
manifesta** -- la posizione vera del robot nella scena -- invece che dove
sarebbe comodo misurarla.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

# Sotto questa distanza si dichiara un **urto**: penetrazione vera, non
# contatto a distanza nulla.
#
# **Perche' non zero, che sarebbe l'ovvio.** ``mj_geomDistance`` restituisce
# ``0.0000`` esatto in casi degeneri -- tipicamente quando la coppia di geom
# si affronta su uno **spigolo** -- anche con 4 cm di luce vera. Misurato: un
# robot fermo a 40 mm da uno scaffale produceva zeri su 3 geom su 5, e con
# soglia ``<= 0`` diventavano 39 urti inesistenti.
#
# Uno zero da spigolo e uno zero da contatto sono indistinguibili, quindi lo
# zero non puo' essere la soglia. -2 mm e' penetrazione vera: il robot e'
# dentro, e nessun artefatto di calcolo ci arriva.
#
# E' la terza volta in questa sessione che un valore plausibile risponde a una
# domanda diversa da quella posta (dopo ``geom_aabb`` e ``geom_rbound``).
CONTACT_MARGIN = -0.002

# Sotto questa si dichiara uno **sfioramento**: nessun danno, ma il robot e'
# passato piu' vicino di quanto un AMR dovrebbe. 5 cm e' il margine tipico
# di una zona di allarme su un LiDAR di sicurezza.
GRAZE_MARGIN = 0.05

# ``distmax`` per ``mj_geomDistance``: oltre questa distanza la funzione
# **satura** e restituisce ``distmax`` invece della distanza vera. Verificato:
# con distmax=0.1 su un oggetto a 1.36 m ritorna 0.1000.
#
# Non e' un difetto -- e' una query di prossimita', e il taglio e' cio' che la
# rende economica. Ma significa che ``min_distance`` nel report e' **limitata
# dall'alto**: se dice 0.55, vuol dire "almeno 0.55", non "esattamente".
QUERY_MAX = 0.55


@dataclass
class CollisionReport:
    """Cosa e' successo, contato per categoria.

    **Tre contatori separati e non uno**: urti, sfioramenti, e la distanza
    minima mai raggiunta. Mescolarli renderebbe un fallimento non attribuibile
    -- la lezione dei contatori del nastro (D17). "Ha urtato" e "e' passato a
    2 cm" hanno cause diverse e rimedi diversi.
    """

    contacts: int = 0
    grazes: int = 0
    min_distance: float = np.inf
    worst_geom: str = ""
    # chi e' stato toccato, e quante volte: serve a capire *dove* si sbaglia,
    # non solo quanto
    by_geom: dict[str, int] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        """Nessun urto. **Non** dice niente sugli sfioramenti, apposta."""
        return self.contacts == 0

    def summary(self) -> str:
        if self.contacts:
            return (
                f"{self.contacts} urti ({', '.join(f'{k}x{v}' for k, v in self.by_geom.items())}), "
                f"distanza minima {self.min_distance * 1000:.0f} mm"
            )
        if self.grazes:
            return (
                f"nessun urto, {self.grazes} sfioramenti, "
                f"minimo {self.min_distance * 1000:.0f} mm da {self.worst_geom}"
            )
        d = "inf" if np.isinf(self.min_distance) else f"{self.min_distance * 1000:.0f} mm"
        return f"pulito, distanza minima {d}"


class CollisionMonitor:
    """Sorveglia la distanza fra il robot e tutto il resto, tick per tick.

    Usa ``mj_geomDistance``, che da' la distanza vera fra due geom (negativa
    se compenetrano). E' la stessa funzione che MuJoCo usa internamente, quindi
    la risposta e' quella del motore, non una approssimazione con sfere.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        robot_body: str = "base_link",
        contact_margin: float = CONTACT_MARGIN,
        graze_margin: float = GRAZE_MARGIN,
        ignore_prefixes: tuple[str, ...] = ("stazione_", "pavimento"),
    ) -> None:
        self._m = model
        self._d = data
        self.contact_margin = contact_margin
        self.graze_margin = graze_margin

        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_body)
        if bid < 0:
            raise ValueError(f"body '{robot_body}' non trovato")
        bodies = {bid}
        for b in range(model.nbody):
            p = b
            while p > 0:
                if p == bid:
                    bodies.add(b)
                    break
                p = model.body_parentid[p]

        self._robot_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies]
        # Gli altri: tutto cio' che ha un corpo fisico e non e' il robot. Il
        # pavimento va escluso perche' il robot ci appoggia sopra per
        # definizione, e i segnaposto perche' sono scenografia.
        self._world_geoms = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] in bodies:
                continue
            if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if any(name.startswith(p) for p in ignore_prefixes):
                continue
            self._world_geoms.append(g)

        if not self._robot_geoms or not self._world_geoms:
            raise ValueError(
                f"niente da sorvegliare: {len(self._robot_geoms)} geom robot, "
                f"{len(self._world_geoms)} geom mondo"
            )
        self.report = CollisionReport()

    def reset(self) -> None:
        """Azzera i contatori. Si chiama all'inizio di ogni missione."""
        self.report = CollisionReport()

    def check(self) -> float:
        """Un controllo: aggiorna i contatori e ritorna la distanza minima.

        Il valore di ritorno serve a chi vuole reagire subito (fermare il
        robot) invece che solo contare.
        """
        best = np.inf
        best_name = ""
        hit: set[str] = set()
        grazed = False

        for rg in self._robot_geoms:
            for wg in self._world_geoms:
                # distmax: oltre questa distanza la funzione satura (vedi
                # QUERY_MAX). Tenerlo stretto e' cio' che rende il controllo
                # economico abbastanza da girare a 20 Hz: 0.35 ms per 110
                # coppie, su un budget di 50 ms.
                dist = mujoco.mj_geomDistance(self._m, self._d, rg, wg, QUERY_MAX, None)

                # lo zero esatto e' degenere, non una misura: si scarta.
                # Tenerlo falserebbe ``min_distance``, che e' il numero con
                # cui si risponde a "quanto vicino e' passato".
                if dist == 0.0:
                    continue

                if dist < best:
                    best = dist
                    best_name = (
                        mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_GEOM, wg) or f"geom{wg}"
                    )
                if dist <= self.contact_margin:
                    hit.add(
                        mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_GEOM, wg) or f"geom{wg}"
                    )
                elif dist <= self.graze_margin:
                    grazed = True

        r = self.report
        if best < r.min_distance:
            r.min_distance = float(best)
            r.worst_geom = best_name
        if hit:
            r.contacts += 1
            for name in hit:
                r.by_geom[name] = r.by_geom.get(name, 0) + 1
        elif grazed:
            r.grazes += 1
        return float(best)
