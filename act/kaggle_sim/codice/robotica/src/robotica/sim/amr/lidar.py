"""LiDAR 2D: N raggi in cerchio, distanza al primo ostacolo.

**Cos'e'.** Un LiDAR 2D gira su se stesso sparando raggi e per ognuno misura
quanto lontano c'e' qualcosa. L'uscita e' un array di N distanze, una per
angolo: non un'immagine, non una nuvola di punti -- un profilo del mondo
all'altezza in cui sta montato. E' il sensore su cui l'industria costruisce la
navigazione degli AMR, perche' funziona al buio e da' distanze vere invece che
apparenze.

**Perche' scritto con ``mj_ray`` invece del sensore nativo.** MuJoCo ha un
sensore ``rangefinder`` in MJCF, e la pratica corrente e' replicarlo N volte
col tag ``<replicate>``. Qui si e' scelto ``mj_ray`` per due ragioni:

1. non tocca i modelli ufficiali del menagerie -- la scena ci si costruisce
   intorno, il modello resta quello del produttore;
2. **il numero di raggi diventa un parametro Python invece che una costante
   XML**, e il passo 2 del piano e' proprio un esperimento su quel numero.
   Con ``<replicate>`` cambiare risoluzione vorrebbe dire rigenerare la scena.

L'alternativa esiste, e' quella che usa la community, ed e' documentata qui
perche' un giorno potrebbe convenire.

**I due filtri, entrambi gia' visti in D18.**

- *Il robot vede se stesso.* I raggi partono dal centro della base, che sta
  **dentro** il guscio: senza escludere i propri geom, ogni raggio tornerebbe
  con ~25 cm. E' il self-filtering del braccio in D18, nella stessa forma.
- *I segnaposto non sono ostacoli.* I dischi delle stazioni hanno
  ``contype=0``; ``mj_ray`` puo' ignorarli, ma glielo si deve chiedere.

**Cosa NON trasferisce al robot vero**, e va detto qui perche' il codice non
lo puo' dire da solo: questi raggi sono geometria pura, quindi niente rumore,
niente raggi persi. Un LiDAR vero ha un errore di qualche centimetro, e
soprattutto **il vetro e le superfici nere opache spariscono**: il raggio non
torna, e il sensore riporta "niente", cioe' via libera dove c'e' una parete di
vetro. Il transfer stimato e' 70-80% (vedi ``docs/transfer-sim-real.md``), e
il mancante sta quasi tutto li'. ``noise_std`` esiste per rendere la cosa
provabile, non per farla sembrare vera.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# Quanti raggi. 360 = un grado di risoluzione, che e' la configurazione comoda
# per lo SLAM 2D e quella dichiarata dalla maggior parte dei LiDAR economici.
# Il numero e' un **parametro misurabile**, non una verita': la letteratura
# dice ~12 raggi bastano a evitare ostacoli, ~180 e' il minimo per SLAM, 720+
# somiglia a un sensore vero. Quale serva *in questa scena* e' un esperimento.
DEFAULT_N_RAYS = 360

# Portata. Un LiDAR da interni tipico arriva a 12-25 m; il magazzino e' 10x8,
# quindi 12 m vede da parete a parete e la saturazione non nasconde niente.
DEFAULT_MAX_RANGE = 12.0

# A che altezza sta montato. Deve stare **sopra il pavimento** (altrimenti ogni
# raggio colpirebbe il piano subito) e **sotto il bordo degli ostacoli piu'
# bassi**, o non li vedrebbe. I cassoni sono alti 0.6 m, gli scaffali 1.0:
# 0.35 m li prende tutti. E' la quota tipica di montaggio su un AMR.
DEFAULT_MOUNT_Z = 0.35

# Valore riportato quando il raggio non colpisce niente entro la portata.
# Si usa ``max_range`` e non ``inf`` o ``nan``: un consumatore che fa
# aritmetica su inf produce silenziosamente nan, e nan confrontato con
# qualunque cosa e' falso -- cioe' un ostacolo che sparisce senza errore.
NO_HIT = "max_range"


@dataclass(frozen=True)
class Scan:
    """Una scansione: angoli, distanze, e cosa e' stato colpito.

    ``angles`` sono nel frame **del robot** (0 = davanti, antiorario), perche'
    e' cosi' che un LiDAR vero riporta: il sensore non sa dove sia il nord.
    ``points_world`` e' la comodita' che serve al planner, ed e' calcolata --
    non misurata -- usando la posa della base.

    ``hit_geom`` esiste per i test: permette di chiedere *"quel ritorno a 1.2 m
    era davvero il muro nord?"*. E' informazione privilegiata, come
    ``known_positions`` in D19: la navigazione non la usa, i test si'.
    """

    angles: np.ndarray
    ranges: np.ndarray
    hit_geom: np.ndarray
    max_range: float
    origin_world: np.ndarray
    theta_world: float

    @property
    def hits(self) -> np.ndarray:
        """Maschera dei raggi che hanno colpito qualcosa entro la portata."""
        return self.ranges < self.max_range

    @property
    def points_world(self) -> np.ndarray:
        """I punti colpiti, in coordinate mondo. Solo i raggi validi.

        Shape ``(k, 2)``: e' la forma che vuole la costmap, e coincide con
        cio' che ``perception/`` chiama gia' "un insieme di punti".
        """
        m = self.hits
        a = self.angles[m] + self.theta_world
        r = self.ranges[m]
        return np.column_stack(
            [self.origin_world[0] + r * np.cos(a), self.origin_world[1] + r * np.sin(a)]
        )

    def nearest(self) -> tuple[float, float]:
        """L'ostacolo piu' vicino: ``(distanza, angolo nel frame robot)``.

        Se non c'e' niente entro la portata, ritorna ``(max_range, 0.0)``.
        Non solleva: "la corsia e' libera" e' una risposta, non un errore.
        """
        m = self.hits
        if not m.any():
            return self.max_range, 0.0
        i = int(np.argmin(np.where(m, self.ranges, np.inf)))
        return float(self.ranges[i]), float(self.angles[i])


class Lidar2D:
    """Un LiDAR 2D montato sulla base, simulato con ``mj_ray``.

    Costruirlo una volta e riusarlo: il costo di setup e' trovare quali geom
    appartengono al robot, e rifarlo a ogni scansione sarebbe spreco su 2 core.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        n_rays: int = DEFAULT_N_RAYS,
        max_range: float = DEFAULT_MAX_RANGE,
        mount_z: float = DEFAULT_MOUNT_Z,
        noise_std: float = 0.0,
        robot_body: str = "base_link",
        seed: int | None = None,
    ) -> None:
        if n_rays < 1:
            raise ValueError(f"servono almeno 1 raggio, ricevuto {n_rays}")
        if max_range <= 0:
            raise ValueError(f"portata deve essere positiva, ricevuta {max_range}")
        self._m = model
        self._d = data
        self.n_rays = n_rays
        self.max_range = float(max_range)
        self.mount_z = float(mount_z)
        self.noise_std = float(noise_std)
        self._rng = np.random.default_rng(seed)

        # Angoli fissi, precalcolati: girano con il robot, non nel mondo.
        # endpoint=False perche' 0 e 2pi sono lo stesso raggio, e contarlo due
        # volte falserebbe qualunque statistica sul numero di ritorni.
        self.angles = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
        self._cos = np.cos(self.angles)
        self._sin = np.sin(self.angles)

        # --- self-filtering: quali geom sono il robot -----------------------
        # ``mj_ray`` accetta un array di flag per gruppo di geom, ma i geom del
        # robot stanno negli stessi gruppi della scena. L'unico criterio solido
        # e' l'appartenenza al body: si escludono i geom del robot e di tutto
        # cio' che gli e' attaccato.
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_body)
        if bid < 0:
            raise ValueError(f"body '{robot_body}' non trovato nel modello")
        self._robot_bodies = self._descendants(bid)
        self._self_geoms = {
            g for g in range(model.ngeom) if model.geom_bodyid[g] in self._robot_bodies
        }
        if not self._self_geoms:
            raise ValueError(
                f"nessun geom trovato per il body '{robot_body}': "
                "il self-filtering non avrebbe niente da filtrare, "
                "e ogni raggio tornerebbe con la distanza al proprio guscio"
            )

    def _descendants(self, root: int) -> set[int]:
        """Il body e tutti quelli appesi sotto. Un braccio montato ne fa parte."""
        out = {root}
        for b in range(self._m.nbody):
            p = b
            while p > 0:
                if p == root:
                    out.add(b)
                    break
                p = self._m.body_parentid[p]
        return out

    def scan(self) -> Scan:
        """Una scansione completa dalla posa attuale della base.

        Un raggio per volta: ``mj_ray`` non e' vettorizzato sulle direzioni, e
        l'alternativa (``mj_multiRay``) vuole un array preparato che su 360
        raggi non cambia il tempo in modo percepibile. Misurato prima di
        decidere, non supposto -- vedi ``esperimenti/tidybot/navigazione.md``.
        """
        bid = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        pos = self._d.xpos[bid].copy()
        # l'orientamento nel piano dalla matrice di rotazione del body: si
        # **legge dalla scena**, non si tiene a parte (convenzione del progetto)
        rot = self._d.xmat[bid].reshape(3, 3)
        theta = float(np.arctan2(rot[1, 0], rot[0, 0]))

        origin = np.array([pos[0], pos[1], self.mount_z])
        ranges = np.full(self.n_rays, self.max_range)
        hit_geom = np.full(self.n_rays, -1, dtype=np.int32)

        # direzioni nel mondo: gli angoli del sensore ruotati con la base
        ca, sa = np.cos(theta), np.sin(theta)
        dx = ca * self._cos - sa * self._sin
        dy = sa * self._cos + ca * self._sin

        geomid = np.zeros(1, dtype=np.int32)
        vec = np.zeros(3)
        for i in range(self.n_rays):
            vec[0], vec[1], vec[2] = dx[i], dy[i], 0.0
            dist = mujoco.mj_ray(
                self._m,
                self._d,
                origin,
                vec,
                None,  # nessun filtro per gruppo: si filtra per body, sotto
                1,  # includi i geom statici (muri e scaffali lo sono)
                -1,  # nessun body escluso a monte: vedi il ciclo di rifiuto
                geomid,
            )
            g = int(geomid[0])
            # rifiuto dei propri geom: se il primo colpo e' il guscio del
            # robot, si rilancia il raggio da appena oltre. Non capita con il
            # montaggio attuale (il centro e' dentro, ma il guscio e' convesso
            # e il raggio esce), ma un braccio montato sopra lo renderebbe
            # possibile -- e un sensore che vede il proprio robot e' il
            # fallimento di D18.
            if g in self._self_geoms and dist >= 0:
                nudge = origin + vec * (dist + 1e-3)
                dist = mujoco.mj_ray(self._m, self._d, nudge, vec, None, 1, -1, geomid)
                g = int(geomid[0])
                if dist >= 0:
                    dist += float(np.linalg.norm(nudge - origin))
            if dist >= 0 and dist < self.max_range:
                ranges[i] = dist
                hit_geom[i] = g

        if self.noise_std > 0:
            m = ranges < self.max_range
            ranges[m] = np.clip(
                ranges[m] + self._rng.normal(0.0, self.noise_std, m.sum()),
                0.0,
                self.max_range,
            )

        return Scan(
            angles=self.angles.copy(),
            ranges=ranges,
            hit_geom=hit_geom,
            max_range=self.max_range,
            origin_world=np.array([pos[0], pos[1]]),
            theta_world=theta,
        )

    def geom_name(self, geom_id: int) -> str:
        """Il nome del geom colpito. Serve ai test, non alla navigazione."""
        if geom_id < 0:
            return ""
        return mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
