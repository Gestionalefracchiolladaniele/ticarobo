"""Disegnare cio' che il robot sta pensando, dentro la scena.

**Perche' esiste questo file.** Guardando la finestra si vede un robot che si
sposta, e sembra andare a caso. Non va a caso: sceglie un ordine, traccia un
percorso, vede un ostacolo, cambia strada, calcola se la batteria basta. Ma
**niente di tutto questo e' visibile**, perche' il ragionamento non ha una
forma.

Il rimedio non e' spiegarlo a parole nel terminale -- quello c'e' gia' e si
legge male mentre la scena si muove. E' disegnarlo **dove succede**: il
percorso sul pavimento, il bersaglio dove si va, i raggi che colpiscono
qualcosa, la carica sopra il robot.

E' la stessa lezione di D18 in un'altra direzione: li' *"quando un sintomo e'
visibile, la verifica va fatta dove il sintomo si manifesta"* -- la camera che
guardava il soffitto si e' scoperta **guardando l'immagine**. Qui vale
l'inverso: se il comportamento e' visibile ma il motivo no, il motivo va reso
visibile.

**Come funziona tecnicamente.** MuJoCo tiene una lista di geometrie
"di scena" (``mjvScene.geoms``) separate dal modello fisico: si possono
aggiungere a ogni fotogramma senza toccare la simulazione. Non collidono, non
pesano, non esistono per la fisica. Il viewer passivo espone quella lista in
``viewer.user_scn``.

Il limite pratico: ``user_scn`` ha una capienza fissa (``maxgeom``), quindi
ogni disegno deve sapere quanto spazio occupa. Quando finisce, si smette di
disegnare invece di sovrascrivere -- sovrascrivere corromperebbe la scena.
"""

from __future__ import annotations

import mujoco
import numpy as np

# Colori. Sono scelti perche' si distinguano **fra loro** e dal magazzino, che
# e' grigio e legno: nessuno di questi e' un colore che la scena gia' usa.
VERDE = (0.15, 0.85, 0.35, 1.0)
GIALLO = (1.0, 0.85, 0.1, 1.0)
ROSSO = (0.95, 0.25, 0.2, 1.0)
CIANO = (0.2, 0.85, 0.95, 0.85)
VIOLA = (0.75, 0.4, 0.95, 1.0)
BIANCO = (1.0, 1.0, 1.0, 0.9)


class Overlay:
    """Disegna percorso, bersaglio, LiDAR e batteria nel viewer.

    Si costruisce una volta col viewer, e si chiama ``draw`` a ogni
    fotogramma. Non tiene stato del mondo: riceve tutto da fuori, perche'
    un disegno che si ricorda le cose diverge da cio' che disegna.
    """

    def __init__(self, viewer) -> None:
        self._v = viewer
        self._scn = viewer.user_scn
        self.max_geoms = self._scn.maxgeom

    # ------------------------------------------------------------------
    # primitive
    # ------------------------------------------------------------------

    def _next(self):
        """La prossima geometria libera, o ``None`` se lo spazio e' finito."""
        if self._scn.ngeom >= self.max_geoms:
            return None
        g = self._scn.geoms[self._scn.ngeom]
        self._scn.ngeom += 1
        return g

    def _sphere(self, pos, radius: float, rgba) -> None:
        g = self._next()
        if g is None:
            return
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0.0, 0.0]),
            np.asarray([pos[0], pos[1], pos[2] if len(pos) > 2 else 0.02], dtype=float),
            np.eye(3).flatten(),
            np.asarray(rgba, dtype=np.float32),
        )

    def _line(self, a, b, width: float, rgba) -> None:
        """Un segmento fra due punti 3D.

        ``mjv_connector`` fa il lavoro sporco: calcola posizione e
        orientamento del cilindro che unisce i due punti. Scriverlo a mano
        vorrebbe dire ricavare una rotazione da un vettore, che e' il tipo di
        conto in cui si sbagliano i segni.
        """
        g = self._next()
        if g is None:
            return
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).flatten(),
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
            np.asarray(a, dtype=float), np.asarray(b, dtype=float),
        )

    def _box(self, pos, half, rgba) -> None:
        g = self._next()
        if g is None:
            return
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_BOX,
            np.asarray(half, dtype=float),
            np.asarray(pos, dtype=float),
            np.eye(3).flatten(),
            np.asarray(rgba, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # i disegni
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Azzera. **Va chiamato a ogni fotogramma, prima di disegnare.**

        Senza, le geometrie si accumulano finche' ``maxgeom`` finisce, e da
        quel momento il disegno smette di aggiornarsi restando fermo
        sull'ultimo stato valido -- un overlay che mente, che e' peggio di
        nessun overlay.
        """
        self._scn.ngeom = 0

    def path(self, punti, *, z: float = 0.03, rgba=CIANO, width: float = 0.022) -> None:
        """Il percorso pianificato, come linea sul pavimento.

        Si disegna **diradato**: un percorso A* ha un punto ogni 5 cm, e
        seicento capsule costerebbero senza aggiungere niente all'occhio.
        """
        if punti is None or len(punti) < 2:
            return
        p = np.asarray(punti, dtype=float)
        passo = max(1, len(p) // 60)
        p = np.vstack([p[::passo], p[-1]])
        for a, b in zip(p[:-1], p[1:], strict=True):
            self._line((a[0], a[1], z), (b[0], b[1], z), width, rgba)

    def goal(self, xy, *, rgba=VERDE, h: float = 0.9) -> None:
        """Il bersaglio: un'asta verticale, visibile da lontano e da sopra."""
        if xy is None:
            return
        self._line((xy[0], xy[1], 0.02), (xy[0], xy[1], h), 0.035, rgba)
        self._sphere((xy[0], xy[1], h), 0.09, rgba)

    def scan(self, scan, *, ogni: int = 6, rgba=(1.0, 0.55, 0.1, 0.30)) -> None:
        """I raggi del LiDAR che colpiscono qualcosa.

        Uno ogni ``ogni`` raggi: 360 capsule renderebbero la scena illeggibile
        e mangerebbero tutto ``maxgeom``. Il punto non e' contarli, e' vedere
        **cosa il robot vede** -- e con 60 raggi la forma dell'ambiente si
        riconosce benissimo.
        """
        if scan is None:
            return
        z = 0.36
        o = scan.origin_world
        idx = np.flatnonzero(scan.hits)[::ogni]
        for i in idx:
            a = scan.angles[i] + scan.theta_world
            r = scan.ranges[i]
            self._line(
                (o[0], o[1], z),
                (o[0] + r * np.cos(a), o[1] + r * np.sin(a), z),
                0.006,
                rgba,
            )

    def hits(self, punti, *, rgba=ROSSO) -> None:
        """Gli ostacoli visti, come dischi rossi a terra."""
        if punti is None or len(punti) == 0:
            return
        for p in np.asarray(punti)[::4]:
            self._sphere((p[0], p[1], 0.05), 0.045, rgba)

    def battery(self, xy, level: float, *, z: float = 0.85) -> None:
        """La carica come barra sopra il robot.

        Il colore cambia con la soglia: verde, giallo sotto il 40%, rosso
        sotto la riserva. Un numero nel terminale dice quanto; un colore
        sopra il robot dice **se preoccuparsi**, che e' la domanda vera
        mentre si guarda.
        """
        lung = 0.6
        x0 = xy[0] - lung / 2
        colore = VERDE if level > 0.4 else (GIALLO if level > 0.25 else ROSSO)
        # fondo scuro: senza, una barra quasi vuota non si distingue
        self._box((xy[0], xy[1], z), (lung / 2, 0.012, 0.035), (0.1, 0.1, 0.1, 0.8))
        if level > 0.01:
            w = lung * level
            self._box((x0 + w / 2, xy[1], z), (w / 2, 0.016, 0.04), colore)

    def footprint(self, xy, radius: float, *, rgba=(1.0, 1.0, 1.0, 0.18)) -> None:
        """L'ingombro del robot, come anello a terra.

        Serve a rendere visibile **perche'** il robot non passa in un varco:
        l'anello e' largo quanto lui, e a occhio si vede se ci sta.
        """
        n = 24
        z = 0.012
        for k in range(n):
            a0 = 2 * np.pi * k / n
            a1 = 2 * np.pi * (k + 1) / n
            self._line(
                (xy[0] + radius * np.cos(a0), xy[1] + radius * np.sin(a0), z),
                (xy[0] + radius * np.cos(a1), xy[1] + radius * np.sin(a1), z),
                0.012,
                rgba,
            )
