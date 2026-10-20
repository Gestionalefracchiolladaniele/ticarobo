"""L'esperto: dimostrazioni di smistamento caramelle, fatte da uno script.

Imita **Daniele**, non un robot ideale: ogni gesto, quota e durata viene dai
50 episodi registrati col leader (esperimenti/so101/act.md), e si estrae a
caso nelle loro distribuzioni. L'esperto sa tutto (dove sta ogni caramella, in
quale bicchiere va); la rete che imparera' dai suoi episodi vedra' solo
l'immagine e gli angoli.

Regole di D26: prima la piu' a sinistra (+y), le gialle a sinistra, le
galatine a destra, la rossana mai. Le posizioni si **rileggono dalla scena**
prima di ogni presa: una caramella urtata si prende dove sta ora.

Il braccio simulato riceve il comando **3 fotogrammi dopo** (0,1 s): e' il
ritardo misurato del follower vero, e col rigioco dei comandi veri riduce lo
scarto simulato-vero (pan al 95° percentile da 4,6 a 1,1 gradi).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np

from robotica.reale.so101 import a_lerobot, da_lerobot
from robotica.sim.so101_caramelle import DITA, MESH, MISURE, Episodio
from robotica.sim.so101_presa import ASSE_BASE, PARTENZE, modello, pinza_su_caramella
from robotica.sim.so101_rps import GIUNTI

FPS = 30
RITARDO_COMANDO = 3  # fotogrammi

#: Posa di riposo, gradi LeRobot: mediana del primo fotogramma dei 50 episodi.
#: Il roll vi variava molto (deviazione 45): lo si estrae.
RIPOSO = np.array([-11.5, -102.9, 94.9, 74.9, 51.1, 1.1])
RIPOSO_ROLL = (10.0, 95.0)

#: Roll consentito, gradi LeRobot: le prese vere stanno fra 4 e 138; oltre
#: ±180 il numero salta e il motore gira dalla parte lunga (D23).
ROLL_LEROBOT = (-15.0, 160.0)

#: Dai 50 episodi (10°-90° percentile). Quote del centro delle dita in metri
#: sopra il tavolo, pinza in unita' LeRobot, durate in secondi. L'inclinazione
#: non e' qui: cambia con la posizione (a 26 cm Daniele prende col gomito
#: girato), e la si copia dalle sue pose (``pose_vere.npz``).
PINZA_APERTA = (14.0, 28.0)
PINZA_RILASCIO = (12.0, 22.0)
QUOTA_TRASPORTO = (0.115, 0.15)
QUOTA_RILASCIO = (0.09, 0.115)
QUOTA_VUOTO = (0.10, 0.125)
QUOTA_AVVICINA = (0.03, 0.05)  # sopra la caramella, prima di scendere piano
CHIUSURA = (0.15, 0.3)


@dataclass
class Registro:
    """Cio' che finisce nel dataset, un fotogramma alla volta."""

    stato: list[np.ndarray] = field(default_factory=list)  # gradi LeRobot, letti
    azione: list[np.ndarray] = field(default_factory=list)  # gradi LeRobot, comandati


class Cinematica:
    """Cinematica inversa sul **centro fra le dita**, col roll dato.

    Il roll non cambia dove puntano le dita, ma sposta il loro centro (10 mm di
    lato rispetto al polso): per questo entra nel calcolo e non si aggiunge
    dopo. Si parte da una **posa vera vicina** (atlante di ``scripts/pose_vere.py``)
    e se ne copia l'inclinazione; poi dalla soluzione precedente, poi dalle pose
    fisse di ``so101_presa``.
    """

    #: Quante pose vere vicine si considerano, e quanto si varia l'inclinazione copiata.
    VICINE = 8
    INCLINAZIONE_VARIA = np.radians(3.0)
    #: Gomito sotto questo valore = steso: le pose vere sopra i bicchieri ci
    #: arrivano (0,04), ma col polso oltre il limite del modello. Qui si chiede
    #: un minimo di piegatura, in entrambi i versi.
    GOMITO_MINIMO = 0.10

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._inizializza_modello()
        self.rng = rng if rng is not None else np.random.default_rng(0)
        atlante = np.load(MESH / "pose_vere.npz")
        self.atlante_q = atlante["q"].astype(float)
        self.atlante_centro = atlante["centro"].astype(float)
        self.atlante_incl = atlante["inclinazione"].astype(float)

    def _inizializza_modello(self) -> None:
        self.m = modello()
        self.d = mujoco.MjData(self.m)
        self.rp = tuple(self.m.joint("gripper").range)
        self.adr = [self.m.joint(n).qposadr[0] for n in GIUNTI]
        self.dof = [self.m.joint(n).dofadr[0] for n in GIUNTI[:4]]
        self.lo = np.array([self.m.joint(n).range[0] for n in GIUNTI[:4]])
        self.hi = np.array([self.m.joint(n).range[1] for n in GIUNTI[:4]])
        self._f = self.m.geom("fixed_jaw_sph_tip1")
        self._g = self.m.geom("moving_jaw_sph_tip1")
        self.precedente: np.ndarray | None = None

    def _pose(self, q6: np.ndarray) -> None:
        self.d.qpos[self.adr] = q6
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def dita(self, q6: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(centro fra le punte, verso delle dita, verso di apertura) per sei angoli."""
        self._pose(q6)
        a, b = self.d.geom_xpos[self._f.id].copy(), self.d.geom_xpos[self._g.id].copy()
        verso = -self.d.body("gripper").xmat.reshape(3, 3)[:, 2]
        apre = (b - a) / (np.linalg.norm(b - a) + 1e-12)
        return (a + b) / 2, verso, apre

    def _risolvi(
        self,
        q4: np.ndarray,
        roll: float,
        pinza: float,
        punto: np.ndarray,
        inclinazione: float,
        iterazioni: int,
    ) -> tuple[np.ndarray, float, float]:
        """Posizione (3) + **sola inclinazione** (1): quattro richieste, quattro giunti.

        Non si chiede anche *in che piano* puntano le dita: stanno sempre nel
        piano del braccio (il roll non le sposta), e il loro centro sta 10 mm di
        lato. Chiedere il piano asse-bersaglio, come fa ``so101_presa.postura``
        per ``gripperframe``, qui e' una richiesta di troppo: la soluzione resta
        a 2,4 mm qualunque sia la partenza (misurato).
        """
        jp_a, jp_b = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        voluto = -np.sin(inclinazione)  # componente verticale del verso delle dita
        q = q4.copy()
        for _ in range(iterazioni):
            self._pose(np.array([*q, roll, pinza]))
            a, b = self.d.geom_xpos[self._f.id], self.d.geom_xpos[self._g.id]
            z = self.d.body("gripper").xmat.reshape(3, 3)[:, 2]  # le dita lungo -z
            err = np.array([*(punto - (a + b) / 2), voluto + z[2]])
            if np.linalg.norm(err[:3]) < 2e-4 and abs(err[3]) < 2e-3:
                break
            mujoco.mj_jacGeom(self.m, self.d, jp_a, None, self._f.id)
            mujoco.mj_jacGeom(self.m, self.d, jp_b, None, self._g.id)
            mujoco.mj_jacBody(self.m, self.d, None, jr, self.m.body("gripper").id)
            jd = np.cross(z[:, None], jr[:, self.dof], axis=0)  # derivata di -z
            J = np.vstack([(jp_a[:, self.dof] + jp_b[:, self.dof]) / 2, jd[2:3]])
            passo = np.linalg.solve(J.T @ J + 1e-4 * np.eye(4), J.T @ err)
            q = np.clip(q + 0.7 * passo, self.lo, self.hi)
        c, v, _ = self.dita(np.array([*q, roll, pinza]))
        scarto = abs(float(np.arcsin(np.clip(-v[2], -1, 1))) - inclinazione)
        return q, float(np.linalg.norm(punto - c)), scarto

    def postura(self, punto: np.ndarray, roll: float, pinza: float) -> np.ndarray:
        """Sei angoli (radianti, modello) col centro delle dita in ``punto``,
        inclinate come Daniele le inclinava li'.

        Solleva ``ValueError`` se nessuna partenza converge.
        """
        dist = np.linalg.norm(self.atlante_centro - punto, axis=1)
        vicine = np.argsort(dist)[: self.VICINE]
        self.rng.shuffle(vicine)
        tentativi: list[tuple[np.ndarray, float]] = []
        for i in vicine:
            incl = self.atlante_incl[i] + self.rng.uniform(-1, 1) * self.INCLINAZIONE_VARIA
            tentativi.append((self.atlante_q[i, :4], float(incl)))
        incl0 = float(self.atlante_incl[vicine[0]])
        if self.precedente is not None:
            tentativi.append((self.precedente[:4], incl0))
        pan = float(np.arctan2(punto[1] - ASSE_BASE[1], punto[0] - ASSE_BASE[0]))
        tentativi += [(np.array([pan, *p]), incl0) for p in PARTENZE]
        # Sopra i bicchieri Daniele tiene il polso oltre il limite del modello
        # (1,69-1,77 contro 1,66 rad): la sua inclinazione li' non si puo'
        # copiare. Si scende di 5 gradi alla volta, fino alla piu' vicina possibile.
        semi = [q0 for q0, _ in tentativi[:3]]
        for meno in np.radians([5, 10, 15, 20, 25, 30]):
            tentativi += [(q0, incl0 - meno) for q0 in semi]
        for q0, incl in tentativi:
            q, errore, scarto = self._risolvi(q0, roll, pinza, punto, incl, 200)
            if errore < 1.5e-3 and scarto < np.radians(3) and abs(q[2]) >= self.GOMITO_MINIMO:
                q6 = np.array([*q, roll, pinza])
                self.precedente = q6
                return q6
        raise ValueError(
            f"nessuna postura per il centro delle dita in {np.round(punto, 3)} "
            f"(posa vera piu' vicina a {dist[vicine].min() * 1000:.0f} mm)"
        )

    def roll_per(self, q6: np.ndarray, asse_lungo: np.ndarray, vicino_a: float) -> float:
        """Il roll per cui le dita si aprono **di traverso** al lato lungo della caramella.

        Si cerca su un ventaglio di angoli, invece di dedurre un segno: la
        caramella e' simmetrica, quindi i candidati sono due ogni giro, e si
        tiene quello consentito piu' vicino a ``vicino_a``.
        """
        lo = da_lerobot(dict(zip(GIUNTI, [0, 0, 0, 0, ROLL_LEROBOT[0], 0], strict=True)), self.rp)[
            4
        ]
        hi = da_lerobot(dict(zip(GIUNTI, [0, 0, 0, 0, ROLL_LEROBOT[1], 0], strict=True)), self.rp)[
            4
        ]
        lo, hi = (
            max(lo, self.m.joint("wrist_roll").range[0]),
            min(hi, self.m.joint("wrist_roll").range[1]),
        )
        asse = asse_lungo[:2] / (np.linalg.norm(asse_lungo[:2]) + 1e-12)
        candidati = np.linspace(lo, hi, 181)
        costo = []
        for r in candidati:
            _, _, apre = self.dita(np.array([*q6[:4], r, q6[5]]))
            costo.append(abs(apre[:2] @ asse))
        costo = np.array(costo)
        buoni = candidati[costo < costo.min() + 0.05]
        return float(buoni[np.argmin(np.abs(buoni - vicino_a))])

    def pinza(self, lerobot: float) -> float:
        return float(
            da_lerobot(dict(zip(GIUNTI, [0, 0, 0, 0, 0, lerobot], strict=True)), self.rp)[5]
        )


def _minimo_jerk(s: float) -> float:
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class Esperto:
    """Esegue un episodio nella scena ``(m, d, ep)`` e lo registra.

    ``guarda`` riceve ogni fotogramma (stato, azione) prima che il comando parta:
    e' il punto dove il generatore rende l'immagine. ``None`` = solo fisica,
    per provare l'episodio prima di pagarne il render.
    """

    def __init__(
        self, m: mujoco.MjModel, d: mujoco.MjData, ep: Episodio, rng: np.random.Generator
    ) -> None:
        self.m, self.d, self.ep, self.rng = m, d, ep, rng
        self.k = Cinematica(rng)
        self.rp = tuple(m.joint("gripper").range)
        self.adr = [m.joint(n).qposadr[0] for n in GIUNTI]
        self.passi = int(round(1 / FPS / m.opt.timestep))
        self.registro = Registro()
        self.coda: list[np.ndarray] = []
        self.comando = np.zeros(6)
        self.guarda: Callable[[np.ndarray, np.ndarray], None] | None = None
        self.da_agganciare: str | None = None  # caramella che le dita stanno stringendo
        self.agganciata: str | None = None
        self.prese: set[str] = set()
        self._dita = [m.body(n).id for n in DITA]
        self._pinza = m.joint("gripper").id
        self._limite_pinza = float(m.jnt_range[self._pinza, 0])

    # --- il tempo ------------------------------------------------------------

    def _fotogramma(self, comando: np.ndarray) -> None:
        stato = self._lerobot(self.d.qpos[self.adr])
        azione = self._lerobot(comando)
        if self.guarda is not None:
            self.guarda(stato, azione)
        self.registro.stato.append(stato)
        self.registro.azione.append(azione)
        self.coda.append(comando.copy())
        self.d.ctrl[:] = self.coda[max(0, len(self.coda) - 1 - RITARDO_COMANDO)]
        for _ in range(self.passi):
            mujoco.mj_step(self.m, self.d)
            # A ogni passo, non a fine fotogramma: la chiusura dura 0,2 s e in
            # 7 passi le dita affondano gia' di millimetri.
            if self.da_agganciare is not None and self._tocca_con_due_dita(self.da_agganciare):
                self._aggancia(self.da_agganciare)

    # --- l'aggancio (vedi so101_caramelle: perche' non basta il contatto) ------

    def _tocca_con_due_dita(self, corpo: str) -> bool:
        b = self.m.body(corpo).id
        toccano = set()
        for c in self.d.contact[: self.d.ncon]:
            g1, g2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if b in (g1, g2):
                toccano.add(g2 if g1 == b else g1)
        return all(dito in toccano for dito in self._dita)

    def _aggancia(self, corpo: str) -> None:
        """Salda la caramella alle dita **dove sta adesso**: la posa relativa e'
        quella del momento, cosi' agganciare non la sposta."""
        i = self.m.equality(f"presa_{corpo}").id
        pg, qg = self.d.xpos[self._dita[0]], self.d.xquat[self._dita[0]]
        pc, qc = self.d.body(corpo).xpos, self.d.body(corpo).xquat
        inv = np.zeros(4)
        mujoco.mju_negQuat(inv, qg)
        rel_p, rel_q = np.zeros(3), np.zeros(4)
        mujoco.mju_rotVecQuat(rel_p, pc - pg, inv)
        mujoco.mju_mulQuat(rel_q, inv, qc)
        # weld: data = [ancora(3), posizione relativa(3), quaternione relativo(4), scala coppia]
        self.m.eq_data[i, 0:3] = 0
        self.m.eq_data[i, 3:6] = rel_p
        self.m.eq_data[i, 6:10] = rel_q
        self.m.eq_data[i, 10] = 1
        self.d.eq_active[i] = 1
        # La caramella non si comprime: il dito si ferma alla sua larghezza,
        # come il follower vero (in tenuta legge 5-6 sulle gialle, 8-10 sulle
        # galatine). La larghezza e' gia' la distanza fra i centri delle punte
        # (cosi' e' stata misurata): non si aggiungono i raggi.
        # Coi contatti morbidi di MuJoCo un servo cosi' forte su 5 g
        # affonda di 5-10 mm e chiude a zero (misurato).
        c = next(c for c in self.ep.caramelle if c.corpo == corpo)
        self.m.jnt_range[self._pinza, 0] = max(
            self._limite_pinza, pinza_su_caramella(self.k.m, c.larghezza)
        )
        self.agganciata, self.da_agganciare = corpo, None
        self.prese.add(corpo)

    def _sgancia(self) -> None:
        if self.agganciata is not None:
            self.d.eq_active[self.m.equality(f"presa_{self.agganciata}").id] = 0
            self.m.jnt_range[self._pinza, 0] = self._limite_pinza
        self.agganciata = self.da_agganciare = None

    def _lerobot(self, q6: np.ndarray) -> np.ndarray:
        g = a_lerobot(q6, self.rp)
        return np.array([g[n] for n in GIUNTI], dtype=np.float32)

    def _vai(self, q6: np.ndarray, durata: float) -> None:
        """Dal comando attuale a ``q6`` in ``durata`` secondi, profilo a minimo jerk."""
        da = self.comando.copy()
        n = max(1, int(round(durata * FPS)))
        for i in range(1, n + 1):
            self._fotogramma(da + (q6 - da) * _minimo_jerk(i / n))
        self.comando = q6.copy()

    def _aspetta(self, durata: float) -> None:
        for _ in range(int(round(durata * FPS))):
            self._fotogramma(self.comando)

    # --- i gesti ---------------------------------------------------------------

    def _u(self, intervallo: tuple[float, float]) -> float:
        return float(self.rng.uniform(*intervallo))

    def _riposo(self) -> np.ndarray:
        g = RIPOSO.copy()
        g[4] = self._u(RIPOSO_ROLL)
        return da_lerobot(dict(zip(GIUNTI, map(float, g), strict=True)), self.rp)

    def _postura(self, punto: np.ndarray, roll: float, pinza: float) -> np.ndarray:
        return self.k.postura(punto, roll, pinza)

    def _prendi(self, corpo: str, tipo: str, destinazione: str) -> None:
        tav = self.ep.tavolo
        pos = self.d.body(corpo).xpos.copy()
        asse = self.d.body(corpo).xmat.reshape(3, 3)[:, 0]
        aperta = self.k.pinza(self._u(PINZA_APERTA))
        chiusa = self.k.pinza(0.0)

        # Roll: calcolato sulla postura di presa, poi tutte le pose ricalcolate con lui.
        presa_xyz = np.array([pos[0], pos[1], pos[2] - 0.002])
        q = self._postura(presa_xyz, self.comando[4], aperta)
        roll = self.k.roll_per(q, asse, self.comando[4])

        sopra = presa_xyz + [0, 0, self._u(QUOTA_AVVICINA)]
        vicino = presa_xyz + [0, 0, self._u((0.015, 0.025))]
        # Prima in alto, sopra la caramella; solo poi giu'. Andando in diagonale
        # dal bicchiere le dita ne agganciavano il bordo (fisso, qui) e il braccio
        # restava li': 18 prese mancate su 40 episodi, tutte vicino ai bicchieri.
        in_alto = np.array([sopra[0], sopra[1], tav + self._u(QUOTA_VUOTO)])
        q_in_alto = self._postura(in_alto, roll, aperta)
        q_sopra = self._postura(sopra, roll, aperta)
        q_vicino = self._postura(vicino, roll, aperta)
        q_presa = self._postura(presa_xyz, roll, aperta)

        self._vai(q_in_alto, self._u((1.2, 2.2)))
        self._vai(q_sopra, self._u((0.7, 1.3)))
        self._vai(q_vicino, self._u((0.5, 1.0)))
        self._vai(q_presa, self._u((0.8, 1.6)))  # gli ultimi 2 cm, piano
        self._aspetta(self._u((0.1, 0.4)))
        self.da_agganciare = corpo
        self._vai(np.array([*q_presa[:5], chiusa]), self._u(CHIUSURA))
        self._aspetta(self._u((0.2, 0.4)))
        self.da_agganciare = None  # mancata: non deve scattare dopo, su un urto

        # Su e verso il bicchiere, in alto come fa Daniele, poi giu' a lasciare.
        b = self.ep.bicchieri[destinazione]
        su = presa_xyz.copy()
        su[2] = tav + self._u((0.06, 0.09))
        alto = np.array(
            [*(b[:2] + self.rng.uniform(-0.01, 0.01, 2)), tav + self._u(QUOTA_TRASPORTO)]
        )
        lascia = np.array([alto[0], alto[1], tav + min(self._u(QUOTA_RILASCIO), alto[2] - tav)])
        q_su = self._postura(su, roll, chiusa)
        q_alto = self._postura(alto, roll, chiusa)
        q_lascia = self._postura(lascia, roll, chiusa)
        self._vai(q_su, self._u((0.6, 1.0)))
        self._vai(q_alto, self._u((1.2, 2.2)))
        self._vai(q_lascia, self._u((0.5, 0.9)))
        self._sgancia()
        apre = self.k.pinza(self._u(PINZA_RILASCIO))
        self._vai(np.array([*q_lascia[:5], apre]), self._u(CHIUSURA))
        self._aspetta(self._u((0.3, 0.8)))
        # Via dal bicchiere, a vuoto, alla quota che si usa fra due prese.
        via = lascia.copy()
        via[2] = tav + self._u(QUOTA_VUOTO)
        self._vai(self._postura(via, roll, self.comando[5]), self._u((0.4, 0.8)))

    def esegui(self) -> bool:
        """L'episodio intero. Vero se ogni caramella e' nel suo bicchiere e la
        rossana non si e' mossa."""
        q0 = self._riposo()
        self.d.qpos[self.adr] = q0
        self.d.ctrl[:] = q0
        mujoco.mj_forward(self.m, self.d)
        for _ in range(int(0.5 / self.m.opt.timestep)):  # le caramelle si assestano
            mujoco.mj_step(self.m, self.d)
        self.comando = q0.copy()
        rossane = {
            c.corpo: self.d.body(c.corpo).xpos.copy()
            for c in self.ep.caramelle
            if c.destinazione is None
        }

        self._aspetta(self._u((0.5, 2.0)))
        da_fare = [c for c in self.ep.caramelle if c.destinazione is not None]
        while da_fare:
            # Prima la piu' a sinistra: +y, letto dalla scena adesso.
            c = max(da_fare, key=lambda c: self.d.body(c.corpo).xpos[1])
            self._prendi(c.corpo, c.tipo, c.destinazione)
            da_fare.remove(c)
        self._vai(self._riposo(), self._u((2.5, 4.0)))
        self._aspetta(self._u((1.0, 3.0)))
        return self.riuscito(rossane)

    def riuscito(self, rossane: dict[str, np.ndarray]) -> bool:
        """Ogni caramella nel suo bicchiere, la rossana ferma. I motivi di un
        fallimento restano in ``self.esito``: un errore sistematico renderebbe
        distorti gli episodi tenuti (per esempio, mai caramelle in una zona)."""
        h = MISURE["bicchiere"]["altezza"]
        self.esito: list[str] = []
        for c in self.ep.caramelle:
            p = self.d.body(c.corpo).xpos
            if c.destinazione is None:
                spostata = np.linalg.norm(p - rossane[c.corpo])
                if spostata > 0.01:
                    self.esito.append(f"{c.corpo} spostata di {spostata * 1000:.0f} mm")
                continue
            b = self.ep.bicchieri[c.destinazione]
            lontana = np.linalg.norm(p[:2] - b[:2])
            if lontana < MISURE["bicchiere"]["raggio_bordo"] and p[2] < b[2] + h:
                continue
            altro = next(k for k in self.ep.bicchieri if k != c.destinazione)
            if (
                np.linalg.norm(p[:2] - self.ep.bicchieri[altro][:2])
                < MISURE["bicchiere"]["raggio_bordo"]
            ):
                self.esito.append(f"{c.corpo} nel bicchiere sbagliato")
            elif p[2] < self.ep.tavolo - 0.02:
                self.esito.append(f"{c.corpo} caduta dal tavolo")
            elif lontana < 0.06:
                self.esito.append(
                    f"{c.corpo} sul bordo o fuori, a {lontana * 1000:.0f} mm dal centro, quota "
                    f"{(p[2] - b[2]) * 1000:.0f} mm"
                )
            else:
                self.esito.append(
                    f"{c.corpo} rimasta a {np.round(p[:2] * 100, 1)} cm"
                    f" ({'presa' if c.corpo in self.prese else 'mai agganciata'})"
                )
        return not self.esito
