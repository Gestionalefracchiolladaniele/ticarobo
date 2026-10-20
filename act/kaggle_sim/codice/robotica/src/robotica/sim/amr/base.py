"""Comandare la base olonoma: da una velocita' voluta al movimento vero.

**Perche' serve uno strato qui.** Il modello TidyBot comanda i tre giunti
``joint_x, joint_y, joint_th`` con attuatori di **posizione** e guadagni
altissimi (``kp=1000000``). Scriverci dentro la posizione di arrivo funziona —
troppo bene: la base ci salta sopra in pochi passi di fisica, con accelerazioni
che nessun robot ha, e nel salto **attraversa gli ostacoli**, perche' il
rilevamento di collisione lavora per passi discreti.

Il risultato sarebbe un robot che arriva sempre. Cioe' un flag che dice "va
tutto bene" (D2): arriva perche' bara, non perche' naviga.

Quindi il bersaglio comandato **insegue** quello voluto a velocita' limitata.
Non e' finta fedelta': e' esattamente cio' che fa un controller locale vero,
che emette ``(vx, vy, vth)`` e non un teletrasporto. E i limiti sono numeri
dichiarati, quindi discutibili e misurabili.

**Olonoma vuol dire una cosa precisa.** La base puo' scegliere
indipendentemente le tre velocita': puo' traslare di lato senza girare, o
girare stando ferma. Una base differenziale (due ruote, tipo Stretch) non puo':
per andare a sinistra deve ruotare, avanzare, riruotare. Per questo il comando
qui e' un vettore di tre numeri e non ha vincoli fra loro -- e per questo
Nav2 comanda proprio in ``(vx, vy, vth)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# Limiti di velocita' e accelerazione della base.
#
# **Presi dal codice degli autori di TidyBot++**, non scelti a occhio: il loro
# ``mujoco_env.py`` usa velocita' ``[0.5, 0.5, 3.14]`` e accelerazioni
# ``[0.5, 0.5, 2.36]``. Sono i numeri con cui il robot **vero** e' stato
# guidato, quindi valgono piu' di una stima ragionevole.
#
# Il primo tentativo era 0.8 m/s e 1.2 m/s^2 -- velocita' del 60% piu' alta e
# accelerazione piu' che doppia. Nessuno dei due sarebbe stato *sbagliato* in
# assoluto, ma non sarebbero stati **i limiti di questo robot**: la stessa
# differenza fra un numero plausibile e una misura.
#
# La velocita' angolare resta piu' bassa della loro (3.14 rad/s = mezzo giro
# al secondo) perche' qui il robot naviga fra scaffali, non insegue una
# teleoperazione: girare in fretta non serve, e un robot che frulla e'
# illeggibile a guardarsi.
MAX_LIN_SPEED = 0.5  # m/s   -- come tidybot2/mujoco_env.py
MAX_ANG_SPEED = 1.5  # rad/s -- piu' prudente dei 3.14 loro
MAX_LIN_ACCEL = 0.5  # m/s^2 -- come tidybot2/mujoco_env.py
MAX_ANG_ACCEL = 2.36  # rad/s^2 -- come tidybot2/mujoco_env.py

# Quanto il bersaglio comandato puo' distanziarsi dal robot prima che
# l'anti-windup lo trattenga. Vedi la nota lunga in ``step``: e' una quota
# fissa e **non derivata dalla velocita'**, perche' lo scarto serve a
# produrre la forza, non a misurare il moto.
MAX_TRACKING_ERROR = 0.12  # m
MAX_TRACKING_ANGLE = 0.35  # rad

# Ogni quanto il controller ricalcola. 20 Hz e' la cadenza tipica di un local
# controller Nav2 (DWB gira a 20, MPPI a 30): il valore non e' inventato, e
# tenerlo realistico conta perche' il ritardo di reazione e' cio' che decide
# se il robot evita l'ostacolo o ci arriva addosso.
CONTROL_HZ = 20.0


def _wrap(angle: float) -> float:
    """Riporta un angolo in (-pi, pi].

    Serve ogni volta che si sottraggono due angoli: senza, un robot a +179
    gradi che deve andare a -179 crede di dover girare di 358 gradi invece che
    di 2, e fa un giro completo. E' un bug classico, e silenzioso.
    """
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class BaseState:
    """Dove sta la base e come si muove. Tutto nel frame mondo.

    ``theta`` e' l'orientamento attorno a z. Il nome dei campi porta il frame
    perche' e' la convenzione del progetto: gli errori di frame sono il bug
    numero uno.
    """

    xy_world: np.ndarray
    theta_world: float
    v_xy_world: np.ndarray
    v_theta: float

    @property
    def pose_world(self) -> np.ndarray:
        """La posa nel piano come ``(x, y, theta)``."""
        return np.array([self.xy_world[0], self.xy_world[1], self.theta_world])


class HolonomicBase:
    """La base del TidyBot, comandata in velocita'.

    Non tiene uno stato proprio della posizione: lo **rilegge dalla scena** a
    ogni chiamata (convenzione del progetto). L'unico stato tenuto qui e' il
    bersaglio di posizione inseguito dagli attuatori, che e' un fatto del
    controller, non del mondo.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._m = model
        self._d = data
        self._jid = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ("joint_x", "joint_y", "joint_th")
        }
        missing = [n for n, i in self._jid.items() if i < 0]
        if missing:
            raise ValueError(
                f"giunti della base non trovati nel modello: {missing}. "
                "La scena include base.xml del menagerie?"
            )
        self._qadr = np.array(
            [model.jnt_qposadr[self._jid[n]] for n in ("joint_x", "joint_y", "joint_th")]
        )
        self._vadr = np.array(
            [model.jnt_dofadr[self._jid[n]] for n in ("joint_x", "joint_y", "joint_th")]
        )
        self._aid = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                for n in ("joint_x", "joint_y", "joint_th")
            ]
        )
        # Il bersaglio parte dove sta la base: senza questo, al primo step gli
        # attuatori tirerebbero verso l'origine e il robot partirebbe da solo.
        self._target = self.state().pose_world.copy()
        self._cmd = np.zeros(3)
        # La velocita' **del bersaglio**, che e' stato del controller, non del
        # mondo. Tenerla separata dalla velocita' misurata non e' pedanteria:
        # vedi la nota in ``step`` -- confonderle costa un fattore 5,7.
        self._v_target = np.zeros(3)

        # --- per l'arresto di sicurezza (vedi ``step``) ---------------------
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        propri = {bid}
        for b in range(model.nbody):
            p = b
            while p > 0:
                if p == bid:
                    propri.add(b)
                    break
                p = model.body_parentid[p]
        self._own_geoms = {
            g for g in range(model.ngeom) if model.geom_bodyid[g] in propri
        }
        self._floor_geoms = {
            g
            for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(
                ("pavimento", "stazione_", "bordo_")
            )
        }
        self._stop_on_contact = True
        self.contacts = 0
        # Quanti step di fisica per un tick di controllo. Si legge dal modello
        # perche' il timestep e' una proprieta' della scena, non una costante.
        self._sub_steps = max(1, int(round(1.0 / (CONTROL_HZ * model.opt.timestep))))

    # --- lettura ------------------------------------------------------------

    def state(self) -> BaseState:
        """Dove sta e come si muove la base, letto dalla scena."""
        q = self._d.qpos[self._qadr]
        v = self._d.qvel[self._vadr]
        return BaseState(
            xy_world=np.array([q[0], q[1]]),
            theta_world=_wrap(float(q[2])),
            v_xy_world=np.array([v[0], v[1]]),
            v_theta=float(v[2]),
        )

    @property
    def control_dt(self) -> float:
        """Durata di un tick di controllo, in secondi."""
        return self._sub_steps * self._m.opt.timestep

    # --- comando ------------------------------------------------------------

    def teleport(self, x: float, y: float, theta: float = 0.0) -> None:
        """Mette la base in un punto, azzerando la velocita'.

        **Solo per preparare una scena**, mai durante una navigazione: un robot
        che si teletrasporta non dimostra niente. Esiste perche' posizionare la
        scena iniziale scrivendo ``qpos`` e' l'unico modo pulito, ed e' la
        stessa scelta del test del tracker in D19.
        """
        self._d.qpos[self._qadr] = [x, y, theta]
        self._d.qvel[self._vadr] = 0.0
        self._target[:] = [x, y, theta]
        self._cmd[:] = 0.0
        self._v_target[:] = 0.0
        self._d.ctrl[self._aid] = self._target
        mujoco.mj_forward(self._m, self._d)

    def command_velocity(self, vx: float, vy: float, vth: float) -> None:
        """Chiede una velocita' nel frame **mondo**, in m/s e rad/s.

        La firma e' quella di Nav2 (``cmd_vel``), tranne che li' le velocita'
        sono nel frame del robot. Qui sono nel mondo perche' il planner
        ragiona nel mondo e la base e' olonoma, quindi la conversione sarebbe
        solo una rotazione in piu' da sbagliare. ``command_velocity_body``
        esiste per quando serve l'altra convenzione.

        La velocita' viene **saturata**, non rifiutata: un controller che
        chiede troppo e' normale, e restituire un errore obbligherebbe ogni
        chiamante a saturare per conto suo.
        """
        v = np.array([vx, vy], dtype=float)
        speed = float(np.linalg.norm(v))
        if speed > MAX_LIN_SPEED:
            v *= MAX_LIN_SPEED / speed
        self._cmd[:2] = v
        self._cmd[2] = float(np.clip(vth, -MAX_ANG_SPEED, MAX_ANG_SPEED))

    def command_velocity_body(self, vx: float, vy: float, vth: float) -> None:
        """Come sopra, ma le velocita' sono nel frame del robot.

        ``vx`` e' "avanti" rispetto a dove il robot guarda. E' la convenzione
        di ``cmd_vel`` in ROS 2, quindi e' la firma che servirebbe il giorno
        che il controller vero venisse da li'.
        """
        th = self.state().theta_world
        c, s = np.cos(th), np.sin(th)
        self.command_velocity(c * vx - s * vy, s * vx + c * vy, vth)

    def stop(self) -> None:
        """Ferma la base. Non e' un freno: azzera la velocita' comandata."""
        self._cmd[:] = 0.0

    def step(self) -> None:
        """Un tick di controllo: applica la velocita' e avanza la fisica.

        **La base si comanda in velocita', scrivendo ``qvel``.** Non e' la
        strada piu' ovvia -- il modello ufficiale ha attuatori di posizione --
        ed e' stata scelta dopo averli provati, misurando cosa costavano.

        **Cosa e' costato il controllo in posizione.** Gli attuatori del
        TidyBot hanno ``kp=1000000``, un guadagno enorme che insegue il
        bersaglio con forza praticamente illimitata. Per usarlo serve
        generare il bersaglio istante per istante (loro usano Ruckig), e
        ogni pezzo di quel meccanismo ha prodotto un guasto proprio:

        - integrando dalla velocita' **misurata**, retroazione che si mangia
          da sola: 0,0879 m/s su 0,5 chiesti, fattore 5,7;
        - contro un muro, **122.426 N** su 60 kg: il contatto perde e il
          robot penetra di 94 mm;
        - con l'anti-windup tarato sulla velocita', 0,048 m/s su 0,4;
        - con l'anti-windup tarato sulla forza, 0,145 m/s su 0,5.

        Quattro tentativi, quattro punti fissi coerenti e sbagliati. Il
        segnale non era che mancasse la taratura giusta: era che **quel
        meccanismo non serve a questo lavoro**. ``kp=1000000`` esiste per la
        teleoperazione di un braccio, non per far navigare una base.

        **Perche' scrivere ``qvel`` e' anche piu' onesto.** Su un robot vero
        i motori non li comanda questo codice: li comanda il firmware del
        produttore, e l'interfaccia che riceve e' ``cmd_vel`` -- una
        velocita'. Nav2 manda ``(vx, vy, vth)`` e il driver la esegue. Il
        controllo di posizione a guadagno altissimo era un pezzo di
        simulazione da tarare che **sul robot vero si butta**: il file con
        il transfer piu' basso del progetto, e quello che costava piu' tempo.

        **Il prezzo, dichiarato.** Una base comandata in velocita' non ha
        dinamica di contatto: se le si dice di andare avanti, va avanti anche
        contro un muro. Il problema pero' c'era gia' con gli attuatori (i
        122 kN vincevano comunque), e la risposta non cambia: l'urto si
        **misura** con ``CollisionMonitor`` invece di impedirlo con una
        fisica ritarata a occhio. Un AMR che tocca un muro ha gia' fallito.

        Resta il limite di accelerazione, che e' vero anche sul robot reale:
        un corpo da 60 kg non cambia velocita' istantaneamente, e i valori
        sono quelli dichiarati da TidyBot++.
        """
        dt = self.control_dt

        # rampa di accelerazione sul comando: e' l'unico pezzo del vecchio
        # meccanismo che sopravvive, perche' e' l'unico che descrive il
        # robot vero invece della simulazione
        dv = self._cmd[:2] - self._v_target[:2]
        dv_max = MAX_LIN_ACCEL * dt
        n = float(np.linalg.norm(dv))
        if n > dv_max:
            dv *= dv_max / n
        self._v_target[:2] += dv
        self._v_target[2] += float(
            np.clip(self._cmd[2] - self._v_target[2], -MAX_ANG_ACCEL * dt, MAX_ANG_ACCEL * dt)
        )

        # La velocita' si impone a ogni sotto-passo, non una volta sola.
        # **E' la lezione di ``belt_speed`` in D17**, applicata prima di
        # sbagliare: scrivere ``qvel`` una volta e lasciar girare la fisica
        # 25 volte significa che l'attrito frena il corpo nei 24 passi
        # rimanenti, e la velocita' vera risulta una frazione di quella
        # chiesta -- li' il fattore era 6,64.
        #
        # Gli attuatori di posizione vengono tenuti allineati al robot cosi'
        # non tirano: il loro bersaglio segue la posa invece di guidarla.
        for _ in range(self._sub_steps):
            self._d.qvel[self._vadr] = self._v_target
            self._d.ctrl[self._aid] = self._d.qpos[self._qadr]
            mujoco.mj_step(self._m, self._d)

        # --- l'arresto di sicurezza -----------------------------------------
        #
        # Una base comandata in velocita' attraversa i muri: la fisica del
        # contatto non ha voce in capitolo. Il rimedio **non e' ritarare il
        # contatto** -- gia' provato, non regge contro ``kp=1000000`` -- ma
        # fermare il robot quando qualcosa lo tocca.
        #
        # **Ed e' quello che fa un AMR vero.** Il LiDAR 2D di un AMR
        # industriale non e' solo un sensore: e' un *safety laser scanner*
        # certificato (SICK, Hokuyo, Pilz) che taglia l'alimentazione ai
        # motori **via hardware**, senza passare dal software di
        # navigazione. La letteratura riporta 70-90% di collisioni in meno
        # rispetto a un mezzo non protetto.
        #
        # Quindi questo blocco non e' una pezza di simulazione: e' il
        # modello del pezzo di sicurezza che sul robot vero esiste davvero,
        # ed e' l'unico livello che ha il diritto di fermare i motori senza
        # chiedere a nessuno.
        if self._stop_on_contact and self._contact_with_world():
            self._d.qvel[self._vadr] = 0.0
            self._v_target[:] = 0.0
            self._cmd[:] = 0.0
            self.contacts += 1

    def _contact_with_world(self) -> bool:
        """C'e' un contatto fra il robot e qualcosa che non sia il pavimento?

        Si guardano i contatti che la fisica ha **gia'** calcolato: non costa
        niente in piu', e la risposta e' quella del motore.
        """
        for i in range(self._d.ncon):
            c = self._d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            mine1, mine2 = g1 in self._own_geoms, g2 in self._own_geoms
            if mine1 == mine2:
                continue  # entrambi miei, o nessuno dei due
            altro = g2 if mine1 else g1
            if altro in self._floor_geoms:
                continue
            return True
        return False

    def settle(self, seconds: float = 0.5) -> None:
        """Lascia assestare la scena tenendo la base ferma dov'e'."""
        self.stop()
        for _ in range(int(seconds / self.control_dt)):
            self.step()
