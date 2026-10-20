"""Stato del robot dietro al server MCP.

Il server e' senza stato per protocollo, ma un robot no: ha una scena
caricata, una posa corrente, un oggetto in mano. Questo modulo tiene quella
continuita' in un punto solo, cosi' che i tool restino funzioni brevi.

La simulazione si costruisce **pigramente**, al primo tool che serve davvero:
caricare Panda e l'arena costa qualche secondo, e un client MCP che si limiti a
elencare i tool non deve pagarlo.

Con ``ROBOTICA_FINESTRA=1`` si apre anche una finestra 3D, e si vede il robot
muoversi mentre l'agent lo comanda. E' spenta per default: nei test e sulle
macchine senza schermo aprirla sarebbe inutile o dannoso.
"""

from __future__ import annotations

import os

import numpy as np

from robotica.sim.env import WorkspaceEnv
from robotica.sim.robot import OSCRobot


class RobotSession:
    """Scena, robot e oggetto in mano: lo stato che i tool condividono."""

    def __init__(self, robot_name: str = "Panda") -> None:
        import threading

        self.robot_name = robot_name
        self._env: WorkspaceEnv | None = None
        self._robot: OSCRobot | None = None
        self.held_object: str | None = None
        self._finestra = None
        # Protegge i dati di MuJoCo: la fisica li modifica, il disegno li legge.
        # Senza, MuJoCo si ferma con "mj_copyDataVisual: attempting to copy
        # mjData while stack is in use". Vedi _avvia_battito.
        self._lucchetto = threading.Lock()

    # --- accesso pigro -------------------------------------------------------

    @property
    def env(self) -> WorkspaceEnv:
        if self._env is None:
            self._boot()
        assert self._env is not None
        return self._env

    @property
    def robot(self) -> OSCRobot:
        if self._robot is None:
            self._boot()
        assert self._robot is not None
        return self._robot

    def _boot(self) -> None:
        """Costruisce scena e controller. Chiamata una volta sola."""
        from robosuite.controllers import load_composite_controller_config

        config = load_composite_controller_config(controller="BASIC", robot=self.robot_name)
        self._env = WorkspaceEnv(robots=self.robot_name, controller_configs=config)
        self._env.reset()
        self._robot = OSCRobot(self._env)
        self.held_object = None
        self._apri_finestra()
        self._settle()

    def _apri_finestra(self) -> None:
        """Apre la finestra 3D, se richiesta con ``ROBOTICA_FINESTRA=1``.

        La finestra e' **passiva** (``launch_passive``): si aggiorna da sola e
        resta aperta mentre il server aspetta il prossimo comando dell'agent.
        Con ``env.render()`` resterebbe congelata, perche' quello disegna solo
        quando viene chiamato, e fra un tool e l'altro non lo chiama nessuno.

        Il ``sync()`` a ogni passo di fisica serve a vedere il **movimento**:
        senza, si vedrebbe solo il risultato finale di ogni comando.

        Se la finestra non si apre (nessuno schermo, server X spento) il server
        prosegue senza: comandare il robot deve restare possibile anche al buio.
        """
        if os.environ.get("ROBOTICA_FINESTRA") != "1" or self._env is None:
            return

        try:
            import mujoco.viewer

            self._finestra = mujoco.viewer.launch_passive(
                self._env.sim.model._model,
                self._env.sim.data._data,
                show_left_ui=False,
                show_right_ui=False,
            )
        except Exception as exc:  # senza schermo si lavora lo stesso
            print(f"[robotica] finestra 3D non aperta: {exc}", flush=True)
            return

        passo = self._env.step

        def passo_con_disegno(action):
            with self._lucchetto:
                out = passo(action)
                if self._finestra is not None and self._finestra.is_running():
                    self._finestra.sync()
            return out

        self._env.step = passo_con_disegno

        # Il battito parte **dopo** la sostituzione di step: prima, la fisica
        # avanzerebbe ancora senza lucchetto e il battito potrebbe disegnarci
        # sopra.
        self._avvia_battito()

    def _avvia_battito(self) -> None:
        """Ridisegna la finestra anche mentre il server non fa nulla.

        Fra un tool e l'altro il server e' fermo ad aspettare la prossima
        richiesta dell'agent: nessuno chiama ``sync()``, e la finestra si
        congela — il sistema puo' arrivare a darla per bloccata. Un battito
        lento la tiene viva e permette di ruotare la vista col mouse mentre si
        aspetta il comando successivo.

        Venti volte al secondo bastano: non e' il battito a mostrare il
        movimento (quello lo fa il ``sync()`` a ogni passo di fisica), gli
        serve solo non far morire la finestra. Il thread e' ``daemon``, cosi'
        non tiene in piedi il processo quando si chiude il server.

        **Il lucchetto non e' facoltativo.** ``sync()`` copia i dati di MuJoCo
        per disegnarli: farlo mentre il thread principale fa avanzare la fisica
        significa leggere una struttura che sta cambiando sotto, e MuJoCo si
        ferma con ``mj_copyDataVisual: attempting to copy mjData while stack is
        in use``. Il battito disegna solo quando la fisica non sta girando.
        """
        import threading
        import time

        def battito() -> None:
            while True:
                finestra = self._finestra
                if finestra is None or not finestra.is_running():
                    return
                try:
                    # acquire=False: se la fisica sta girando si salta questo
                    # giro invece di aspettare. Il passo disegna gia' da se',
                    # e un battito in coda rallenterebbe il movimento.
                    if self._lucchetto.acquire(blocking=False):
                        try:
                            finestra.sync()
                        finally:
                            self._lucchetto.release()
                except Exception:  # la finestra puo' chiudersi in qualsiasi momento
                    return
                time.sleep(0.05)

        threading.Thread(target=battito, daemon=True, name="finestra").start()

    def reset(self) -> None:
        """Riporta la scena allo stato iniziale."""
        self.env.reset()
        self._robot = OSCRobot(self.env)
        self.held_object = None
        self._settle()

    def _settle(self, steps: int = 40) -> None:
        """Lascia assestare gli oggetti.

        Senza questo, i cubi vengono letti mentre cadono ancora di quei
        millimetri che li separano dal piano, e ``get_scene`` riporta quote
        che cambiano fra una chiamata e l'altra.
        """
        action = np.zeros(self.env.action_dim)
        action[-1] = self.robot._gripper
        for _ in range(steps):
            self.env.step(action)
