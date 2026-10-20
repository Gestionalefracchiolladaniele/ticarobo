"""Server MCP: il robot esposto come strumenti per un agent.

Tre tool leggono (``get_scene``, ``get_audit_log``), tre muovono
(``move_to_pose``, ``pick``, ``place``), due governano l'autorizzazione
(``arm_motion``, ``disarm_motion``), uno reimposta (``reset_scene``).

**Il principio e' D2: ogni tool passa dalla verifica di sicurezza prima di
muovere, e in caso di rifiuto restituisce il motivo in forma leggibile.** Non
e' cortesia verso l'utente: e' la condizione perche' l'agent si corregga da
solo. "Rifiutato" lascia un LLM a indovinare; "z=0.79 sotto il piano del
tavolo (z=0.80)" gli dice esattamente cosa cambiare.

Per lo stesso motivo le risposte sono testo strutturato e non JSON crudo: sono
scritte per essere *lette* da un modello, con le unita' di misura esplicite.

Ai controlli geometrici di ``safety/`` se ne affiancano due di altra natura,
in ``audit.py``: il movimento va autorizzato e scade, e ogni chiamata resta
registrata. Rispondono a domande che la geometria non copre — chi ha
autorizzato, e cosa e' stato eseguito davvero.

Non c'e' un tool per le immagini: finche' non c'e' un modello di percezione
che le legga (Fase 3, su Modal), ``get_scene`` da' le stesse informazioni in
modo esatto e senza costo. Per guardare la scena da umano c'e'
``scripts/viewer.py``.

    uv run python -m robotica.mcp.server            # stdio
    uv run python -m robotica.mcp.server --http     # HTTP, scena che resta viva
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import numpy as np
from mcp.server.mcpserver import MCPServer

from robotica.mcp.audit import AuditLog, MotionGate
from robotica.mcp.session import RobotSession
from robotica.sim.env import GREEN_ZONE_HALF, table_surface_z

server = MCPServer(
    name="robotica",
    instructions=(
        "Braccio robotico Panda in simulazione, su un tavolo con oggetti "
        "afferrabili e una zona di posa verde. Le coordinate sono in metri nel "
        "frame mondo. Chiama get_scene prima di muovere: le posizioni degli "
        "oggetti cambiano dopo ogni pick e place. Ogni comando viene verificato "
        "prima dell'esecuzione; se un tool rifiuta, il messaggio contiene il "
        "motivo e va usato per correggere il comando."
    ),
)

_session = RobotSession()
_audit = AuditLog()
_gate = MotionGate()


def _logged(fn: Callable[..., str]) -> Callable[..., str]:
    """Registra la chiamata e il suo esito.

    L'esito si deduce dal prefisso della risposta, perche' i tool restituiscono
    testo e non eccezioni: un agent legge meglio "RIFIUTATO: ..." di uno stack
    trace. Come decoratore, nessun tool puo' dimenticarsi di registrare.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> str:
        shown = ", ".join(
            [f"{a!r}" if isinstance(a, str) else f"{a}" for a in args]
            + [f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}" for k, v in kwargs.items()]
        )
        try:
            out = fn(*args, **kwargs)
        except Exception as exc:
            _audit.record(fn.__name__, shown, "fallito", f"eccezione: {exc}")
            raise

        head = out.split("\n", 1)[0]
        outcome = (
            "rifiutato" if head.startswith("RIFIUTATO")
            else "fallito" if head.startswith("FALLITO")
            else "ok"
        )
        _audit.record(fn.__name__, shown, outcome, head)
        return out

    return wrapper


def _requires_motion(fn: Callable[..., str]) -> Callable[..., str]:
    """Blocca il tool se il movimento non e' autorizzato.

    Va **sotto** ``_logged`` nella pila dei decoratori, cosi' che anche il
    rifiuto per mancata autorizzazione finisca nel registro.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> str:
        if not _gate.armed:
            return _gate.refuse_reason()
        return fn(*args, **kwargs)

    return wrapper


@server.tool(
    description=(
        "Stato della scena: oggetti afferrabili con le loro coordinate, zona di "
        "posa, posizione della pinza, limiti del workspace. Da chiamare prima di "
        "muovere e dopo ogni manipolazione."
    )
)
@_logged
def get_scene() -> str:
    """Descrive la scena in forma leggibile da un agent."""
    env = _session.env
    robot = _session.robot
    ws = robot.safety.workspace

    lines = ["SCENA", ""]

    lines.append("Oggetti afferrabili:")
    for name in env.object_names():
        p = env.object_pos(name)
        mano = "  [IN MANO]" if _session.held_object == name else ""
        zona = "  [nella zona verde]" if env.object_in_zone(name) else ""
        lines.append(f"  {name}: x={p[0]:.3f} y={p[1]:.3f} z={p[2]:.3f}{mano}{zona}")

    zone = env.green_zone_center()
    lines += [
        "",
        f"Zona di posa (verde): centro x={zone[0]:.3f} y={zone[1]:.3f} "
        f"z={zone[2]:.3f}, lato {GREEN_ZONE_HALF * 2:.2f} m",
        f"Piano del tavolo: z={table_surface_z():.3f}",
        "",
        f"Pinza: {robot.tcp_pos.round(3).tolist()}, "
        f"apertura {robot.gripper_width() * 1000:.0f} mm",
        f"In mano: {_session.held_object or 'niente'}",
        "",
        "Workspace ammesso (metri):",
        f"  x [{ws.x[0]:.2f}, {ws.x[1]:.2f}]  "
        f"y [{ws.y[0]:.2f}, {ws.y[1]:.2f}]  z [{ws.z[0]:.2f}, {ws.z[1]:.2f}]",
    ]
    return "\n".join(lines)


@server.tool(
    description=(
        "Porta la pinza in una posizione (metri, frame mondo), tenendola rivolta "
        "verso il basso. Verifica workspace e collisione col tavolo prima di "
        "muovere. Per afferrare usa pick, non questo. Richiede arm_motion."
    )
)
@_logged
@_requires_motion
def move_to_pose(x: float, y: float, z: float) -> str:
    """Muove il TCP su una posizione, se la verifica lo consente."""
    target = np.array([x, y, z], dtype=float)
    robot = _session.robot

    result = robot.move_to_pos(target)
    if not result:
        return (
            f"RIFIUTATO: movimento verso [{x:.3f}, {y:.3f}, {z:.3f}] non eseguito.\n"
            f"{result.reason}\n"
            f"La pinza e' rimasta in {robot.tcp_pos.round(3).tolist()}."
        )

    return (
        f"OK: pinza in {robot.tcp_pos.round(3).tolist()}, "
        f"errore {result.pos_error * 1000:.1f} mm."
    )


@server.tool(
    description=(
        "Afferra un oggetto per nome: si posiziona sopra, scende, stringe e "
        "risolleva. Verifica che l'oggetto sia davvero in mano al termine, e "
        "fallisce dicendolo se la presa non ha tenuto. Richiede arm_motion."
    )
)
@_logged
@_requires_motion
def pick(object_name: str) -> str:
    """Prende un oggetto, verificando l'esito invece di darlo per fatto."""
    env = _session.env
    robot = _session.robot

    if object_name not in env.object_names():
        return (
            f"RIFIUTATO: nessun oggetto '{object_name}' nella scena.\n"
            f"Disponibili: {', '.join(env.object_names())}."
        )

    if _session.held_object:
        return (
            f"RIFIUTATO: la pinza tiene gia' {_session.held_object}.\n"
            f"Usa place per posarlo prima di prendere {object_name}."
        )

    result = robot.pick(object_name)
    if not result:
        return f"FALLITO: {object_name} non preso.\n{result.reason}"

    _session.held_object = object_name
    pos = env.object_pos(object_name)
    return (
        f"OK: {object_name} in mano, sollevato a z={pos[2]:.3f}, "
        f"pinza chiusa su {robot.gripper_width() * 1000:.0f} mm."
    )


@server.tool(
    description=(
        "Posa l'oggetto che si ha in mano in una posizione (metri). La z e' "
        "opzionale: se omessa si posa sul piano del tavolo, che e' quasi sempre "
        "cio' che si vuole. Richiede arm_motion."
    )
)
@_logged
@_requires_motion
def place(x: float, y: float, z: float | None = None) -> str:
    """Posa l'oggetto in mano, verificando dove e' finito davvero."""
    env = _session.env
    robot = _session.robot

    if not _session.held_object:
        return (
            "RIFIUTATO: la pinza non tiene niente.\n"
            "Usa pick per afferrare un oggetto prima di posarlo."
        )

    held = _session.held_object
    target = np.array([x, y, table_surface_z() if z is None else z], dtype=float)

    result = robot.place(target, held)
    if not result:
        return (
            f"RIFIUTATO: {held} non posato in [{target[0]:.3f}, {target[1]:.3f}, "
            f"{target[2]:.3f}].\n{result.reason}\n{held} e' ancora in mano."
        )

    _session.held_object = None
    _session._settle()

    final = env.object_pos(held)
    scarto = float(np.linalg.norm(final[:2] - target[:2]))
    dentro = " nella zona verde" if env.object_in_zone(held) else ""
    return (
        f"OK: {held} posato in {final.round(3).tolist()}{dentro}, "
        f"a {scarto * 1000:.0f} mm dal punto richiesto."
    )


@server.tool(
    description=(
        "Autorizza il movimento del braccio per un breve periodo, dichiarando "
        "cosa si sta per fare. Va chiamato prima di move_to_pose, pick e place. "
        "Le letture come get_scene non lo richiedono."
    )
)
@_logged
def arm_motion(reason: str) -> str:
    """Abilita il movimento, registrando il motivo dichiarato."""
    if not reason.strip():
        return (
            "RIFIUTATO: serve un motivo non vuoto.\n"
            "Esempio: arm_motion(reason='sposto il cubo rosso nella zona verde')."
        )
    seconds = _gate.arm(reason.strip())
    return (
        f"OK: movimento autorizzato per {seconds:.0f} secondi.\n"
        f"Motivo registrato: {reason.strip()}"
    )


@server.tool(
    description=(
        "Revoca subito l'autorizzazione al movimento. Da usare a compito finito "
        "o se qualcosa non sta andando come previsto."
    )
)
@_logged
def disarm_motion() -> str:
    """Revoca l'autorizzazione. Non richiede condizioni, di proposito."""
    _gate.disarm()
    return "OK: movimento non piu' autorizzato. Il braccio non si muovera' finche' non si riarma."


@server.tool(
    description=(
        "Registro delle chiamate di questa sessione, con esito e motivo dei "
        "rifiuti. Utile per capire cosa e' stato eseguito e dove si e' fallito."
    )
)
def get_audit_log(limit: int = 20) -> str:
    """Le ultime chiamate, in ordine. Non si registra da solo, per non sporcare."""
    entries = _audit.recent(limit)
    if not entries:
        return "Registro vuoto: nessuna chiamata finora."

    counts = ", ".join(f"{k}: {v}" for k, v in sorted(_audit.counts().items()))
    body = "\n".join(str(e) for e in entries)
    return (
        f"REGISTRO ({len(_audit)} chiamate totali — {counts})\n"
        f"Stato: {_gate.status()}\n\n{body}"
    )


@server.tool(description="Riporta la scena allo stato iniziale: oggetti e braccio come all'avvio.")
@_logged
def reset_scene() -> str:
    """Reset della scena, utile fra un tentativo e l'altro."""
    _session.reset()
    return "OK: scena reimpostata.\n\n" + get_scene()


HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8765
HTTP_PATH = "/mcp"


def main() -> None:
    """Avvia il server. Su stdio per default, su HTTP con ``--http``.

    **stdio** e' il trasporto piu' semplice: l'agent avvia il server come
    sottoprocesso e ci parla. Ma nasce e muore con la sessione, quindi la scena
    riparte da zero a ogni comando e non c'e' modo di tenere aperta una finestra.

    **HTTP** tiene il server acceso come processo a se': la scena sopravvive fra
    un comando e l'altro e la finestra 3D resta aperta mentre l'agent lavora, il
    che rende visibile cio' che sta facendo. E' anche piu' vicino al robot reale,
    che e' un processo sempre acceso a cui ci si collega, non qualcosa che si
    riavvia a ogni frase.

    L'ascolto e' su ``0.0.0.0`` e non su ``127.0.0.1`` perche' il server gira in
    WSL e il client sta su Windows: con la rete in NAT, un socket legato al solo
    localhost non sarebbe raggiungibile da fuori la distro.
    """
    import os
    import sys

    if "--http" in sys.argv:
        # Con la finestra richiesta si carica **subito**, invece di aspettare il
        # primo tool. La sessione e' pigra di proposito (un client che elenca i
        # tool non deve pagare venti secondi di robosuite), ma chi ha chiesto di
        # guardare vuole la scena davanti agli occhi prima di comandarla, non
        # dopo il primo comando.
        if os.environ.get("ROBOTICA_FINESTRA") == "1":
            print("Carico la scena e apro la finestra 3D...", flush=True)
            _session.env  # noqa: B018  — l'accesso costruisce la scena
            print("Finestra aperta.", flush=True)

        print(
            f"Server MCP su http://{HTTP_HOST}:{HTTP_PORT}{HTTP_PATH}\n"
            "La scena resta viva finche' questo processo gira. Ctrl-C per fermarlo.",
            flush=True,
        )
        server.run(
            transport="streamable-http",
            host=HTTP_HOST,
            port=HTTP_PORT,
            streamable_http_path=HTTP_PATH,
        )
    else:
        server.run()


if __name__ == "__main__":
    main()
