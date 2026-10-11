"""Esecutore del SO-101: riceve pose gia' approvate da `safety/` e le esegue con limiti suoi.

    .venv/bin/python esecutore.py --porta /dev/ttyACM0   sul ferro
    .venv/bin/python esecutore.py --prova                braccio finto, nessuna porta

Lo lancia `robotica/scripts/gesto_vero.py`: il progetto calcola il gesto, lo fa
approvare a `safety/gesti.py`, lo converte in gradi LeRobot e lo scrive qui su
stdin. Due processi perche' numpy e' 2 qui e 1 la' (D22, D23).

**Difesa in profondita'.** `safety/` del progetto conosce il **modello**; questo
file conosce il **ferro**. Un gesto puo' essere giusto per il modello e
impossibile per il braccio: "carta" ruota il polso a +191,5 gradi LeRobot, il
nostro arriva a +101,5. Come il PLC che ha i suoi limiti anche quando il PC
sopra sbaglia.

Protocollo, una riga JSON per verso:

    stdin   {"id": 0, "gesto": "pronto", "pos": {"shoulder_pan": 0.0, ..., "gripper": 12.0},
             "velocita": 30}          (gradi/s, facoltativa; tetto VELOCITA_MAX)
    stdout  {"id": 0, "esito": "fatto" | "rifiutato" | "fermato", "motivo": "...",
             "errore": {"shoulder_pan": 0.4, ...}, "inizi": []}

Oppure una **coreografia**, pose di fila senza fermarsi (le battute del gioco):

    stdin   {"id": 1, "gesto": "mano", "tappe": [{...}, {...}], "velocita": 60}
    stdout  come sopra, con "inizi": l'ora (time.time()) d'inizio di ogni tappa

**stdout e' il tubo**: solo JSON. I messaggi per chi guarda vanno su stderr.

Cinque cancelli per ogni comando, nell'ordine:

1. bersaglio dentro il **range vero** (fine corsa della calibrazione meno un
   margine; il roll solo dove l'abbiamo visto arrivare) -> altrimenti
   **rifiutato**, e il braccio non si muove;
2. lettura **fresca**: il bus via usbipd a volte tace per oltre un secondo, e un
   comando deciso su una posa vecchia e' deciso su una posa che non c'e' piu';
3. **inseguimento**: il motore non resta indietro piu' di tanto (urto, peso,
   tensione che cala col 12 V da 2 A);
4. **passi piccoli**: `ensure_safe_goal_position`, la stessa di LeRobot in
   teleoperazione (D5);
5. **velocita' bassa**, con smoothstep: picco = 1,5 volte la media, quindi la
   durata si **calcola** dal giunto che si muove di piu' (come
   `so101_rps.durata`).

All'accensione: **prima il bersaglio = posizione attuale, poi la coppia**.
`SOFollower.connect()` riaccende la coppia senza farlo, e un servo Feetech
appena alimentato va verso l'ultimo bersaglio che ha in memoria.

All'uscita: torna lentamente alla posa di partenza, poi spegne la coppia. Se non
ci riesce, **tiene** il braccio e chiede di sostenerlo prima di lasciarlo.
"""

import argparse
import json
import math
import sys
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

# --- il braccio: come SOFollower (so_follower.py), stessa lingua del driver ---

MOTORI = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}
CORPO = [m for m in MOTORI if m != "gripper"]
CALIBRAZIONE = HF_LEROBOT_CALIBRATION / "robots" / "so_follower" / "follower.json"

#: Passi del motore per giro, meno uno: lo stesso `max_res` di `_normalize`.
PASSI_GIRO = 4095

# --- i limiti, ognuno col suo perche' ------------------------------------------

#: Gradi tenuti lontani dai fine corsa: un fine corsa si tocca solo a mano.
MARGINE = 5.0

#: Dove il polso roll e' **stato visto** arrivare (gemello, 2026-10-10). La
#: calibrazione dice giro intero perche' in calibrazione il roll non si muove:
#: e' falso. Il roll non ha un fine corsa netto, lo ferma il **cavo**: -72,6
#: sembrava la fine ed era durezza, poi si e' arrivati a -98,2. L'alto:
#: +169,4 senza che il cavo tirasse (fermati li' apposta, non dal cavo).
#: Mai vicino a +-180: li' il numero salta, e un bersaglio a cavallo farebbe
#: girare il polso dalla parte lunga. Col margine: [-93,2, +164,4].
ROLL_VISTO = (-98.2, 169.4)

#: Velocita' di picco dei giunti, gradi/s. Il servo ne fa ~250 a vuoto: 20 e'
#: per i primi movimenti, lenti e prevedibili. Pinza: percento al secondo.
VELOCITA = 20.0
VELOCITA_PINZA = 40.0

#: Il comando puo' chiedere di andare piu' svelto (il gioco: la carta a 20
#: gradi/s ci mette 7 s), ma **il tetto e' qui**, non nel progetto. La pinza
#: scala nella stessa proporzione.
VELOCITA_MAX = 60.0

#: Una lettura piu' lenta di cosi' e' vecchia: si ferma tutto.
ETA_MAX = 0.3

#: Ritardo massimo di un giunto dietro al comando. Il follower ha P=16
#: (morbido) e la gravita' lo tira: qualche grado e' normale, 20 no.
INSEGUIMENTO_MAX = 20.0

#: Scarto massimo fra posizione letta e bersaglio scritto, per ogni comando.
PASSO_MAX = 10.0

#: Comandi al secondo durante un movimento.
HZ = 20

#: Un movimento e' arrivato quando ogni giunto e' entro questi gradi.
ARRIVATO = 3.0
ATTESA_ARRIVO = 2.0

#: --prova: il braccio finto parte dalla posa di riposo letta il 2026-10-10.
RIPOSO_LETTO = {
    "shoulder_pan": 1.6, "shoulder_lift": -102.9, "elbow_flex": 81.7,
    "wrist_flex": 28.6, "wrist_roll": -10.5, "gripper": 1.3,
}


class Fermo(Exception):
    """Un cancello ha detto no a meta' movimento. Il braccio e' gia' stato tenuto."""


def log(msg: str) -> None:
    print(f"[esecutore] {msg}", file=sys.stderr, flush=True)


def rispondi(**campi) -> None:
    sys.stdout.write(json.dumps(campi) + "\n")
    sys.stdout.flush()


def carica_calibrazione() -> dict[str, MotorCalibration]:
    return {k: MotorCalibration(**v) for k, v in json.loads(CALIBRAZIONE.read_text()).items()}


def range_vero(calib: dict[str, MotorCalibration]) -> dict[str, tuple[float, float]]:
    """Il range di ogni giunto **sul ferro**, in gradi LeRobot, meno il margine.

    Lo zero LeRobot e' il centro del range calibrato, quindi i fine corsa
    stanno a +-meta' ampiezza. Si leggono dalla calibrazione, non si copiano.
    """
    rng = {}
    for nome in CORPO:
        c = calib[nome]
        meta = (c.range_max - c.range_min) / 2 * 360 / PASSI_GIRO
        rng[nome] = (-meta + MARGINE, meta - MARGINE)
    rng["wrist_roll"] = (ROLL_VISTO[0] + MARGINE, ROLL_VISTO[1] - MARGINE)
    rng["gripper"] = (0.0, 100.0)
    return rng


#: Tolleranza del cancello 0 sui fine corsa calibrati: appoggiato, un giunto
#: sta proprio li' e puo' leggere un grado oltre.
TOLLERANZA_PARTENZA = 2.0


def range_visto(calib: dict[str, MotorCalibration]) -> dict[str, tuple[float, float]]:
    """Dove il braccio **puo' trovarsi**: i fine corsa calibrati e il roll visto,
    senza margine. E' il cancello 0: la partenza, non il bersaglio."""
    rng = {}
    for nome in CORPO:
        c = calib[nome]
        meta = (c.range_max - c.range_min) / 2 * 360 / PASSI_GIRO + TOLLERANZA_PARTENZA
        rng[nome] = (-meta, meta)
    rng["wrist_roll"] = ROLL_VISTO
    rng["gripper"] = (-5.0, 105.0)
    return rng


def fuori_range(
    pos: dict[str, float], rng: dict[str, tuple[float, float]], che_cosa: str = "range vero"
) -> list[str]:
    """Cancelli 0 e 1. Una riga per ogni giunto che non va bene; vuota se va tutto bene."""
    problemi = [f"manca {m}" for m in MOTORI if m not in pos]
    for m, v in pos.items():
        if m not in rng:
            problemi.append(f"giunto sconosciuto {m}")
            continue
        lo, hi = rng[m]
        if not lo <= v <= hi:
            problemi.append(f"{m}={v:.1f} fuori dal {che_cosa} [{lo:.1f}, {hi:.1f}]")
    return problemi


def partenza_sbagliata(qui: dict[str, float], visto: dict[str, tuple[float, float]]) -> str:
    """Cancello 0. Vuoto se si puo' partire; altrimenti cosa fare, a mano.

    **Perche' esiste (2026-10-10):** appoggiato dopo il gemello, il polso e'
    scivolato oltre +180; il motore conta un giro solo e ha letto -160,9. Il
    bersaglio "pronto" era nel range, la **partenza** no, e per arrivarci il
    motore e' andato dalla parte lunga: +262 gradi, avvolgendo il cavo.
    L'inseguimento l'ha fermato. Un motore in posizione non puo' attraversare
    il punto in cui 4095 diventa 0: da li' si torna solo a mano.
    """
    problemi = fuori_range(qui, visto, "range in cui il braccio puo' stare")
    if not problemi:
        return ""
    return ("; ".join(problemi) + ". Riportalo a mano col gemello (il polso roll a circa +100, "
            "la ganascia con le viti in alto) e riprova.")


def dentro(pos: dict[str, float], rng: dict[str, tuple[float, float]]) -> dict[str, float]:
    """La posa piu' vicina dentro il range: per il rientro, che parte da un fine corsa."""
    return {m: min(max(v, rng[m][0]), rng[m][1]) for m, v in pos.items()}


# --- il bus: vero o finto -------------------------------------------------------


class BusFinto:
    """--prova: un braccio ideale, sempre esattamente dove gli si dice. Nessuna porta."""

    def __init__(self, partenza: dict[str, float]) -> None:
        self.pos = dict(partenza)

    def sync_read(self, nome: str, num_retry: int = 0) -> dict[str, float]:
        return dict(self.pos)

    def sync_write(self, nome: str, valori: dict[str, float]) -> None:
        self.pos.update(valori)

    def disable_torque(self, num_retry: int = 0) -> None:
        pass

    def disconnect(self, disable_torque: bool = True) -> None:
        pass


def apri(porta: str) -> FeetechMotorsBus:
    """Connette e configura come `SOFollower.configure()`, ma accende la coppia **dopo**
    aver messo il bersaglio dove il braccio gia' e'."""
    bus = FeetechMotorsBus(port=porta, motors=MOTORI, calibration=carica_calibrazione())
    bus.connect()
    bus.disable_torque(num_retry=10)
    bus.configure_motors()
    for m in bus.motors:
        bus.write("Operating_Mode", m, OperatingMode.POSITION.value)
        bus.write("P_Coefficient", m, 16)
        bus.write("I_Coefficient", m, 0)
        bus.write("D_Coefficient", m, 32)
    # Pinza: coppia e corrente a meta', come in SOFollower, per non bruciarla stringendo.
    bus.write("Max_Torque_Limit", "gripper", 500)
    bus.write("Protection_Current", "gripper", 250)
    bus.write("Overload_Torque", "gripper", 25)

    qui = bus.sync_read("Present_Position", num_retry=10)
    motivo = partenza_sbagliata(qui, range_visto(bus.calibration))
    if motivo:
        bus.disconnect(disable_torque=True)
        log(f"NON accendo la coppia: {motivo}")
        sys.exit(3)
    bus.sync_write("Goal_Position", qui)
    bus.enable_torque(num_retry=10)
    return bus


# --- il movimento ---------------------------------------------------------------


def leggi(bus) -> dict[str, float]:
    """Cancello 2: la posizione, solo se arriva in tempo. Altrimenti si tiene e si ferma."""
    t0 = time.perf_counter()
    try:
        pos = bus.sync_read("Present_Position", num_retry=3)
    except ConnectionError as e:
        raise Fermo(f"il bus non risponde: {e}") from e
    eta = time.perf_counter() - t0
    if eta > ETA_MAX:
        tieni(bus, pos)
        raise Fermo(f"lettura vecchia: {eta * 1000:.0f} ms (massimo {ETA_MAX * 1000:.0f})")
    return pos


def tieni(bus, pos: dict[str, float]) -> None:
    """Bersaglio = dove si trova: il braccio si ferma li' e resta in coppia."""
    try:
        bus.sync_write("Goal_Position", pos)
    except ConnectionError:
        log("non riesco nemmeno a tenerlo: bus muto")


def durata(da: dict[str, float], a: dict[str, float], velocita: float = VELOCITA) -> float:
    """Cancello 5. Con lo smoothstep il picco e' 1,5 volte la media: decide il giunto
    che si muove di piu' rispetto alla sua velocita'. Oltre ``VELOCITA_MAX`` no."""
    velocita = min(velocita, VELOCITA_MAX)
    corpo = max(abs(a[m] - da[m]) for m in CORPO) / velocita
    pinza = abs(a["gripper"] - da["gripper"]) / (VELOCITA_PINZA * velocita / VELOCITA)
    return max(1.5 * max(corpo, pinza), 0.2)


def muovi(bus, verso: dict[str, float], velocita: float = VELOCITA) -> dict[str, float]:
    """Porta il braccio a ``verso``. Restituisce l'errore finale per giunto; ``Fermo``
    se un cancello dice no (il braccio e' gia' stato tenuto dov'era)."""
    partenza = partenza_controllata(bus)
    tratto(bus, partenza, verso, durata(partenza, verso, velocita), partenza)
    return assestati(bus, verso)


def coreografia(
    bus, tappe: list[dict[str, float]], velocita: float = VELOCITA
) -> tuple[dict[str, float], list[float]]:
    """Le tappe **di fila**, senza assestamento in mezzo: il ritmo lo tiene questo
    ciclo, non chi manda i comandi (sei comandi separati aspetterebbero ognuno
    fino a ``ATTESA_ARRIVO``: battute irregolari).

    Ogni tratto parte dal bersaglio precedente, non dalla posizione letta: un
    servo morbido e' sempre un po' indietro, e ripartire da dov'e' allungherebbe
    ogni battuta. Il cancello 4 tiene comunque ogni comando entro ``PASSO_MAX``
    dalla posizione letta. La durata di ogni tratto e' il cancello 5, come per un
    gesto singolo. Restituisce l'errore finale e l'ora (``time.time()``) d'inizio
    di ogni tappa: chi legge la camera sa quando e' partito il segno.
    """
    partenza = partenza_controllata(bus)
    da, comandato, inizi = partenza, partenza, []
    for verso in tappe:
        inizi.append(time.time())
        comandato = tratto(bus, da, verso, durata(da, verso, velocita), comandato)
        da = verso
    return assestati(bus, tappe[-1]), inizi


def partenza_controllata(bus) -> dict[str, float]:
    """Cancello 0 su cio' che il braccio legge adesso."""
    partenza = leggi(bus)
    motivo = partenza_sbagliata(partenza, range_visto(carica_calibrazione()))
    if motivo:
        tieni(bus, partenza)
        raise Fermo(f"partenza fuori range: {motivo}")
    return partenza


def tratto(
    bus, da: dict[str, float], verso: dict[str, float], t_tot: float, comandato: dict[str, float]
) -> dict[str, float]:
    """Da ``da`` a ``verso`` in ``t_tot`` secondi, smoothstep, cancelli 2-4 a ogni passo.
    Restituisce l'ultimo bersaglio scritto (serve al cancello 3 del tratto dopo)."""
    passi = max(1, math.ceil(t_tot * HZ))
    for k in range(1, passi + 1):
        t0 = time.perf_counter()
        s = k / passi
        s = s * s * (3 - 2 * s)
        obiettivo = {m: da[m] + s * (verso[m] - da[m]) for m in MOTORI}

        ora = leggi(bus)
        for m in CORPO:  # cancello 3; la pinza no: stringendo un oggetto resta indietro apposta
            ritardo = abs(ora[m] - comandato[m])
            if ritardo > INSEGUIMENTO_MAX:
                tieni(bus, ora)
                raise Fermo(f"{m} non segue: {ritardo:.1f} gradi dietro al comando")

        # cancello 4: mai piu' di PASSO_MAX dalla posizione letta
        sicuro = ensure_safe_goal_position({m: (obiettivo[m], ora[m]) for m in MOTORI}, PASSO_MAX)
        bus.sync_write("Goal_Position", sicuro)
        comandato = sicuro
        time.sleep(max(0.0, 1 / HZ - (time.perf_counter() - t0)))
    return comandato


def assestati(bus, verso: dict[str, float]) -> dict[str, float]:
    """I servi morbidi arrivano un po' dopo l'ultimo comando: si aspetta, al massimo
    ``ATTESA_ARRIVO``. Restituisce l'errore finale per giunto."""
    t_fine = time.perf_counter() + ATTESA_ARRIVO
    while True:
        ora = leggi(bus)
        errore = {m: round(ora[m] - verso[m], 1) for m in MOTORI}
        if max(abs(errore[m]) for m in CORPO) < ARRIVATO or time.perf_counter() > t_fine:
            return errore
        time.sleep(1 / HZ)


def rientra(bus, partenza: dict[str, float], rng, prova: bool) -> None:
    """Torna da dove si e' partiti, poi lascia la coppia. Se non ci riesce, tiene."""
    log("rientro lento alla posa di partenza")
    try:
        muovi(bus, dentro(partenza, rng))
    except Fermo as e:
        log(f"rientro fermato: {e}")
        if not prova:
            log("il braccio e' TENUTO in coppia. Sostienilo con la mano, poi premi Invio")
            with open("/dev/tty") as tty:  # stdin e' il tubo: la tastiera si legge qui
                tty.readline()
    bus.disable_torque(num_retry=10)
    log("coppia spenta")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    gruppo = ap.add_mutually_exclusive_group(required=True)
    gruppo.add_argument("--porta", help="porta del follower (da porte.sh)")
    gruppo.add_argument("--prova", action="store_true", help="braccio finto, nessuna porta")
    args = ap.parse_args()

    rng = range_vero(carica_calibrazione())
    for m, (lo, hi) in rng.items():
        log(f"range vero {m:<14} [{lo:7.1f}, {hi:6.1f}]")

    bus = BusFinto(RIPOSO_LETTO) if args.prova else apri(args.porta)
    partenza = bus.sync_read("Present_Position", num_retry=10)
    log("PROVA: braccio finto" if args.prova else f"coppia accesa su {args.porta}, fermo dov'era")

    try:
        for riga in sys.stdin:
            cmd = json.loads(riga)
            nome = cmd.get("gesto", "?")
            tappe = cmd["tappe"] if "tappe" in cmd else [cmd["pos"]]
            problemi = [p for pos in tappe for p in fuori_range(pos, rng)]
            if problemi:
                log(f"{nome}: RIFIUTATO, il braccio non si muove")
                rispondi(id=cmd["id"], esito="rifiutato", motivo="; ".join(problemi), errore={})
                break
            velocita = float(cmd.get("velocita", VELOCITA))
            log(f"{nome}: in movimento a {min(velocita, VELOCITA_MAX):.0f} gradi/s")
            try:
                if "tappe" in cmd:
                    errore, inizi = coreografia(bus, tappe, velocita)
                else:
                    errore, inizi = muovi(bus, cmd["pos"], velocita), []
            except Fermo as e:
                log(f"{nome}: FERMATO: {e}")
                rispondi(id=cmd["id"], esito="fermato", motivo=str(e), errore={})
                break
            rispondi(id=cmd["id"], esito="fatto", motivo="", errore=errore, inizi=inizi)
    except KeyboardInterrupt:
        log("interrotto")
    finally:
        rientra(bus, partenza, rng, args.prova)
        bus.disconnect(disable_torque=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
