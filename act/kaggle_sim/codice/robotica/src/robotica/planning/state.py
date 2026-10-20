"""Lo stato del mondo come fatti, non come coordinate.

Il pianificatore non sa cosa sia un metro: ragiona su **relazioni**. Questo
modulo traduce la scena fisica in fatti logici, e quella traduzione e' il ponte
fra i due mondi.

I predicati sono quelli standard del *blocks world*, il dominio con cui si
insegna il planning da trent'anni:

    clear(X)      X non ha niente sopra
    on_table(X)   X e' appoggiato al tavolo
    on(X, Y)      X e' appoggiato sopra Y
    holding(X)    la pinza tiene X
    arm_empty     la pinza non tiene niente
    in_zone(X, Z) X e' dentro la zona Z

Sono standard di proposito, non per pigrizia: un motore di planning vero si
aspetta questi nomi, e un LLM li ha visti nei libri di testo. Inventarne di
nuovi renderebbe il modello intraducibile e piu' difficile da generare.

**I fatti si rileggono dalla scena a ogni pianificazione.** Tenere uno stato a
parte e aggiornarlo sembra piu' efficiente ed e' il modo piu' rapido per
pianificare su una finzione: basta un cubo che rotola, o una presa che cede, e
il modello interno non corrisponde piu' al mondo. Qui la scena e' l'unica fonte
di verita', e costa una lettura.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Un cubo e' 40 mm. Due cubi impilati hanno i centri a 40 mm l'uno dall'altro:
# meta' del cubo sotto piu' meta' del cubo sopra.
CUBE_SIZE = 0.04

# Tolleranze per decidere se un cubo e' sopra un altro.
#
# In verticale si e' larghi (60% dell'altezza attesa): il cubo sopra si assesta,
# rimbalza, resta di qualche millimetro piu' alto o piu' basso del nominale.
# In orizzontale si e' stretti (60% del lato): due cubi affiancati sul tavolo
# non devono mai risultare "uno sopra l'altro".
STACK_Z_TOL = CUBE_SIZE * 0.6
STACK_XY_TOL = CUBE_SIZE * 0.6


@dataclass(frozen=True)
class Fact:
    """Un fatto: un predicato applicato a degli oggetti.

    Immutabile e hashabile, cosi' che uno stato sia un insieme di fatti e due
    stati si confrontino con ``==``. Al pianificatore serve: visita molti stati
    e deve riconoscere quelli gia' visti, altrimenti gira in tondo.
    """

    predicate: str
    args: tuple[str, ...] = ()

    def __str__(self) -> str:
        if not self.args:
            return self.predicate
        return f"{self.predicate}({', '.join(self.args)})"


# Costruttori: piu' leggibili di Fact("on", ("a", "b")) sparso nel codice.


def clear(x: str) -> Fact:
    """``x`` non ha niente sopra: si puo' afferrare, o ci si puo' posare sopra."""
    return Fact("clear", (x,))


def on_table(x: str) -> Fact:
    """``x`` e' appoggiato al tavolo, non su un altro oggetto."""
    return Fact("on_table", (x,))


def on(x: str, y: str) -> Fact:
    """``x`` e' appoggiato sopra ``y``."""
    return Fact("on", (x, y))


def holding(x: str) -> Fact:
    """La pinza tiene ``x``."""
    return Fact("holding", (x,))


def arm_empty() -> Fact:
    """La pinza non tiene niente."""
    return Fact("arm_empty")


def in_zone(x: str, zone: str) -> Fact:
    """``x`` e' dentro la zona ``zone``."""
    return Fact("in_zone", (x, zone))


def color(x: str, c: str) -> Fact:
    """``x`` e' di colore ``c``.

    E' un fatto di tipo diverso dagli altri: **non cambia mai**. Nessuna azione
    lo aggiunge o lo toglie — un cubo rosso resta rosso anche dopo essere stato
    spostato. In letteratura si chiama *fatto statico*, e serve a esprimere
    obiettivi su una proprieta' invece che su un nome: "i rossi di qua" invece
    di "cube_a di qua, cube_e di qua, cube_h di qua".

    Sta qui, e non nel nome del cubo, perche' in Fase 3 sara' una camera a
    produrlo. Un nome non si puo' percepire; una proprieta' si'.
    """
    return Fact("color", (x, c))


State = frozenset[Fact]
"""Uno stato e' l'insieme dei fatti veri in un dato momento.

``frozenset`` e non ``set``: il pianificatore usa gli stati come chiavi per
ricordare dove e' gia' passato, e una chiave deve essere immutabile.

Vale la **ipotesi del mondo chiuso**: cio' che non e' scritto e' falso. Se
``clear(cube_red)`` non c'e', il cubo ha qualcosa sopra. Serve a non dover
elencare anche tutto cio' che non e' vero, che sarebbe infinito.
"""


@dataclass
class WorldModel:
    """Come si legge lo stato dalla scena, e dove sono le cose.

    Tiene insieme due cose che il pianificatore non deve conoscere: la geometria
    (dove sta un oggetto, in metri) e le zone di posa. Il pianificatore ragiona
    sui fatti; chi esegue ha bisogno delle coordinate.
    """

    objects: list[str] = field(default_factory=list)
    zones: dict[str, np.ndarray] = field(default_factory=dict)
    zone_half: float = 0.06
    colors: dict[str, str] = field(default_factory=dict)
    """Colore di ogni oggetto, se la scena ne ha uno. Vuoto se il colore non
    conta — la scena a due cubi non ne ha bisogno, e aggiungere fatti inutili
    allarga lo spazio di ricerca senza guadagno."""

    def read(
        self,
        positions: dict[str, np.ndarray],
        held: str | None = None,
    ) -> State:
        """Traduce le posizioni in fatti.

        ``positions`` sono le coordinate lette dalla scena; ``held`` e' cio' che
        la pinza tiene, che dalle sole coordinate non si dedurrebbe in modo
        affidabile — un cubo in mano e un cubo sollevato si somigliano troppo.
        """
        facts: set[Fact] = set()

        # Chi sta sopra chi. Si calcola prima, perche' serve per sapere chi e'
        # libero e chi e' sul tavolo.
        supports: dict[str, str] = {}  # sopra -> sotto
        for upper in positions:
            if upper == held:
                continue
            lower = self._supporting_object(upper, positions, held)
            if lower is not None:
                supports[upper] = lower

        occupied = set(supports.values())

        for name, pos in positions.items():
            if name == held:
                facts.add(holding(name))
                continue

            if name in supports:
                facts.add(on(name, supports[name]))
            else:
                facts.add(on_table(name))

            # Libero se nessuno gli sta sopra. Un oggetto in mano non e' libero:
            # non ci si puo' posare niente sopra.
            if name not in occupied:
                facts.add(clear(name))

            for zone_name in self.zones:
                if self.in_zone_xy(pos, zone_name):
                    facts.add(in_zone(name, zone_name))

        # I colori sono fatti statici: veri sempre, anche per il cubo in mano.
        for name, c in self.colors.items():
            if name in positions:
                facts.add(color(name, c))

        facts.add(holding(held) if held else arm_empty())
        return frozenset(facts)

    def _supporting_object(
        self,
        upper: str,
        positions: dict[str, np.ndarray],
        held: str | None,
    ) -> str | None:
        """L'oggetto su cui ``upper`` e' appoggiato, se c'e'.

        Due condizioni: allineato in orizzontale, e piu' alto di circa un cubo.
        Le tolleranze sono asimmetriche di proposito — vedi ``STACK_Z_TOL``.
        """
        up = positions[upper]
        for lower, low in positions.items():
            if lower == upper or lower == held:
                continue
            if abs(up[2] - low[2] - CUBE_SIZE) > STACK_Z_TOL:
                continue
            if float(np.linalg.norm(up[:2] - low[:2])) > STACK_XY_TOL:
                continue
            return lower
        return None

    def in_zone_xy(self, pos: np.ndarray, zone: str) -> bool:
        """Se una posizione cade dentro una zona. Quadrata, come in ``sim/env``."""
        center = self.zones[zone]
        return bool(np.all(np.abs(pos[:2] - center[:2]) <= self.zone_half))


def describe(state: State) -> str:
    """Lo stato in forma leggibile, ordinato.

    Ordinato perche' due stati uguali devono *apparire* uguali: un insieme non
    ha ordine, e senza questo confrontare due stampe sarebbe inutile.
    """
    return "\n".join(f"  {f}" for f in sorted(state, key=str))
