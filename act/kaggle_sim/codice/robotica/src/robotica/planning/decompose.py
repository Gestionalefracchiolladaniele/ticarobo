"""Spezzare un obiettivo grande in obiettivi piccoli, quando conviene.

## Il problema, in concreto

``planner.py`` cerca in **ampiezza**: prova tutte le sequenze di un passo, poi
tutte quelle di due, e cosi' via, finche' una raggiunge l'obiettivo. E' il modo
piu' semplice e da' sempre il piano piu' corto. Ha un costo, pero', e il costo
cresce male.

Con due cubi e una zona le azioni applicabili in uno stato sono una manciata, e
un piano da quattro passi si trova visitando poche centinaia di stati. Con
quattro cubi e due zone le azioni diventano una cinquantina, e un piano da otto
passi richiederebbe di visitare **milioni** di stati: la ricerca sbatte contro
``MAX_VISITED`` e risponde "probabilmente irraggiungibile" su un problema che a
un umano sembra ovvio. La formula e' spietata — se in media ci sono `b` azioni
applicabili e il piano e' lungo `d`, gli stati sono nell'ordine di `b**d`. Non
e' un difetto dell'implementazione: e' come si comporta la ricerca esaustiva.

## La risposta

Smistare quattro cubi **non e' un problema da otto passi**. Sono quattro
problemi da due passi, indipendenti l'uno dall'altro: dove finisce il cubo rosso
non cambia cosa fare col verde. Risolverli uno alla volta costa `4 * b**2`
invece di `b**8`, che con `b=50` significa diecimila stati invece di
seicentomila miliardi.

Questa e' la differenza fra *cercare* e *sapere*. La ricerca in ampiezza non sa
che i quattro sotto-problemi sono indipendenti: lo scopre riprovando ogni
combinazione. Qui glielo diciamo.

## Quando si puo' fare, e quando no

Decomporre e' corretto solo se i sotto-obiettivi **non interferiscono**: portare
a termine il secondo non deve disfare il primo. Vale per lo smistamento (ogni
cubo va in una zona sua) e **non vale** per le pile: ``on(a, b)`` e ``on(b, c)``
vanno risolti insieme, perche' impilare `a` su `b` prima di aver messo `b` su
`c` costringe a smontare tutto.

Per questo la decomposizione non e' automatica su qualsiasi obiettivo: si
riconosce il caso sicuro e negli altri si lascia lavorare il pianificatore
intero. **Un'euristica che sbaglia in silenzio sarebbe peggio del problema che
risolve** — produrrebbe piani che si disfano da soli, e la causa sarebbe
invisibile guardando il piano.

## L'ordine conta

I sotto-obiettivi vengono ordinati mettendo per primi i cubi **liberi**. Un cubo
sepolto sotto un altro richiede di sgombrarlo, e se si sgombera dopo aver
spostato quello sopra, il lavoro e' gia' fatto. E' un ordinamento e basta: se e'
sbagliato il piano resta corretto, solo un po' piu' lungo.
"""

from __future__ import annotations

from dataclasses import dataclass

from robotica.planning.state import Fact, State, clear


@dataclass
class Decomposition:
    """Un obiettivo spezzato, o la ragione per cui non lo si e' spezzato.

    Si porta dietro il motivo anche quando la risposta e' "non si puo'": chi
    guarda un'esecuzione lenta deve poter capire *perche'* ha pianificato tutto
    insieme, senza leggere questo file.
    """

    parts: list[frozenset[Fact]]
    reason: str = ""

    def __len__(self) -> int:
        return len(self.parts)

    def __bool__(self) -> bool:
        """Vero se l'obiettivo e' stato davvero spezzato in piu' pezzi."""
        return len(self.parts) > 1

    def explain(self) -> str:
        if not self:
            return f"Obiettivo non decomposto: {self.reason}"
        return (
            f"Obiettivo decomposto in {len(self.parts)} sotto-obiettivi indipendenti "
            f"({self.reason})."
        )


def decompose(goal: frozenset[Fact], state: State) -> Decomposition:
    """Spezza l'obiettivo in parti indipendenti, se e' sicuro farlo.

    Restituisce sempre qualcosa di utilizzabile: nel caso peggiore una sola
    parte, cioe' l'obiettivo intero. Chi chiama non deve gestire un caso
    speciale, e questo e' voluto — un'ottimizzazione che cambia la forma del
    risultato si dimentica di gestire.
    """
    if len(goal) <= 1:
        return Decomposition(parts=[goal], reason="e' gia' un obiettivo singolo")

    # Le relazioni fra oggetti legano i sotto-obiettivi fra loro: ``on(a, b)``
    # parla di due cubi insieme, e spostarne uno cambia l'altro. Non si tocca.
    if any(f.predicate == "on" for f in goal):
        return Decomposition(
            parts=[goal],
            reason="contiene relazioni 'on' fra oggetti, che vanno risolte insieme",
        )

    per_oggetto: dict[str, set[Fact]] = {}
    for f in goal:
        if f.predicate not in ("in_zone", "on_table"):
            return Decomposition(
                parts=[goal],
                reason=f"contiene '{f.predicate}', di cui non so se sia indipendente",
            )
        per_oggetto.setdefault(f.args[0], set()).add(f)

    if len(per_oggetto) <= 1:
        return Decomposition(parts=[goal], reason="riguarda un solo oggetto")

    # Prima i cubi gia' liberi: quelli sepolti spesso si liberano da soli
    # mentre si spostano gli altri, e cosi' il loro piano si accorcia.
    ordinati = sorted(
        per_oggetto,
        key=lambda nome: (clear(nome) not in state, nome),
    )

    return Decomposition(
        parts=[frozenset(per_oggetto[nome]) for nome in ordinati],
        reason="un oggetto alla volta, nessuno interferisce con gli altri",
    )
