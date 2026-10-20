"""I nomi dal tempo, non dalle posizioni note.

E' il limite che D18 aveva dichiarato e lasciato aperto: ``observer.perceive``
associa ogni detection al nome noto piu' vicino, e le posizioni note vengono dal
simulatore. La percezione non barava — ``detect.py`` non chiama mai MuJoCo — ma
**l'ultimo passo si', e proprio quello che produce i nomi**.

Qui l'informazione privilegiata sparisce. Il tracker riceve soltanto le
detection di un fotogramma e cio' che ha visto nei fotogrammi precedenti.

**L'identita' e' inventata, non scoperta.** Non esiste nulla in un'immagine che
dica "questo e' ``cube_a``": il nome e' una convenzione del resto del sistema.
Il tracker non prova a indovinarlo. Al primo avvistamento assegna ``obj_0``,
arbitrariamente, e da quel momento il suo unico compito e' **non cambiarlo**.

    il nome non significa niente, significa qualcosa che **non cambi**

E' cio' che rende la metrica misurabile sul robot vero: che ``obj_3`` sia il
cubo rosso non lo si puo' verificare senza un oracolo, ma che ``obj_3`` smetta
all'improvviso di esistere e compaia ``obj_7`` al suo posto **si vede**, e
significa che il tracker ha perso.

## L'associazione: assegnamento ottimo, non avidita'

Con sei tracce e sei detection ci sono 720 abbinamenti. Prendere per ogni
traccia la detection piu' vicina — il modo avido — sembra ragionevole e si
incastra: se la traccia A prende la detection che serviva a B, B resta con una
lontana, e lo scambio delle due avrebbe dato un totale migliore. L'esito dipende
**dall'ordine in cui si servono le tracce**, che e' una proprieta' del codice,
non della scena.

L'assegnamento ottimo costruisce la matrice dei costi (traccia x detection, il
costo e' la distanza) e cerca l'abbinamento che minimizza la **somma**. Lo
risolve l'algoritmo ungherese, in ``scipy.optimize.linear_sum_assignment`` —
gia' installato, BSD. E' esattamente cio' che fa SORT, lo standard del settore,
la cui implementazione di riferimento e' **GPL**: la stessa trappola di licenza
che in D18 ha escluso YOLO-World, evitata scrivendo le ottanta righe che stanno
sopra una funzione di libreria.

## Il gating: l'assegnamento ottimo assegna comunque

Una guardia che non e' un dettaglio. ``linear_sum_assignment`` abbina **tutto
cio' che puo'**: data una traccia e una detection a tre metri le accoppia, non
perche' siano la stessa cosa ma perche' e' l'unica coppia disponibile. Senza un
limite, un oggetto che scompare **cede la propria identita'** al primo oggetto
libero che passa — il fallimento peggiore di un tracker, perche' produce un
sistema che non dice mai di aver perso.

Quindi dopo l'assegnamento gli abbinamenti oltre ``max_distance`` si scartano, e
le due parti tornano separate: detection senza traccia (nasce un'identita'),
traccia senza detection (comincia a sparire).

## Perche' morire richiede pazienza e nascere no

Asimmetria voluta. Una detection nuova diventa subito una traccia: costa poco,
e se e' rumore muore da sola. Una traccia che non si vede **non muore subito**:
resta viva per ``max_missed`` fotogrammi. Un'occlusione e' l'evento piu' comune
in una scena con un braccio che ci passa davanti, e un cubo coperto per un
istante non e' un cubo diverso quando riappare.

E' la forma che il progetto conosce: ``execution.py`` ritenta prima di
ripianificare, per la stessa ragione. Un fallimento singolo non e' uno stato.

## Cosa questo modulo non sa

Non sa cos'e' un cubo, un braccio, MuJoCo o un AMR. Prende punti 3D e restituisce
identita'. **Su un robot mobile vale identico** — anzi: e' li' che serve
davvero, perche' su una base che si muove nulla sta dove l'hai lasciato e le
posizioni note non esistono proprio.

*Limite dichiarato:* l'associazione e' per **sola vicinanza**. Due oggetti che
si incrociano a meno della loro distanza tipica possono scambiarsi le identita',
e il tracker non se ne accorge. Rimedi noti: predire il moto (filtro di Kalman a
velocita' costante, che sul nastro servira' davvero) e confrontare l'aspetto,
non solo la posizione. Nessuno dei due e' qui, perche' nessun esperimento
attuale li mette sotto sforzo — un codice che nessun test puo' far fallire non
dimostra niente (D5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from robotica.perception.detect import Detection

__all__ = ["Track", "Tracker", "TrackedScene"]


# Oltre questa distanza, fra un fotogramma e il successivo, non e' lo stesso
# oggetto. 8 cm: piu' di quanto un cubo si sposta fra due fotogrammi anche sul
# nastro, e meno della distanza fra due cubi vicini (16 cm), quindi non puo'
# scambiarli. **E' un limite di plausibilita', non di qualita'.**
MAX_MATCH_DISTANCE = 0.08

# Per quanti fotogrammi una traccia sopravvive senza essere vista. 5: piu' di
# un'occlusione del braccio che transita, meno di una sparizione vera.
MAX_MISSED = 5

# Quante volte una traccia dev'essere vista prima di essere considerata reale.
# 1 significa subito: il rumore della depth e' gia' filtrato da ``detect.py``
# (area minima, coerenza in quota), e filtrare due volte nasconderebbe quale
# dei due filtri sta lavorando.
MIN_HITS = 1


@dataclass
class Track:
    """Un oggetto seguito nel tempo.

    Attributes:
        name: l'identita' assegnata alla nascita, ``obj_<n>``. **Non descrive
            l'oggetto**: e' un'etichetta che vale finche' il tracker la
            mantiene.
        detection: l'ultima osservazione associata. Porta con se' colore,
            incertezza e dimensione: il tracker non li ricalcola, li trasporta.
        position: dove sta, secondo l'ultima osservazione.
        hits: quante volte e' stata vista in tutto. Una traccia con molti hit ha
            una storia, e una storia e' una prova che non e' rumore.
        misses: da quanti fotogrammi consecutivi non si vede. Torna a zero a
            ogni associazione.
        first_frame: il fotogramma in cui e' nata. Con ``hits`` dice se e' stata
            seguita **con continuita'** o a singhiozzo, che sono due esiti
            diversi e un conteggio solo li confonderebbe (D17).
        last_frame: l'ultimo fotogramma in cui e' stata vista.
    """

    name: str
    detection: Detection
    position: np.ndarray
    hits: int
    misses: int
    first_frame: int
    last_frame: int

    @property
    def color(self) -> str | None:
        """Il colore dell'ultima osservazione, o ``None`` se non riconoscibile."""
        return self.detection.color

    @property
    def uncertainty(self) -> float:
        """Quanto dista il colore misurato dal riferimento scelto."""
        return self.detection.color_distance

    @property
    def visible(self) -> bool:
        """Se e' stata vista nell'ultimo fotogramma processato."""
        return self.misses == 0


@dataclass
class TrackedScene:
    """La scena vista nel tempo, esposta come la vede il planning.

    Ha la stessa firma di ``PerceivedScene`` — ``object_names``, ``color_of``,
    ``position_of`` — cosi' ``planning/bridge.py`` puo' riceverla al posto
    dell'una o dell'altra senza modifiche. **Non eredita da nulla**: la
    compatibilita' sta nei metodi, ed e' il modo di non legare il planning a
    quale percezione ha prodotto i fatti.

    La differenza con ``PerceivedScene`` non e' nell'interfaccia ma nella
    provenienza dei nomi: li' venivano dalle posizioni vere, qui dal tempo.
    """

    tracks: dict[str, Track] = field(default_factory=dict)
    frame: int = 0

    def object_names(self) -> list[str]:
        """Le identita' visibili adesso.

        **Solo quelle visibili**: una traccia in attesa di riapparire esiste
        ancora — il tracker non l'ha dimenticata — ma non viene dichiarata al
        planning, perche' pianificare su un oggetto che nessuno sta vedendo e'
        pianificare su una finzione, ed e' precisamente cio' che il progetto
        evita rileggendo lo stato dalla scena.
        """
        return sorted(n for n, t in self.tracks.items() if t.visible)

    def color_of(self, name: str) -> str:
        """Il colore percepito dell'oggetto.

        Raises:
            KeyError: se l'identita' non esiste, o se esiste ma non e' visibile
                adesso, o se e' visibile ma il colore non e' riconoscibile. Sono
                **tre** casi e il messaggio li distingue: "non c'e'", "c'era e
                ora non si vede" e "non so dirlo" portano a decisioni diverse.
        """
        if name not in self.tracks:
            raise KeyError(
                f"'{name}' non e' un'identita' nota: {sorted(self.tracks)}"
            )
        track = self.tracks[name]
        if not track.visible:
            raise KeyError(
                f"'{name}' esiste ma non e' visibile da {track.misses} "
                f"fotogrammi: il colore sarebbe quello dell'ultima volta, "
                "non di adesso"
            )
        if track.color is None:
            raise KeyError(
                f"'{name}' e' visibile ma il colore non e' riconoscibile "
                f"(distanza {track.uncertainty:.2f})"
            )
        return track.color

    def position_of(self, name: str) -> np.ndarray:
        """Dove sta l'oggetto, secondo l'ultima osservazione.

        Raises:
            KeyError: se l'identita' non esiste o non e' visibile adesso. Una
                posizione vecchia e' **peggio di nessuna posizione**: sembra un
                dato e non lo e', e il braccio ci andrebbe. E' il difetto che
                sul nastro fa puntare ``pick`` dove il cubo *era*.
        """
        if name not in self.tracks:
            raise KeyError(f"'{name}' non e' un'identita' nota")
        track = self.tracks[name]
        if not track.visible:
            raise KeyError(
                f"'{name}' non e' visibile da {track.misses} fotogrammi: "
                "l'ultima posizione nota e' di allora, non di adesso"
            )
        return track.position


class Tracker:
    """Segue gli oggetti fra un fotogramma e l'altro e da' loro un'identita'.

    Si costruisce una volta e si chiama ``update`` a ogni fotogramma. Lo stato
    — quali tracce esistono, da quanto — vive qui dentro: e' l'unico punto di
    ``perception/`` che **ricorda**, e non poteva essere altrimenti, perche'
    l'identita' nel tempo e' per definizione memoria.

    Args:
        max_distance: oltre questa distanza un abbinamento si scarta, anche se
            era il migliore disponibile. Vedi il gating in cima al modulo.
        max_missed: per quanti fotogrammi una traccia sopravvive senza essere
            vista, prima di essere dimenticata.
        min_hits: quante osservazioni servono prima che una traccia conti.

    Example:
        >>> tracker = Tracker()
        >>> scena = tracker.update(detect_objects(...))
        >>> scena.object_names()
        ['obj_0', 'obj_1']
    """

    def __init__(
        self,
        max_distance: float = MAX_MATCH_DISTANCE,
        max_missed: int = MAX_MISSED,
        min_hits: int = MIN_HITS,
    ) -> None:
        if max_distance <= 0:
            raise ValueError(f"max_distance = {max_distance}: dev'essere positiva")
        if max_missed < 0:
            raise ValueError(f"max_missed = {max_missed}: non puo' essere negativo")

        self.max_distance = max_distance
        self.max_missed = max_missed
        self.min_hits = min_hits

        self._tracks: dict[str, Track] = {}
        self._next_id = 0
        self._frame = -1

        # Quante volte un'identita' e' stata dimenticata e quante ne sono nate
        # dopo il primo fotogramma. Sono **la diagnostica del tracker**: un
        # sistema che perde e ricrea identita' ha lo stesso numero di oggetti
        # visibili di uno che funziona, e ``object_names()`` non lo rivela.
        self.deaths = 0
        self.late_births = 0

    @property
    def frame(self) -> int:
        """Quanti fotogrammi sono stati processati, meno uno. -1 prima del primo."""
        return self._frame

    def update(self, detections: list[Detection]) -> TrackedScene:
        """Un fotogramma: associa, aggiorna, fa nascere e morire.

        Args:
            detections: cio' che la camera vede adesso, da ``detect_objects``.
                **Nessun'altra informazione**: niente posizioni note, niente
                nomi. E' il punto di tutto il modulo.

        Returns:
            La scena con le identita' correnti.
        """
        self._frame += 1

        vive = list(self._tracks.values())
        i_tracce, i_detections = self._associate(vive, detections)

        associate: set[int] = set()
        for it, idd in zip(i_tracce, i_detections, strict=True):
            track = vive[it]
            det = detections[idd]
            track.detection = det
            track.position = det.center
            track.hits += 1
            track.misses = 0
            track.last_frame = self._frame
            associate.add(idd)

        # Le tracce non associate invecchiano; quelle troppo vecchie spariscono.
        for i, track in enumerate(vive):
            if i in set(i_tracce):
                continue
            track.misses += 1
            if track.misses > self.max_missed:
                del self._tracks[track.name]
                self.deaths += 1

        # Ogni detection rimasta e' un oggetto mai visto prima: nasce.
        for i, det in enumerate(detections):
            if i in associate:
                continue
            nome = f"obj_{self._next_id}"
            self._next_id += 1
            self._tracks[nome] = Track(
                name=nome,
                detection=det,
                position=det.center,
                hits=1,
                misses=0,
                first_frame=self._frame,
                last_frame=self._frame,
            )
            if self._frame > 0:
                self.late_births += 1

        return TrackedScene(
            tracks={
                n: t for n, t in self._tracks.items() if t.hits >= self.min_hits
            },
            frame=self._frame,
        )

    def _associate(
        self, tracce: list[Track], detections: list[Detection]
    ) -> tuple[list[int], list[int]]:
        """Chi va con chi: assegnamento ottimo, poi gating.

        Returns:
            Due liste parallele di indici, ``(indici_tracce, indici_detection)``,
            gia' ripulite dagli abbinamenti troppo lontani.

        L'ordine delle due operazioni conta e non e' intercambiabile. Si
        ottimizza **prima** e si scarta **dopo**: scartare prima le coppie
        lontane cambierebbe la matrice dei costi e quindi l'ottimo, facendo
        rientrare dalla finestra la dipendenza dall'ordine che l'algoritmo
        ungherese serve proprio a togliere.
        """
        if not tracce or not detections:
            return [], []

        # Matrice dei costi: riga = traccia, colonna = detection, valore =
        # distanza in metri. E' la distanza euclidea fra punti 3D nel mondo,
        # **non fra pixel**: significa che il tracker non cambia se la camera si
        # sposta, ed e' cio' che lo rende utile su un robot mobile.
        posizioni_t = np.array([t.position for t in tracce], dtype=float)
        posizioni_d = np.array([d.center for d in detections], dtype=float)
        costi = np.linalg.norm(
            posizioni_t[:, None, :] - posizioni_d[None, :, :], axis=2
        )

        righe, colonne = linear_sum_assignment(costi)

        # Il gating. Senza, una traccia orfana si prende la prima detection
        # libera per quanto lontana: vedi in cima al modulo.
        i_tracce, i_detections = [], []
        for r, c in zip(righe, colonne, strict=True):
            if costi[r, c] <= self.max_distance:
                i_tracce.append(int(r))
                i_detections.append(int(c))

        return i_tracce, i_detections

    def reset(self) -> None:
        """Dimentica tutto: nessuna traccia, identita' da capo da ``obj_0``.

        Serve fra un esperimento e l'altro. **Non** e' cio' che va chiamato
        quando la scena cambia: un tracker che si azzera a ogni cambiamento non
        sta tracciando niente.
        """
        self._tracks.clear()
        self._next_id = 0
        self._frame = -1
        self.deaths = 0
        self.late_births = 0
