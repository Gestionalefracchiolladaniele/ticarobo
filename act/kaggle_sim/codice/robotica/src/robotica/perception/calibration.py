"""Calibrazione mano-occhio: dove sta la camera rispetto al robot.

E' il pezzo che sul robot vero **non si puo' saltare**. In simulazione
``render.camera_pose`` da' la posa esatta perche' il simulatore l'ha costruita;
nella realta' nessuno la conosce — un metro a nastro da' centimetri, e un
centimetro fa mancare ogni presa.

La posa si ricava da **osservazioni**: si porta il braccio in pose note, si
guarda dove la camera vede un riferimento solidale alla pinza, e si risolve per
la trasformazione che mette d'accordo le due cose.

**Perche' questo modulo esiste ed e' corto.** Il calcolo lo fa
``cv2.calibrateHandEye``, che e' lo standard industriale: ``easy_handeye``, il
pacchetto ROS piu' diffuso, non fa altro che avvolgerlo. Cercare cosa esiste
prima di scrivere (D5) qui vale piu' che altrove — un solutore di calibrazione
scritto a mano e' molto codice e molti modi di sbagliare in silenzio.

**Il cambio di ruolo, che e' la trappola.** ``calibrateHandEye`` risolve il caso
**eye-in-hand**: camera sul polso, restituisce ``T_gripper_camera``. Il nostro
e' **eye-to-hand** (D18): camera ferma, serve ``T_base_camera``. Non serve un
altro solutore — si **invertono i ruoli**:

    eye-in-hand:   base -> gripper  e  camera -> target(fermo)
    eye-to-hand:   gripper -> base  e  camera -> target(sul polso)

Passando ``T_gripper_base`` (l'inversa di quella che si userebbe) al posto di
``T_base_gripper``, lo stesso solutore restituisce ``T_base_camera``. E' il
trucco standard, ed e' documentato qui perche' **dedurlo dai nomi delle
variabili e' il modo tipico di sbagliare di mezzo metro** — lo stesso genere
di errore di frame che in D16 e' costato l'offset di 96 mm.

**Un solutore che fallisce deve dire di aver fallito.** Con pose che traslano
soltanto, la rotazione e' matematicamente indeterminata. Misurato: il risultato
non e' un numero sbagliato ma un crash — ``numpy.linalg.LinAlgError: SVD did
not converge``, sollevato da scipy mentre prova a leggere come rotazione una
matrice degenere. **Il messaggio non nomina la calibrazione, ne' le pose, ne' il
fatto che sono parallele**, e chi lo incontra cerca un bug numerico.

Da qui ``_check_pose_variety``, che guarda le pose **prima** di passarle al
solutore e rifiuta nei termini del problema. E' la forma dei rifiuti di
``safety`` — regola, valore e limite — applicata qui: un sistema che si arrende
deve dire di essersi arreso (D14).

**La verifica che solo la simulazione permette.** Qui la risposta esatta e'
nota: si calibra col metodo reale e si confronta con ``camera_pose``. Sul robot
vero quel termine di paragone **non esiste** — si puo' solo sperare che la
calibrazione sia giusta. E' il posto dove un risultato dall'aria plausibile puo'
essere **smentito**, ed e' la ragione per cui vale la pena farlo in sim prima.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from robotica.transforms import Pose, inverse, position_error, rotation_error

__all__ = ["CalibrationResult", "calibrate_eye_to_hand", "METHODS"]

# I cinque solutori di OpenCV. Tsai e' il classico ed e' veloce; Daniilidis usa
# i quaternioni duali e risolve rotazione e traslazione **insieme** invece che
# in sequenza, il che lo rende in genere piu' stabile quando le pose sono poche.
METHODS: dict[str, int] = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


# Quanto devono differire gli orientamenti fra le pose perche' il problema sia
# determinato. 5 gradi e' poco piu' del rumore di un encoder e molto meno di
# cio' che serve a una calibrazione buona: e' una soglia di **assurdita'**, non
# di qualita'.
MIN_ROTATION_SPREAD_DEG = 5.0


@dataclass(frozen=True)
class CalibrationResult:
    """L'esito di una calibrazione.

    Attributes:
        T_base_camera: la posa cercata — dove sta la camera nel frame della
            base del robot.
        method: quale solutore l'ha prodotta.
        n_poses: quante osservazioni sono state usate. Sotto tre la
            calibrazione e' sottodeterminata e il risultato e' **plausibile ma
            privo di significato**, che e' il caso peggiore.
        position_error_mm: scarto dalla verita', se fornita. ``None`` sul robot
            reale, dove la verita' non esiste — ed e' esattamente il motivo per
            cui questo campo esiste separato invece di essere stampato: un
            numero assente e' un'informazione, un numero finto no.
        rotation_error_deg: idem per l'orientamento.
    """

    T_base_camera: Pose
    method: str
    n_poses: int
    position_error_mm: float | None = None
    rotation_error_deg: float | None = None


def calibrate_eye_to_hand(
    T_base_gripper: list[Pose],
    T_camera_target: list[Pose],
    method: str = "daniilidis",
    T_truth: Pose | None = None,
) -> CalibrationResult:
    """Ricava ``T_base_camera`` da osservazioni. Montaggio eye-to-hand.

    Args:
        T_base_gripper: per ogni scatto, la posa della pinza nel frame base.
            **La sa il robot**: e' cinematica diretta dagli angoli dei giunti,
            ed e' nota anche sul robot reale.
        T_camera_target: per ogni scatto, la posa del riferimento (solidale
            alla pinza) vista **dalla camera**. Sul robot reale la si ottiene
            da un marker ArUco o una scacchiera con ``cv2.solvePnP``.
        method: uno di ``METHODS``.
        T_truth: la posa vera, quando e' nota (simulazione). Se c'e', il
            risultato porta con se' lo scarto.

    Returns:
        Il risultato, con la posa e — se ``T_truth`` c'era — quanto sbaglia.

    Raises:
        ValueError: se le liste hanno lunghezze diverse o meno di tre pose.

    Le pose devono essere **varie**: tre scatti in cui il braccio trasla soltanto
    lasciano la rotazione indeterminata, e il solutore restituisce comunque un
    risultato. E' la forma che il progetto conosce bene — un valore che si
    presenta senza dire di essere privo di fondamento (D2).
    """
    if len(T_base_gripper) != len(T_camera_target):
        raise ValueError(
            f"{len(T_base_gripper)} pose del braccio e {len(T_camera_target)} "
            "osservazioni: devono coincidere"
        )
    if len(T_base_gripper) < 3:
        raise ValueError(
            f"{len(T_base_gripper)} pose: ne servono almeno 3, e la rotazione "
            "deve variare fra l'una e l'altra"
        )
    if method not in METHODS:
        raise ValueError(f"metodo '{method}' sconosciuto; disponibili: {sorted(METHODS)}")

    spread = _rotation_spread_deg(T_base_gripper)
    if spread < MIN_ROTATION_SPREAD_DEG:
        raise ValueError(
            f"le pose ruotano solo di {spread:.2f}°, sotto il minimo di "
            f"{MIN_ROTATION_SPREAD_DEG}°: con pose quasi parallele la rotazione "
            "e' indeterminata e la calibrazione non ha soluzione unica. "
            "Servono pose che cambiano **orientamento**, non solo posizione."
        )

    # Il cambio di ruolo descritto in cima al modulo: si passa gripper->base
    # dove la firma dice base->gripper, e il solutore eye-in-hand risolve il
    # caso eye-to-hand.
    R_g2b, t_g2b = [], []
    for T in T_base_gripper:
        T_inv = inverse(np.asarray(T, dtype=float))
        R_g2b.append(T_inv[:3, :3])
        t_g2b.append(T_inv[:3, 3].reshape(3, 1))

    R_t2c, t_t2c = [], []
    for T in T_camera_target:
        T = np.asarray(T, dtype=float)
        R_t2c.append(T[:3, :3])
        t_t2c.append(T[:3, 3].reshape(3, 1))

    R, t = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=METHODS[method]
    )

    T_base_camera = np.eye(4)
    T_base_camera[:3, :3] = R
    T_base_camera[:3, 3] = np.asarray(t).ravel()

    err_pos = err_rot = None
    if T_truth is not None:
        truth = np.asarray(T_truth, dtype=float)
        err_pos = position_error(T_base_camera, truth) * 1000.0
        err_rot = float(np.degrees(rotation_error(T_base_camera, truth)))

    return CalibrationResult(
        T_base_camera=T_base_camera,
        method=method,
        n_poses=len(T_base_gripper),
        position_error_mm=err_pos,
        rotation_error_deg=err_rot,
    )


def _rotation_spread_deg(poses: list[Pose]) -> float:
    """Di quanto differiscono, al massimo, gli orientamenti di un gruppo di pose.

    Serve a distinguere un insieme di osservazioni utile da uno in cui il
    braccio ha solo traslato. Si confronta ogni posa con la prima: basta che
    **una** sia ruotata perche' il problema smetta di essere degenere.
    """
    if len(poses) < 2:
        return 0.0
    prima = np.asarray(poses[0], dtype=float)
    return max(
        float(np.degrees(rotation_error(prima, np.asarray(T, dtype=float))))
        for T in poses[1:]
    )
