"""L'AMR: un robot che si muove nel piano, senza braccio e senza tavolo.

Perche' una cartella separata da ``sim/``: le scene esistenti stanno su
robosuite, che presuppone **un braccio montato davanti a un tavolo**. Un robot
mobile non ha ne' l'uno ne' l'altro, e forzarlo dentro ``ManipulationEnv``
significherebbe combattere con l'ambiente invece di descrivere la scena. Qui si
usa **MuJoCo diretto**: si costruisce l'XML, si carica, si comanda.

Cosa *non* cambia, ed e' il punto: ``transforms.py``, ``perception/`` e
``planning/`` restano quelli. Se servisse riscriverli, la tesi del progetto —
che il valore sta nei layer alti, non nel robot — sarebbe falsa.
"""
