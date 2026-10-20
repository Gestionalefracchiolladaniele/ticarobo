"""Giochi: i compiti in cui il robot interagisce con una persona, non con oggetti.

La differenza col resto del progetto e' l'avversario. In ``planning/`` il mondo
e' passivo: un cubo sta dove lo lasci. Qui il mondo **reagisce**, cambia
strategia quando perde, e sa che lo stai guardando.

Un livello per gioco, perche' il robot che lo esegue puo' cambiare: lo stesso
sasso-carta-forbici gira sul SO-101 in simulazione e sul SO-101 vero, e il
pezzo che decide non se ne accorge.
"""
