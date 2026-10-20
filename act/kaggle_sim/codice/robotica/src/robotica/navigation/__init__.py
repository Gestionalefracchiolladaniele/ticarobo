"""Navigazione: dalla mappa al comando di velocita'.

La scomposizione **imita quella di Nav2**, la navigation stack ufficiale di
ROS 2 e lo standard di fatto per gli AMR industriali:

===========================  ==============================  ==================
qui                          in Nav2                         cosa fa
===========================  ==============================  ==================
``costmap.Costmap``          ``nav2_costmap_2d``             la mappa dei costi
``planner.astar``            ``nav2_navfn`` / ``nav2_smac``   il percorso globale
``controller.PurePursuit``   ``nav2_dwb`` / ``nav2_mppi``     la velocita' adesso
``mission``                  ``nav2_bt_navigator``           l'orchestrazione
===========================  ==============================  ==================

**Perche' imitarla invece di installarla.** ROS 2 su 2 core e 14 GB sarebbe
proibitivo, e il progetto non ha bisogno del middleware: ha bisogno
dell'**architettura**. Imitarla costa zero -- sono nomi -- e il giorno che
servisse Nav2 davvero i pezzi corrispondono uno a uno, invece di dover
riconoscere le proprie invenzioni nei concetti di qualcun altro.

**Cosa NON c'e', ed e' dichiarato.** Manca la **localizzazione** (AMCL in
Nav2): qui la posa della base si legge dalla scena, cioe' e' esatta. Sul robot
vero non esiste nessun ``qpos`` da interrogare: la posa e' una *stima* che
deriva, e AMCL la ricorregge confrontando il LiDAR con la mappa. E' il punto
di crollo del transfer (40-60%, vedi ``docs/transfer-sim-real.md``): tutto cio'
che sta qui funziona **assumendo di sapere dove si e'**, e quella e'
l'assunzione che il reale rompe per prima.

Non e' una svista: e' un limite dichiarato, che e' l'unico tipo controllabile.
"""
