# Notes

## Global histogram sum

* Ci sono due modi di ottenere gli istogrammi globali:
  1. * Il primo kernel calcola un istogramma per blocco e li memorizza in global memory
     * Un secondo kernel (e.g. lo stesso della prefix sum) legge li istogrammi e li somma
  1. Il primo kernel calcola un istogramma per blocco e li somma atomicamente in global memory
* Il primo approccio:
  * Costa più memoria, nel nostro caso `blocchi*posizioni*cifre*dim_contatore = 43*4*256*4 = 184 kB` contro `posizioni*cifre*dim_contatore = 4*256*4 = 4 kB`
  * È più veloce nella lettura degli istogrammi globali (da parte del secondo kernel), l'array da leggere è più piccolo
* Il secondo approccio:
  * È più costoso nel calcolo degli istogrammi globali
  * Somme atomiche costano più di scritture semplici
* Abbiamo optato per il secondo approccio perché quello scelto da Nvidia nel paper
