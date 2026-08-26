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

---

* I dati sono divisi in Slices, `slice_size = threads_per_block * items_per_thread_per_slice`
* Vengono avviati al più N blocchi per SM, non uno solo per massimizzare l'occupacy
  * Mentre un blocco è in attesa dell'a memoria, l'SM può eseguirne un altro
* Vengono avviati più thread per blocco del numero di cores per SM
  * Stessa ragione di sopra, ma a livello di warp
* Ogni thread processa più items in una slice
  * Riduce l'overhead di avviare più thread/blocchi per accomodare slice più piccole
* I blocchi iterano sulle slice in stride di `gridDim`
* I thread iterano sugli item in stride di `blockDim`

## Upfront global exclusive prefix sum

* Dobbiamo calcolare la prefix sum degli istogrammi globali, per digit place
  * Siccome la radix è "piccola", un singolo blocco può occuparsi di un intero place
* I due algoritmi di prefix sum parallela sono
  * Hillis-Steele: latency efficient, buono quando hai tanti processori quanti dati
  * Blelloch: work efficient, buono quando i dati sono molto più dei processori
* GPU Gems usa Blelloch perché assume di voler trattare array di dimensioni arbitrarie
* Noi possiamo permetterci di usare Hillis-Steele per sfruttare gli intrinsics hardware
  * Usando $\log_2(32)$ operazioni `shfl_up_sync` eseguiamo una inclusive prefix sum di 32 dati a livello di warp
  * Usando $\log_2(\frac{\mathrm{RADIX}}{32})$ operazioni `shfl_up_sync` eseguiamo una inclusive prefix sum sui totali per warp
  * Con un ultimo `shfl_up_sync(1, ...)` e sommando il prefix del totale precedente otteniamo la exclusive prefix sum

## Onesweep

* Un'iterazione per digit place
* Avviamo tanti blocchi quante slice, per sfruttare lo scheduling della gpu
  * Ogni blocco ottiene un ID runtime (indicativo dell'ordine in cui sono avviati) e si occupa solo di quella slice
* Ogni warp esegue un binning warp-privatized di una subslice della slice
  * Contemporaneamente, ogni thread calcola una thread-privatized warp-local prefix sum per digit
* Viene eseguita una prefix sum per radix digit attraverso i bin per-warp
  * A seconda del numero di warp-privatized bins, un solo warp può eseguire più prefix sum contemporaneamente
  * Potremmo farlo più efficiente nel caso di alcuni warp counts non potenze di due, ma non vale la pena
    * Check più complessi ad ogni iterazione potrebbero essere peggio della ridotta efficienza
    * Si applica solo alcuni warp count appena superiori ad una potenza di due (e.g. 10 ma non 11)
* I bin privatizzati sono memorizzati in modo da minimizzare i bank conflict
  * L'ordine degli indici massimizza "stride" tra bin in conflitto, riduce la possibilità di conflitti popolando i bin
  * Extra celle in ogni array di bin garantisce 0 conflitti in fase di prefix sum
