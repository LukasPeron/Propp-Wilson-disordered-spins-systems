# Propp-Wilson Disordered Spins Systems

This repository contains the work I have done during my Master 2 Internship at INRIA and DI-ENS under the supervision of Dr. Louise Budzynski on an implementation of the Coupling From The Past Propp-Wilson algorithm[^1] for disordered spins systems with bounding chains[^2][^3]. The following models are studied:

- Curie-Weiss model
- Ferromagnetic Ising model in spatial dimension 2 and 3
- Sherrington-Kirkpatrick model
- Edwards-Anderson model in spatial dimension 2 and 3 with $\pm 1$ couplings
- Viana-Bray model with $\pm 1$ couplings
- Spin glass model on random regular graphs with $\pm 1$ couplings

All the results can be found in the "figures" folder in which one can find the coupling time analysis in "algo" and the observables predictions in "physics".
See also [This repository](https://github.com/LouiseBudzynski/Bounding_Chain_Disordered_Ising_Models) and the paper we wrote (to be added).

For any further information, please contact [lukas.peron@ens.psl.eu](mailto:lukas.peron@ens.psl.eu)

[^1]: James Gary Propp and David Bruce Wilson. Exact sampling with coupled markov chains and applications to statistical mechanics. Random Structures & Algorithms, 9(1-2):223–252, 1996.
[^2]: Mark Huber. Exact sampling and approximate counting techniques. In Proceedings of the Thirtieth Annual ACM Symposium on Theory of Computing, STOC ’98, page 31–40, New York, NY, USA, 1998. Association for Computing Machinery.
[^3]: Olle Häggström and Karin Nelander. On exact simulation of markov random fields using coupling from the past. Scandinavian Journal of Statistics, 26(3):395–411, 1999.
