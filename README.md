# SubMAPP

SubMAPP is a python package specialized in the application of machine learning to biogeochemical oceanography. More specifically, it trains and uses **Self-Organizing Maps (SOM)** and **Hidden Markov Models (HMM)** in order to infer subsurface variables from surface data.

The original motivation of this SubMAPP package comes from Charantonis et al. (2015). [^Char]. We then used SubMAPP to carry out experiments on the impact of sparsity in the surface data on the reconstruction of the subsurface variables, this work was published at the 9th International Workshop on Climate Informatics, (Galmiche et al. (2019) [^Gal]).

In addition, 2 tutorials are available, to give an overview of how to use SubMAPP:

- [`tuto_som.ipynb`](./tuto_som.ipynb): Trains 2 SOMs, one for the surface data and one for the subsurface data.
- [`tuto_hmm.ipynb`](./tuto_hmm.ipynb): Trains a HMM model in order to infer subsurface data from the surface data, using the 2 trained SOMs.

[^Gal]: N. Galmiche, J. Brajard, A. Charantonis, and T. Wakamatsu, “Impact of sparse profile sampling on the reconstruction of subsurface ocean temperature from surface information,” in Proceedings of the 9th International Workshop on Climate Informatics: CI 2019, NCAR, 2019, pp. 87–91. doi: 10.5065/y82j-f154.
[^Char]: A. A. Charantonis, F. Badran, and S. Thiria, “Retrieving the evolution of vertical proles of chlorophyll-a from satellite observations using hidden markov models and self-organizing topological maps,” Remote Sensing of Environment, vol. 163, pp. 229–239, 2015.