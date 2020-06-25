# Planning and Execution using Inaccurate Models with Provable Guarantees
## BibTeX Citation
```
@INPROCEEDINGS{Vemula-RSS-20, 
    AUTHOR    = {Anirudh Vemula AND Yash Oza AND J. Bagnell AND Maxim Likhachev}, 
    TITLE     = {{Planning and Execution using Inaccurate Models with Provable Guarantees}}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2020}, 
    ADDRESS   = {Corvalis, Oregon, USA}, 
    MONTH     = {July}, 
    DOI       = {10.15607/RSS.2020.XVI.001} 
} 
```
## Dependencies
Most of the dependencies are listed in the `requirements.txt` file, and can be installed using the command

``` shell
pip install -r requirements.txt
```

Make sure to install the `gym` package that is provided locally instead of the latest version to ensure compatibility

``` shell
pip install -e gym
```

To run the 7D PR2 experiments, we need additional dependencies given in the `external/` folder that can be installed as follows

``` shell
pip install -e external/pyglet
pip install -e external/pyopengl
pip install -e external/urdfpy
```

## Reproducing Experiments

### Simulated 4D Planar Pushing
To run the experiments corresponding to the fetchpush environment (Table I in paper and Fig 2 (right)) run the following command inside `src/` folder

``` shell
python -m odium.experiment.experiment_fetch --exp-agent <agent> --exp-model <model>
```

where `<agent>` should be one of `{'rts', 'dqn', 'mbpo', 'mbpo_knn', 'rts_correct'}` where 
- `rts` corresponds to our approach, 
- `dqn` corresponds to Q-learning
- `mbpo` corresponds to Model learning with neural network model (NN)
- `mbpo_knn` corresponds to Model learning with K-nearest neighbors model (KNN)
- `rts_correct` corresponds to always planning with accurate model,

and `<model>` should be one of `{'accurate', 'inaccurate'}`.


## Contributors

The repository is maintained and developed by [Anirudh Vemula](vvanirudh.github.io) from the Search based Planning
Lab (SBPL) at CMU.
