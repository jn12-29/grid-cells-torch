# Banino et al. 2018 Supervised-Learning Hyperparameters

This note records the gradient-clipping behavior and supervised-learning
hyperparameters checked from the local paper files:

- Main paper PDF: `Banino et al. - 2018 - Vector-based navigation using grid-like representations in artificial agents.pdf`
- Supplementary PDF: `41586_2018_102_MOESM1_ESM.pdf`

## Key conclusions

- The paper describes gradient clipping as elementwise value clipping, not norm
  clipping: each gradient-vector element is clipped to the interval `[-gc, gc]`.
- In the supervised path-integration experiments, the clipping threshold is
  `gc = 1e-5`.
- The clipping scope described in the main paper is the parameters projecting
  from the dropout linear layer `g_t` to the place-cell and head-direction-cell
  predictions `y_t` and `z_t`.

## Where this appears in the PDFs

| Finding | Location |
| --- | --- |
| Training uses RMSProp, with weight decay for weights from the dropout linear layer to the place-cell and head-direction-cell predictions; hyperparameter values are in Supplementary Methods Table 1. | Main paper PDF, PDF page 6, Methods, "Path integration: supervised learning experiments". |
| Gradient clipping is used for parameters projecting from dropout linear layer `g_t` to predictions `y_t` and `z_t`; clipping is defined as clipping each gradient-vector element into `[-gc, gc]`. | Main paper PDF, PDF page 6, Methods, "Gradient clipping". |
| Dropout probability is `0.5` on each `g_t` unit. | Main paper PDF, PDF page 6, Methods, "Grid cell network architecture". |
| Backpropagation through time unrolls the network into blocks of `100` time steps. | Main paper PDF, PDF page 6, Methods, "Supervised learning loss". |
| The numeric supervised-learning hyperparameters, including `gc`, minibatch size, trajectory length, learning rate, momentum, L2 regularisation, and parameter updates, are listed in Supplementary Table 1. | Supplementary PDF, PDF page 24, "Table 1: Supervised learning hyperparameters." The main paper PDF points to this table but does not include the numeric table itself. |

## Supplementary Table 1 values

| Parameter | Value | Description |
| --- | ---: | --- |
| `T` | `15` | Duration of simulated trajectories in seconds. |
| `L` | `2.2` | Width and height of the environment, or diameter for the circular environment, in meters. |
| `d` | `0.03` | Perimeter-region distance to walls in meters. |
| `sigma(v)` | `0.13` | Forward-velocity Rayleigh distribution scale in meters per second. |
| `mu(phi)` | `0` | Rotation-velocity Gaussian distribution mean in degrees per second. |
| `sigma(phi)` | `330` | Rotation-velocity Gaussian distribution standard deviation in degrees per second. |
| `rho_RH` | `0.25` | Velocity reduction factor when located in the perimeter. |
| `Delta_RH` | `90` | Change in angle when located in the perimeter, in degrees. |
| `Delta t` | `0.02` | Simulation-step time increment in seconds. |
| `N` | `256` | Number of place cells. |
| `sigma(c)` | `0.01` | Place-cell standard deviation parameter in meters. |
| `M` | `12` | Number of target head-direction cells. |
| `kappa(h)` | `20` | Head-direction concentration parameter. |
| `gc` | `1e-5` | Gradient clipping threshold. |
| minibatch size | `10` | Number of trajectories used to calculate a stochastic gradient. |
| trajectory length | `100` | Number of time steps in the trajectories used for supervised learning. |
| learning rate | `1e-5` | Step-size multiplier in RMSProp. |
| momentum | `0.9` | RMSProp momentum parameter. |
| L2 regularisation | `1e-5` | Regularisation parameter for the linear layer. |
| parameter updates | `300000` | Total number of gradient-descent steps. |

## Notes

- `pdftotext` extracts the scientific-notation entries in Supplementary Table 1
  as `10 5`, but visual inspection of Supplementary PDF page 24 shows `10^-5`.
  The `1e-5` values above follow the visible PDF table.
- The original DeepMind reference implementation is consistent with these
  values: `training_clipping = 1e-5`, `learning_rate = 1e-5`, and
  `model_weight_decay = 1e-5`; its clipping helper uses TensorFlow
  `clip_by_value`, matching elementwise value clipping.
