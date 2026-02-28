# configs/training/

Hyperparamètres et configurations d'entraînement par module.

**Statut :** À créer lors du développement de chaque module ML.

## Fichiers prévus

| Fichier | Module | Statut |
|---------|--------|--------|
| `module1_temporal.yaml` | NeuralODE + GATv2 temporal | TODO |
| `module2b_surrogate.yaml` | GATv2 surrogate spatial | TODO |

## Convention

Chaque fichier YAML doit contenir :
- `model` : architecture (dimensions, nombre de couches, têtes d'attention)
- `training` : lr, batch_size, n_epochs, early_stopping
- `loss` : coefficients de pondération (λ_div, pondération spatiale)
- `data` : split train/val/test, random seed (obligatoire pour reproductibilité)
- `inference` : paramètres spécifiques à l'inférence (N MC Dropout passes)

## Exemple de structure

```yaml
model:
  hidden_dim: 256
  n_layers: 8          # couches GATv2
  n_heads: 4           # têtes d'attention multi-tête
  dropout: 0.1

training:
  lr: 1.0e-4
  batch_size: 32
  n_epochs: 200
  early_stopping_patience: 20
  random_seed: 42      # fixé pour reproductibilité

loss:
  lambda_divergence: 0.1
  spatial_weight_strategy: "inverse_density"

data:
  train_fraction: 0.7
  val_fraction:   0.15
  test_fraction:  0.15
  test_period: "2017-05-01/2017-06-15"  # IOP Perdigão

inference:
  mc_dropout_n_passes: 20
```
