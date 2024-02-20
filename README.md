# DeepGlyco

DeepGlyco is a framework for predicting fragment mass spectra of intact glycopeptides.

## Dependency
For model training, NVIDIA graphics cards with CUDA are recommended.

The following software and packages are required:
- Python (version 3.9.16, [Anaconda](https://www.anaconda.com/) distribution is recommended)
- [PyTorch](https://pytorch.org/) (version 1.12.1)
- [DGL](https://www.dgl.ai/) (version 1.0.1)
- numpy (version 1.23.5)
- pandas (version 1.5.2)
- scipy (version 1.10.1)
- scikit-learn (version 1.2.1)
- statsmodels (version 0.13.2)
- h5py (version 3.7.0)
- pymzml (version 2.5.2)

The following packages are optional for visualization:
- matplotlib (version 3.6.2)
- networkx (version 3.0)
- tensorboard (version 2.10.0)

Later versions may be compatible, but have not been tested.


## Tutorial
Tutorials are avaliable in the [`docs`](docs) folder.

### Model Training for MS/MS Prediction
[**DeepGlyco Tutorial: Training New Models for MS/MS**](docs/train_ms2.md) describes the analysis workflow for training MS/MS models.

### Model Finetuning with New Data
[**DeepGlyco Tutorial: Model Finetuning with New Data**](docs/finetune_ms2.md) describes the analysis workflow for finetuning MS/MS models using users' data.

### Model Training for iRT Prediction
[**DeepGlyco Tutorial: Training New Models for iRT**](docs/train_rt.md) describes the analysis workflow for training iRT models.

### DIA Analysis Using Predicted Libraries
[**DeepGlyco Tutorial: DIA Analysis Using Predicted Libraries**](docs/DIA.md) describes the DIA analysis workflow using GproDIA and spectral libraries predicted by DeepGlyco.

### Rescoring Glycopeptide Spectral Matches
[**DeepGlyco Tutorial: Rescoring Glycopeptide Structures**](docs/rescoring_gpsm.md) describes the analysis workflow for rescoring glycopeptide spectral matches reported by other tools using spectral libraries predicted by DeepGlyco.

## License
DeepGlyco is distributed under a BSD license. See the LICENSE file for details.
