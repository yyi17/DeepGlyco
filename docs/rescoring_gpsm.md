# DeepGlyco Tutorial: Rescoring Glycopeptide Structures
Rescoring glycopeptide spectral matches using predicted spectra.

## Prerequisites
### System Requirements
This tutorial has been tested on a workstation with Intel Core i9-12900K CPU, 64 GB RAM, and Microsoft Windows 10 Version 22H2 (OS Build 19045.2604) operating system. A NVIDIA GeForce RTX 3090 GPU is needed, with CUDA Version 11.6.

### Software Dependency
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
- matplotlib (version 3.6.2)
- tensorboard (version 2.10.0)

Later versions may be compatible, but have not been tested.

## Tutorial Data
### Starting Materials
Starting materials of this tutorial are available at ProteomeXchange and iProX with identifier [`PXD045248`](http://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD045248) or [`IPX0007075002`](https://www.iprox.cn/page/project.html?id=IPX0007075002).

MS/MS data in mzML format (6 files):
- Fut8_*_max_IGP_mousebrain_*.mzML

StrucGP search result files (6 files):
- Fut8_*_max_IGP_mousebrain_*_result.xlsx

A pretrained model for glycopeptide MS/MS prediction:
- PXD025859_MouseFiveTissues_gpms2b_model.pt

A glycan structure space file:
- MouseBrainFut.gdb

## Rescoring
Create a directory for the project, and place all the files in the folder.

Open Anaconda PowerShell Prompt, set the project folder as working directory, and set path to the scripts as global parameter.
``` powershell
cd "Path_to_project_data"
$script_path = "Path_to_DeepGlyco\src"
```

Start rescoring.
``` powershell
python "$script_path\score_gpsmb.py" `
--config = "$script_path\deepglyco\gpscore\gpscore_strucgp_branch.yaml" `
--gdb "MouseBrainFut.gdb" `
--model "PXD025859_MouseFiveTissues_gpms2b_model.pt" `
--gpsm (ls "Fut*_result.xlsx" | select -ExpandProperty FullName) `
--spec (ls "Fut*.mzML" | select -ExpandProperty FullName) `
--pred "PXD035158_MouseBrain_StrucGP_result_branch_prediction.speclib.h5" `
--out "PXD035158_MouseBrain_StrucGP_DeepGlyco_branch_prediction_result.csv"
```
The results are saved in a CSV file (`*_result.csv`).

The predicted spectral library (`*_prediction.speclib.h5`) and temp files of extracted spectra (`*.spec.h5`) from mzML files are cached. When reanalyzing, delete them if you do not want to use cached data.

