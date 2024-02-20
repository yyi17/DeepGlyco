# DeepGlyco Tutorial: Training New Models for MS/MS
Training new models for glycopeptide MS/MS prediction.

## Prerequisites
### System Requirements
In this tutorial, model training has been tested on a workstation with Intel Core i9-12900K CPU, 64 GB RAM, and Microsoft Windows 10 Version 22H2 (OS Build 19045.2604) operating system. A NVIDIA GeForce RTX 3090 GPU is needed, with CUDA Version 11.6.

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
Starting materials of this tutorial are available at ProteomeXchange and iProX with identifier [`PXD045248`](http://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD045248) or [`IPX0007075001`](https://www.iprox.cn/page/project.html?id=IPX0007075001).


MS/MS data in mzML format (25 files):
- MouseBrain-Z-T-*.mzML.gz
- MouseHeart-Z-T-*.mzML.gz
- MouseKidney-Z-T-*.mzML.gz
- MouseLiver-Z-T-*.mzML.gz
- MouseLung-Z-T-*.mzML.gz

A StrucGP search result file:
- MouseFiveTissues_result.xlsx

A pretrained model for peptide MS/MS prediction:
- PXD004452_pepms2_model.pt


## Train a MS/MS model
### Prepare Training Data
Create a directory for the project.

Create a subdirectory named `mzML` in the project directory, and place the mzML files and StrucGP report in the subfolder:
- mzML\*.mzML.gz
- mzML\MouseFiveTissues_result.xlsx

Open Anaconda PowerShell Prompt, set the project folder as working directory, and set path to the scripts as global parameter.
``` powershell
cd "Path_to_project_data"
$script_path = "Path_to_DeepGlyco\src"
```

#### Spectrum Annotation
Create a subdirectory named `gpms2` in the project directory. Extract and annotate identified MS/MS spectra from the StrucGP result.
``` powershell
mkdir "gpms2" -f | Out-Null

python "$script_path\build_gpspeclib.py" `
--in "mzML\MouseFiveTissues_result.xlsx" `
--spectra (ls "mzML" "Mouse*.mzML.gz" | select -ExpandProperty FullName) `
--out "gpms2\PXD025859_MouseFiveTissues_StrucGP.speclib.h5"
```
A subdirectory named `gpms2` is created in the project directory. The extracted and annotated spectra are imported into a HDF5 file (`*.speclib.h5`).

#### Building Consensus Spectra
Combine replicate spectra into consensus spectra (one spectrum per glycopeptide precursor).
``` powershell
python "$script_path\build_consensus_gpspeclib.py" `
--in "gpms2\PXD025859_MouseFiveTissues_StrucGP.speclib.h5" `
--out "gpms2\PXD025859_MouseFiveTissues_StrucGP_consensus.speclib.h5"
```
The consensus spectra are saved in a HDF5 file (`*_consensus.speclib.h5`), which will be used as training data.

### Model Training
Create a subdirectory named `training` in the project directory, and place the pretrained model in the project folder:
- training\PXD004452_pepms2_model.pt

Start model training.
``` powershell
python "$script_path\train_gpms2.py" `
--in "gpms2\PXD025859_MouseFiveTissues_StrucGP_consensus.speclib.h5" `
--wkdir "training\deepgpms2\PXD025859_MouseFiveTissues" `
--pretrained "training\PXD004452_pepms2_model.pt"
```
The trained model checkpoints are saved in Python binary files (`epoch_*.pt`) in the path `training\deepgpms2\PXD025859_MouseFiveTissues\checkpoints`.

Open another Anaconda PowerShell Prompt, start TensorBoard to monitor the training process.
``` powershell
tensorboard --logdir=training\deepgpms2\PXD025859_MouseFiveTissues
```
Select the best model based on the metrics on training and validation set.

### Model Testing
Test the trained model.
``` powershell
python "$script_path\validate_gpms2.py" `
--model "training\deepgpms2\PXD025859_MouseFiveTissues\checkpoints\epoch_465.pt" `
--in "gpms2\PXD025859_MouseFiveTissues_StrucGP_consensus.speclib.h5" `
--out "training\deepgpms2\PXD025859_MouseFiveTissues\PXD025859_MouseFiveTissues_StrucGP_consensus_prediction.ms2score.csv"
```
Note that `epoch_465.pt` should be replaced with the model selected for testing. The `--in` argument should be changed when testing on other data (`*.speclib.h5`).

The spectral similarites of predicted and target data are saved as a CSV file(`*_prediction.ms2score.csv`).

### Model finetuning
When finetuning a pretrained model with other data, change the config and epoch number.
``` powershell
python "$script_path\train_gpms2.py" `
--in "gpms2\FinetuningData_consensus.speclib.h5" `
--wkdir "training\deepgpms2\FinetuningData" `
--pretrained "training\deepgpms2\PXD025859_MouseFiveTissues\checkpoints\epoch_465.pt" `
--config "$script_path\deepglyco\deeplib\gpep\ms2\gpms2model_finetune.yaml" `
--epochs 50
```


## Train a MS/MS model with B ions
The process is similar to the model without B ions, but different configs are used.

Prepare training data.
``` powershell
mkdir "gpms2b" -f | Out-Null

python "$script_path\build_gpspeclib.py" `
--in "mzML\MouseFiveTissues_result.xlsx" `
--spectra (ls "mzML" "Mouse*.mzML.gz" | select -ExpandProperty FullName) `
--out "gpms2b\PXD025859_MouseFiveTissues_StrucGP.speclib.h5" `
--config "$script_path\deepglyco\speclib\gpep\parser\strucgp_branch.yaml"

python "$script_path\build_consensus_gpspeclib.py" `
--in "gpms2b\PXD025859_MouseFiveTissues_StrucGP.speclib.h5" `
--out "gpms2b\PXD025859_MouseFiveTissues_StrucGP_consensus.speclib.h5" `
--min_num_fragments "{'Y':5,'B':2}"
```
All the data will be saved in a subdirectory named `gpms2b` in the project directory.

Start model training using a pretrained model without B ions.
``` powershell
python "$script_path\train_gpms2b.py" `
--in "gpms2b\PXD025859_MouseFiveTissues_StrucGP_consensus.speclib.h5" `
--wkdir "training\deepgpms2b\PXD025859_MouseFiveTissues" `
--pretrained "training\deepgpms2\PXD025859_MouseFiveTissues\checkpoints\epoch_465.pt"
```

Test the trained model.
``` powershell
python "$script_path\validate_gpms2b.py" `
--pretrained "training\deepgpms2b\PXD025859_MouseFiveTissues\checkpoints\epoch_448.pt" `
--in "gpms2b\PXD025859_MouseFiveTissues_StrucGP_consensus.speclib.h5" `
--out "training\deepgpms2b\PXD025859_MouseFiveTissues\PXD025859_MouseFiveTissues_StrucGP_consensus_prediction.ms2score.csv"
```
Note that `epoch_448.pt` should be replaced with the model selected for testing. The `--in` argument should be changed when testing on other data (`*.speclib.h5`).

