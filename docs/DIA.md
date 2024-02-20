# DeepGlyco Tutorial: DIA Analysis Using Predicted Libraries
Generate in silico spectral libraries from glycopeptide lists for data-independent acquisition (DIA) analysis.

## Prerequisites
### System Requirements
This tutorial has been tested on a workstation with Intel Core i9-12900K CPU, 64 GB RAM, and Microsoft Windows 10 Version 22H2 (OS Build 19045.2604) operating system. A NVIDIA GeForce RTX 3090 GPU is needed, with CUDA Version 11.6.

### Software Dependency
The following software and packages are required for spectral library prediction:
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

[GproDIA](https://github.io/lmsac/GproDIA) is not needed in the process of spectral library generation, but is used in the complete DIA data analysis workflow. The following software and packages are required:
- Python (version 3.5.6, [Anaconda](https://www.anaconda.com/) distribution is recommended)
- [OpenSWATH](http://openswath.org/) (version 2.6.0)
- [PyProphet](https://github.com/PyProphet/pyprophet) (version 2.1.5)
- [msproteomicstools](https://github.com/msproteomicstools/msproteomicstools) (version 0.11.0)
- numpy (version 1.18.5)
- pandas (version 0.25.3)
- scipy (version 1.4.1)
- scikit-learn (version 0.22.2.post1)

**Note that GproDIA and DeepGlyco require different versions of Python.** You must install them in separate virtual environment. Conda is recommended to manage the environment.

## Tutorial Data
### Starting Materials
Starting materials of this tutorial are available at ProteomeXchange and iProX.

[`PXD023980`](http://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD023980) or [`IPX0002792000`](https://www.iprox.cn/page/project.html?id=IPX0002792000)

DIA data in mzML format (3 files):
- 20200615_serum_NGlycP_1hDIA_rep*.mzML

DDA database search results for retention time calibration:
- pGlycoDB-GP-FDR-Pro.txt (in serum_DDA1h_singlerun_pGlyco.zip)
- 20200615_serum_NglycP_1hDDA_HCDFT.mgf

A SWATH window file:
- swath_window.tsv

A background glycans file:
- background_glycan.txt

[`PXD045248`](http://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD045248) or [`IPX0007075003`](https://www.iprox.cn/page/project.html?id=IPX0007075003)

A starting glycopeptide list:
- serum.glycopeptides.csv

Pretrained models for glycopeptide MS/MS and iRT prediction:
- PXD023980_HumanSerum_gpms2_model.pt
- PXD023980_HumanSerum_gprt_model.pt


## Spectral Library Prediction
Create a directory for the project.
Create a subdirectory named `library` in the project directory, and place the starting glycopeptide list and models in the subfolder:
- library\serum.glycopeptides.csv
- library\PXD023980_HumanSerum_gpms2_model.pt
- library\PXD023980_HumanSerum_gprt_model.pt

Open Anaconda PowerShell Prompt (**Python 3.9 for DeepGlyco**), set the project folder as working directory, and set path to the scripts as global parameter.
``` powershell
cd "Path_to_project_data"
$script_path = "Path_to_DeepGlyco\src"
```

### MS/MS Prediction
Predict MS/MS spectra for the glycopeptides in the starting glycopeptide list.
``` powershell
python "$script_path\predict_gpms2.py" `
--model "library\PXD023980_HumanSerum_gpms2_model.pt" `
--in "library\serum.glycopeptides.csv" `
--out "library\serum_prediction_finetune.speclib.h5"
```
The predicted spectra are saved in a HDF5 file (`*.speclib.h5`).

### iRT Prediction
Predict iRT for the glycopeptides in the starting glycopeptide list.
``` powershell
python "$script_path\predict_gprt.py" `
--no-validate `
--model "library\PXD023980_HumanSerum_gprt_model.pt" `
--in "library\serum.glycopeptides.csv" `
--out "library\serum_prediction_finetune.speclib.h5"
```
The predicted iRT are added to the HDF5 file (`*.speclib.h5`).

### Conversion to GproDIA Library
Convert the spectral library from HDF5 to GproDIA binary format.
``` powershell
python "$script_path\convert_speclib_to_GproDIA_assays.py" `
--in "library\serum_prediction_finetune.speclib.h5" `
--glycopeptide "library\serum.glycopeptides.csv" `
--out "library\serum_prediction.assay.pickle"
```

## DIA Analysis
Create a subdirectory named `mzML` in the project directory, and place the mzML files in the subfolder:
- mzML\20200615_serum_NGlycP_1hDIA_rep*.mzML

Create a subdirectory named `DDA` in the project directory, and place the DDA files in the subfolder:
- DDA\pGlycoDB-GP-FDR-Pro.txt
- DDA\20200615_serum_NglycP_1hDDA_HCDFT.mgf

Place the SWATH window file and background glycan file in the project directory:
- swath_window.tsv
- background_glycan.txt

Open Anaconda PowerShell Prompt (**Python 3.5 for GproDIA**), set the project folder as working directory, and set path to the scripts as global parameter.
``` powershell
cd "Path_to_project_data"
$script_path = "Path_to_GproDIA\src"
```

### Spectral Library Processing
Extract retention time anchors from the DDA results and combine replicate spectra into consensus spectra within each run..
``` powershell
python "$script_path\build_assays_from_pGlyco_mgf.py" `
--clean_glycan_struct `
--clean_glycan_site `
--psm "DDA\pGlycoDB-GP-FDR-Pro.txt" `
--mgf "DDA\20200615_serum_NglycP_1hDDA_HCDFT.mgf" `
--out "library\serum_1h.assay.pickle"

python "$script_path\remove_redundant_assays.py" `
--within_run `
--action consensus `
--in "library\serum\serum_1h.assay.pickle" `
--out "library\serum\serum_1h_consensus.assay.pickle"
```
The consensus spectra are saved in Python binary files (`*_consensus.assay.pickle`).

Transform retention times in iRT space to experimental gradient.
``` powershell
python "$script_path\calibrate_assays_rt.py" `
--multiple_runs `
--smooth lowess --lowess_frac 0.25 --lowess_it 3 `
--in "library\serum_prediction.assay.pickle" `
--reference "library\serum_1h_consensus.assay.pickle" `
--out "library\serum_prediction_rtcalibrated.assay.pickle" `
--out_anchor "library\serum_1h_consensus_rtanchor.assay.pickle"
```

Filtering library entries using a set of rules.
``` powershell
python "$script_path\filter_assays.py" `
--swath_windows "swath_window.tsv" `
--min_fragment_mz 200 --max_fragment_mz 2000 `
--in "library\serum_prediction_rtcalibrated.assay.pickle"
```

Generate a spectral library containing identification transitions for glycoform inference.
``` powershell
python "$script_path\generate_glycoform_uis_assays.py" `
--swath_windows "swath_window.tsv" `
--min_fragment_mz 200 --max_fragment_mz 2000 `
--background_glycans "background_glycan.txt" `
--max_background_glycan_number 50 `
--in "library\serum_prediction_rtcalibrated_filtered.assay.pickle"
```

Generate decoy libraries.
``` powershell
python "$script_path\generate_decoy_assays.py" `
--in "library\serum_prediction_rtcalibrated_filtered.assay.pickle"
```

Combine the target and decoy spectral libraries and convert them to peptide query parameter (PQP) format.
``` powershell
python "$script_path\convert_assays_to_OpenSWATH_library.py" `
--enable_glycoform_uis `
--in "library\serum_prediction_rtcalibrated_filtered_uis.assay.pickle" `
 (ls "library" "serum_prediction_rtcalibrated_filtered*_decoy.assay.pickle" | select -ExpandProperty FullName) `
--out "library\serum_prediction_uis.PQP"
```

Convert the retention time anchors to TraML format.
``` powershell
python "$script_path\filter_assays.py" `
--swath_windows "swath_window.tsv" `
--min_fragment_mz 200 --max_fragment_mz 2000 `
--in "library\serum_1h_consensus_rtanchor.assay.pickle"

python "$script_path\convert_assays_to_traml.py" `
--in "library\serum_1h_consensus_rtanchor_filtered.assay.pickle" `
--out "library\serum_prediction_rtanchor.traML"
```
The processed spectral library (including decoys) is saved as a PQP file. The processed anchors are saved in a TraML files. They will be the input for OpenSWATH.

### Targeted Data Extraction
Change working directory to the `mzML` subfolder. Run the OpenSWATH analysis workflow.

``` powershell
cd "mzML"

ls *.mzML -Name | % {
    OpenSwathWorkflow `
    -Library:retentionTimeInterpretation seconds `
    -RTNormalization:alignmentMethod lowess `
    -RTNormalization:estimateBestPeptides `
    -mz_extraction_window_ms1 10 -mz_extraction_window_ms1_unit ppm `
    -mz_extraction_window 20 -mz_extraction_window_unit ppm `
    -enable_ms1 true -enable_ipf true `
    -threads 4 `
    -swath_windows_file "..\swath_window.tsv" `
    -tr "..\library\serum_prediction_uis.PQP" `
    -tr_irt "..\library\serum_prediction_rtanchor.traML" `
    -in $_ `
    -out_osw $_.Replace('.mzML', '.osw') `
    -rt_extraction_window 900
}
```

### Scoring and Statistical Control
Create a subdirectory named `result` in the project directory, and move the OpenSWATH files (`*.osw`) in the subfolder. Set the working directory to the `result` subfolder.
``` powershell
mkdir "..\result" -f | Out-Null
mv *.osw "..\result"
cd "..\result"
```

Conduct semi-supervised learning and error-rate estimation.
``` powershell
pyprophet merge `
--template="..\library\serum_prediction_uis.PQP" `
--out="serum_prediction_uis.osw" `
(ls *.osw -Name)

python "$script_path\score_glycopeptide_peakgroups.py" `
--level ms2 `
--threads 4 `
--in "serum_prediction_uis.osw" `
--test

python "$script_path\score_feature_glycoform.py" `
--level ms1 `
--threads 4 `
--in "serum_prediction_uis.osw" `
--test

python "$script_path\score_feature_glycoform.py" `
--level transition `
--threads 4 `
--in "serum_prediction_uis.osw" `
--test

python "$script_path\infer_glycoforms.py" `
--in "serum_prediction_uis.osw"

python "$script_path\infer_glycopeptides.py" `
--context global `
--in "serum_prediction_uis.osw"
```
Note that the `--test` option is used to enable test mode with with fixed random seed to get reproducible results. This option should be turned off in practical scenarios.

Export the results to text report.
``` powershell
python "$script_path\export_results.py" `
--in "serum_prediction_uis.osw" `
--out "serum_prediction_uis.tsv" `
--format legacy_merged `
--glycoform --max_glycoform_qvalue 0.05 `
--max_rs_peakgroup_qvalue 0.05 `
--max_global_glycopeptide_qvalue 0.01 `
--no-transition_quantification


Perform multi-run alignment. Set `$TRIC_path` as the path to TRIC script (`feature_alignment.py`) and run TRIC.
``` powershell
$TRIC_path = "C:\Users\{User Name}\.conda\envs\{Python 3.5 Environment}\Scripts\feature_alignment.py"

python $TRIC_path `
--in "serum_prediction_uis.tsv" `
--out "serum_prediction_uis_aligned.tsv" `
--file_format openswath `
--fdr_cutoff 0.01 `
--max_fdr_quality 0.2 `
--mst:useRTCorrection True `
--mst:Stdev_multiplier 3.0 `
--method LocalMST `
--max_rt_diff 90 `
--alignment_score 0.001 `
--frac_selected 0 `
--realign_method lowess `
--disable_isotopic_grouping
```
The aligned results are saved in tab-separated format (`*_aligned.tsv`).

## DIA Analysis Using an Extended Spectral Library
### Spectral Library Prediction
In Anaconda PowerShell Prompt **Python 3.9 for DeepGlyco**, predict MS/MS spectra and iRT using an extended starting glycopeptide list.
``` powershell
python "$script_path\predict_gpms2.py" `
--pretrained "library\PXD023980_HumanSerum_gpms2_model.pt" `
--in "library\serum_extend.glycopeptides.csv" `
--out "library\serum_extend_prediction_finetune.speclib.h5"

python "$script_path\predict_gprt.py" `
--no-validate `
--pretrained "library\PXD023980_HumanSerum_gprt_model.pt" `
--in "library\serum_extend.glycopeptides.csv" `
--out "library\serum_extend_prediction_finetune.speclib.h5"

python "$script_path\convert_speclib_to_GproDIA_assays.py" `
--in "library\serum_extend_prediction_finetune.speclib.h5" `
--glycopeptide "library\serum_extend.glycopeptides.csv" `
--out "library\serum_extend_prediction.assay.pickle"
```

In Anaconda PowerShell Prompt **Python 3.5 for GproDIA**, process the predicted spectral library.
``` powershell
python "$script_path\calibrate_assays_rt.py" `
--multiple_runs `
--smooth lowess --lowess_frac 0.25 --lowess_it 3 `
--in "library\serum_extend_prediction.assay.pickle" `
--reference "library\serum_1h_consensus.assay.pickle" `
--out "library\serum_extend_prediction_rtcalibrated.assay.pickle" `
--out_anchor "library\serum_1h_consensus_rtanchor.assay.pickle"

python "$script_path\filter_assays.py" `
--swath_windows "swath_window.tsv" `
--min_fragment_mz 200 --max_fragment_mz 2000 `
--in "library\serum_extend_prediction_rtcalibrated.assay.pickle"

python "$script_path\filter_assays.py" `
--swath_windows "swath_window.tsv" `
--min_fragment_mz 200 --max_fragment_mz 2000 `
--in "library\serum_1h_consensus_rtanchor.assay.pickle"

python "$script_path\convert_assays_to_traml.py" `
--in "library\serum_1h_consensus_rtanchor_filtered.assay.pickle" `
--out "library\serum_extend_prediction_rtanchor.traML"

```

### Prelim Search
In order to save computing resources when using a large spectral library, generate a PQP for prelim search without glycoform inference.
``` powershell
python "$script_path\generate_decoy_assays.py" `
--in "library\serum_extend_prediction_rtcalibrated_filtered.assay.pickle"

python "$script_path\convert_assays_to_OpenSWATH_library.py" `
--disable_glycoform_uis `
--in "library\serum_extend_prediction_rtcalibrated_filtered.assay.pickle" `
(ls "library" "serum_extend_prediction_rtcalibrated_filtered*_decoy.assay.pickle" | select -ExpandProperty FullName) `
--out "library\serum_extend_prediction_prelim.PQP"
```

Run the OpenSWATH analysis workflow.
``` powershell
cd "mzML"

ls *.mzML -Name | % {
    OpenSwathWorkflow `
    -Library:retentionTimeInterpretation seconds `
    -RTNormalization:alignmentMethod lowess `
    -RTNormalization:estimateBestPeptides `
    -mz_extraction_window_ms1 10 -mz_extraction_window_ms1_unit ppm `
    -mz_extraction_window 20 -mz_extraction_window_unit ppm `
    -threads 4 `
    -swath_windows_file "..\swath_window.tsv" `
    -tr "..\library\serum_extend_prediction_prelim.PQP" `
    -tr_irt "..\library\serum_extend_prediction_rtanchor.traML" `
    -in $_ `
    -out_osw $_.Replace('.mzML', '.osw') `
    -rt_extraction_window 900
}
```

Score and export the prelim search result.
``` powershell
mkdir "..\prelim" -f | Out-Null
mv *.osw "..\prelim"
cd "..\prelim"

pyprophet merge `
--template="serum_extend_prediction_prelim.PQP" `
--out="serum_extend_prediction_prelim.osw" `
(ls *.osw -Name)

python "$script_path\score_glycopeptide_peakgroups.py" `
--level ms2 `
--threads 4 `
--in "serum_extend_prediction_prelim.osw" `
--test

python "$script_path\infer_glycopeptides.py" `
--context global `
--in "serum_extend_prediction_prelim.osw"

python "$script_path\export_results.py" `
--in "serum_extend_prediction_prelim.osw" `
--out "serum_extend_prediction_prelim.tsv" `
--format legacy_merged `
--max_rs_peakgroup_qvalue 0.05 `
--max_global_glycopeptide_qvalue 1 `
--no-transition_quantification
```

### Main Search
Generate a subset spectral library based on the prelim search result. This library contains identification transitions for glycoform inference.
``` powershell
cd ..

python "$script_path\subset_assays.py" `
--in "library\serum_extend_prediction_rtcalibrated_filtered.assay.pickle" `
--out "library\serum_extend_prediction_second_filtered.assay.pickle" `
--result "prelim\serum_extend_prediction_prelim.tsv"

python "$script_path\generate_glycoform_uis_assays.py" `
--swath_windows "swath_window.tsv" `
--min_fragment_mz 200 --max_fragment_mz 2000 `
--background_glycans "background_glycan.txt" `
--max_background_glycan_number 50 `
--in "library\serum_extend_prediction_second_filtered.assay.pickle"

python "$script_path\generate_decoy_assays.py" `
--in "library\serum_extend_prediction_second_filtered.assay.pickle"

python "$script_path\convert_assays_to_OpenSWATH_library.py" `
--enable_glycoform_uis `
--in "library\serum_extend_prediction_second_filtered_uis.assay.pickle" `
 (ls "library" "serum_extend_prediction_second_filtered*_decoy.assay.pickle" | select -ExpandProperty FullName) `
--out "library\serum_extend_prediction_uis.PQP"
```

Run the OpenSWATH analysis workflow.
``` powershell
cd "mzML"

ls *.mzML -Name | % {
    OpenSwathWorkflow `
    -Library:retentionTimeInterpretation seconds `
    -RTNormalization:alignmentMethod lowess `
    -RTNormalization:estimateBestPeptides `
    -mz_extraction_window_ms1 10 -mz_extraction_window_ms1_unit ppm `
    -mz_extraction_window 20 -mz_extraction_window_unit ppm `
    -enable_ms1 true -enable_ipf true `
    -threads 4 `
    -swath_windows_file "..\..\swath_window.tsv" `
    -tr "..\library\serum_extend_prediction_uis.PQP" `
    -tr_irt "..\library\serum_extend_prediction_rtanchor.traML" `
    -in $_ `
    -out_osw $_.Replace('.mzML', '.osw') `
    -rt_extraction_window 900
}
```

Score and export the main search result.
``` powershell
mkdir "..\result" -f | Out-Null
mv *.osw "..\result"
cd "..\result"

pyprophet merge `
--template="..\library\serum_prediction_uis.PQP" `
--out="serum_extend_prediction_uis.osw" `
(ls *.osw -Name)

python "$script_path\score_glycopeptide_peakgroups.py" `
--level ms2 `
--threads 4 `
--in "serum_extend_prediction_uis.osw" `
--test

python "$script_path\score_feature_glycoform.py" `
--level ms1 `
--threads 4 `
--in "serum_extend_prediction_uis.osw" `
--test

python "$script_path\score_feature_glycoform.py" `
--level transition `
--threads 4 `
--in "serum_extend_prediction_uis.osw" `
--test

python "$script_path\infer_glycoforms.py" `
--in "serum_extend_prediction_uis.osw"

python "$script_path\infer_glycopeptides.py" `
--context global `
--in "serum_extend_prediction_uis.osw"

python "$script_path\export_results.py" `
--in "serum_extend_prediction_uis.osw" `
--out "serum_extend_prediction_uis.tsv" `
--format legacy_merged `
--glycoform --max_glycoform_qvalue 0.05 `
--max_rs_peakgroup_qvalue 0.05 `
--max_global_glycopeptide_qvalue 0.01 `
--no-transition_quantification

$TRIC_path = "C:\Users\{User Name}\.conda\envs\{Python 3.5 Environment}\Scripts\feature_alignment.py"

python $TRIC_path `
--in "serum_extend_prediction_uis.tsv" `
--out "serum_extend_prediction_uis_aligned.tsv" `
--file_format openswath `
--fdr_cutoff 0.01 `
--max_fdr_quality 0.2 `
--mst:useRTCorrection True `
--mst:Stdev_multiplier 3.0 `
--method LocalMST `
--max_rt_diff 90 `
--alignment_score 0.001 `
--frac_selected 0 `
--realign_method lowess `
--disable_isotopic_grouping
```
The aligned results are saved in tab-separated format (`*_aligned.tsv`).

