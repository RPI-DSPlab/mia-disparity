# Membership Inference Attacks as Privacy Tools: Reliability, Disparity and Ensemble

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15491989.svg)](https://doi.org/10.5281/zenodo.15491989) [![arXiv](https://img.shields.io/badge/arXiv-2506.13972-b31b1b.svg)](https://arxiv.org/abs/2506.13972)

 This is a repository for the paper "Membership Inference Attacks as Privacy Tools: Reliability, Disparity and Ensemble", accepted by ACM CCS 2025. This is a cleaned-up version of our [MIAE framework repository](https://github.com/RPI-DSPlab/MIAE) to contain only essential scripts for reproducing results in this paper. Main scripts are in the [experiment/mia_comp](experiment/mia_comp) directory. Our paper is available at [paper.pdf](./paper.pdf).

 Note that throughout this repo, we refer coverage and stability (2 definition defined in the paper) as union and intersection, respectively. We also refer instances and seeds, since each instance is `prepared` with a different seed.



 ## Table of Contents

- [Membership Inference Attacks as Privacy Tools: Reliability, Disparity and Ensemble](#membership-inference-attacks-as-privacy-tools-reliability-disparity-and-ensemble)
  - [Table of Contents](#table-of-contents)
  - [Abstract](#abstract)
  - [Set up the environment](#set-up-the-environment)
  - [Quick fix for relative imports](#quick-fix-for-relative-imports)
    - [Obtaining Membership Predictions - `obtain_pred.py`](#obtaining-membership-predictions---obtain_predpy)
  - [Instance Level Comparisons](#instance-level-comparisons)
    - [Disparity Graphs - `obtain_graph.py`](#disparity-graphs---obtain_graphpy)
    - [Jaccard Similarity Between Attacks - `obtain_jaccard.py`](#jaccard-similarity-between-attacks---obtain_jaccardpy)
    - [Importing and partitioning the CINIC10 dataset - `process_CINIC10.ipynb`](#importing-and-partitioning-the-cinic10-dataset---process_cinic10ipynb)
    - [Top-k Class-NN Distribution Shift - `same_attack_different_signal/same_attack_different_signal.ipynb`](#top-k-class-nn-distribution-shift---same_attack_different_signalsame_attack_different_signalipynb)
    - [Disparity Empirical Analysis - `disparity_empirical_analysis.ipynb`](#disparity-empirical-analysis---disparity_empirical_analysisipynb)
  - [`/ensemble` Directory](#ensemble-directory)
    - [Obtain Ensemble predictions - `max_ensemble_low_fpr.ipynb`](#obtain-ensemble-predictions---max_ensemble_low_fpripynb)
    - [ROC Curves of Ensemble Attacks - `ensemble_roc.py`](#roc-curves-of-ensemble-attacks---ensemble_rocpy)
    - [Plot Performance Chart - `ensemble_performance.ipynb`](#plot-performance-chart---ensemble_performanceipynb)
 
 ## Abstract
 
 > Membership inference attacks (MIAs) pose a significant threat to the privacy of machine learning models and are widely used as tools for privacy assessment, auditing, and machine unlearning. While prior MIA research has primarily focused on performance metrics such as AUC, accuracy, and TPR@low FPR—either by developing new methods to enhance these metrics or using them to evaluate privacy solutions—we found that it overlooks the disparities among different attacks. These disparities, both between distinct attack methods and between multiple instantiations of the same method, have crucial implications for the reliability and completeness of MIAs as privacy evaluation tools. In this paper, we systematically investigate these disparities through a novel framework based on coverage and stability analysis. Extensive experiments reveal significant disparities in MIAs, their potential causes, and their broader implications for privacy evaluation.
 To address these challenges, we propose an ensemble framework with three distinct strategies to harness the strengths of state-of-the-art MIAs while accounting for their disparities. This framework not only enables the construction of more powerful attacks but also provides a more robust and comprehensive methodology for privacy evaluation.



 ⚠️ **NOTE**: To be able to set up the directory correctly, please replace the `DATA_DIR` in all the scripts with the path to the directory where you want to store the attack predictions and results. We also recommend to run all scripts (especially those bash script) in the `miae/experiment/mia_comp` directory.

 ⚠️ **NOTE**: Most bash scripts have a different config set by commenting/uncommenting the lines. For example, in `experiment_scripts/obtain_venn.sh`, you can set the config of the venn diagram by commenting and uncommenting the lines. The same applies to other bash scripts.


 ## Set up the environment
 ```bash
 conda env create -f miae_env.yml
 conda activate miae
 ```
 
 ## Quick fix for relative imports
 
 go to the root directory of the repo and run:
 ```bash
 pip install -e .
 ```
 
 -------------------
 ###  Obtaining Membership Predictions - `obtain_pred.py`
 
 1. Initialize the specified target model and the target dataset.
 2. Split the target dataset into the target dataset and the auxiliary dataset.
 3. Train the target model on the target dataset.
 4. Prepare the specified MIAs with the (black box access) target model and the auxiliary dataset.
 5. Save the prediction on the target dataset.
 
**Workflow diagram:**  
![obtain_pred_fpr_workflow](./obtain_pred_fpr_workflow.png)


 This file is being called by `experiment_script/prepare_target.sh` to prepare and save the target model and datasets. And it's also called by `experiment_scripts/obtain_pred.sh` to run the experiments.
 
 Usage:
 
 1. Prepare the target datasets and target models
     ```bash
     bash experiment_scripts/prepare_target.sh
     ```
 2. Train attacks on the target models and target datasets, then save predictions
     ```bash
    bash experiment_scripts/obtain_pred.sh [seed]
    ```
    where the [seed] is the seed of that instance.
 
 to launch multiple instances at one time:
      ```bash
    bash run_multi_seed.sh {0..5}
    ```
    0..5 is the range of the seeds you want to run.

We also provide the intermediate results of the target model, target datasets, and the predictions from multiple different instances in the `experiment/mia_comp/target_model` directory. You can use them to skip the preparation step and directly run the attack predictions. The intermediate results are available at Huggingface [here](https://huggingface.co/datasets/ZhiqiEliWang/mia-disparity).
 
 -------------------
  ## Instance Level Comparisons
 
 ### Disparity Graphs - `obtain_graph.py`
 The `obtain_graph.py` script is designed to load data, generate various plots, and evaluate metrics. 
 The code is divided into three primary categories: Data Loading, Plot Diagram, and Evaluation.
 
 - Data Loading
 
    - **`load_and_create_predictions`**: Loads data and create prediction object used in later evaluations and plotting.
    - **`load_diff_distribution`**: Loads data and create prediction object for *same attack different distribution* case.
 
 - Plot Diagram
    
    - **`plot_venn`**: Plots a Venn diagram for comparisons between attacks.
       ```bash
       bash experiment_scripts/obtain_venn.sh 
        ``` 
    - **`plot_auc`**: Plots an AUC diagram (Area Under the Curve)  for different models or attacks.
       ```bash
      bash experiment_scripts/obtain_auc.sh
        ```
    - **`multi_seed_convergence`**: Visualizes the convergence across multiple seeds for the model or experiment.   
       ```bash
      bash experiment_scripts/obtain_multi_seed_conv.sh
        ```


 ### Jaccard Similarity Between Attacks - `obtain_jaccard.py`
   The `obtain_jaccard.py` is designed to save the Jaccard similarity between different MIAs and plot a heatmap to visualize the Jaccard similarity matrix. Before running this shell script, make sure you have already have the
   pair-wise jaccard similarity, which can be calculated via running obtain_venn.sh.
   ```bash
   bash experiment_scripts/obtain_jaccard.sh
   ```
    
 ### Importing and partitioning the CINIC10 dataset - `process_CINIC10.ipynb`
   The `process_CINIC10.ipynb` is designed to process the CINIC10 dataset 30,000 ImageNet samples and 30,000 CIFAR10 samples. Run it to make sure cinic10 is available in the `DATA_DIR`. 

 ### Top-k Class-NN Distribution Shift - `same_attack_different_signal/same_attack_different_signal.ipynb`

 To make sure this work, you need to run the `obtain_pred.py` script to prepare `top_k_shokri`. Make sure you replace the k with the number of top-k logits you are using for the prediction files.

 This notebook is designed to compare the performance of the same attack on different signals. Corresponding to the paper's *Attack Signals of A-covered Samples*. This directory also contains top-x shokri, which inherits from the original Shokri's attack in our package but limits the logits to the top-x logits. To plot Figure 13 in the paper, you can run `obtain_graph.py` with top-k shokri are 
 
 ### Disparity Empirical Analysis - `disparity_empirical_analysis.ipynb`
 
 This notebook is designed to analyze the disparity of MIAs in the empirical study. Corresponding to the paper's Section 4.4.
 
 ## `/ensemble` Directory
 
 This directory contains the code for the ensemble strategies proposed in the paper: Coverage Ensemble and Stability ensemble. 
 
 ### Obtain Ensemble predictions - `max_ensemble_low_fpr.ipynb`
   This notebook is designed to perform Coverage Ensemble and Stability Ensemble. It starts with thresholding the predictions of the base instances at the same low FPR, then ensemble the predictions follows our paper's definition of 2 step ensemble approach.
 
 ### ROC Curves of Ensemble Attacks - `ensemble_roc.py` 
   Ensemble roc samples n thresholds for n FPRs for each base instance. Then each attack goes through the steps of ensemble in `max_ensemble_low_fpr.ipynb` for n times with different thresholds to get n samples for each ensemble TPR@FPR. It also calculates the AUC, ACC, and TPR@Low FPR for each ensemble. To run it for multiple configurations, you can run the `ensemble/obain_roc.sh` script.
 
 ### Plot Performance Chart - `ensemble_performance.ipynb` 
 
 This notebook is designed to compare the performance of the ensemble strategies proposed in the paper. It organizes the performance result (TPR@low FPR, auc, acc) with respect to the number of instances used in the ensemble. As seen in the paper's Table 1.
 
 ___
 
