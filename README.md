# DyMFusion-ChargeNet: Dynamic Multi-Source Fusion for Adaptive EV Charging Prediction

This repository contains the PyTorch implementation for **DyMFusion-ChargeNet**, a novel deep learning framework for dynamic and adaptive Electric Vehicle (EV) charging behavior prediction, as presented in our research paper. DyMFusion-ChargeNet is designed to effectively process heterogeneous data sources, model complex spatiotemporal dependencies, and adapt to real-time contextual shifts.

Our work is evaluated on the UrbanEV dataset, demonstrating improved predictive accuracy and adaptability, especially during simulated dynamic shock events.

## Core Contributions & Features

Our research and this implementation highlight four primary contributions:

1.  **Novel Dynamic Contextual Fusion (DCF) Mechanism**:
    *   Transcends static fusion by dynamically adjusting multi-source feature fusion using a real-time "context vector" (\texttt{event\_rt}).
    *   This vector incorporates base contextual features (e.g., extreme weather, temporal patterns) and simulated dynamic shock signals (e.g., traffic jams, large local events).
    *   Enables the model to adapt its feature weighting and integration on-the-fly, enhancing responsiveness to sudden environmental changes and improving prediction accuracy during atypical periods. Our experiments show that DCF-enabled configurations outperform static fusion counterparts, particularly during these shock events.

2.  **Comprehensive Architecture for Heterogeneous Data & Spatiotemporal Modeling**:
    *   **Multi-Source Asynchronous Feature Encoder (MAFE)**: Utilizes specialized sub-encoders (primarily MLPs in this implementation) to process diverse input data types (historical demand, weather derivatives via `event_rt`, POI counts, static station attributes, and event signals) into rich, unified representations.
    *   **Spatiotemporal Graph Attention (STGA)**: Explicitly models inter-station spatial interdependencies and their temporal evolution.
        *   Leverages a **dynamically constructed distance-based adjacency matrix** (using Haversine distance) when pre-defined graph information is sparse, proving crucial for effective spatial modeling on real-world datasets like UrbanEV.
        *   Combines Graph Attention Networks (GATConv from PyTorch Geometric) for spatial feature aggregation and LSTMs for temporal sequence modeling.

3.  **Demonstrated Superior Performance and Dynamic Adaptability**:
    *   Extensive experiments on the UrbanEV dataset (augmented with simulated shocks) show that optimized configurations of DyMFusion-ChargeNet (e.g., `NoWeatherMAFE_AllEvents` which uses DCF with engineered weather events in `event_rt`) achieve superior overall performance compared to LSTM baselines and static fusion versions.
    *   The model exhibits enhanced adaptability under simulated dynamic contextual shifts, where the DCF mechanism demonstrates clear advantages in mitigating prediction error during shock periods.

4.  **Valuable Insights from In-depth Ablation Studies**:
    *   Our code supports running various ablation scenarios, and the results validate the critical efficacy of the STGA module for spatiotemporal pattern capture.
    *   The studies underscore the significant performance gains attributable to the DCF mechanism when provided with relevant contextual signals.
    *   We also reveal nuanced insights into MAFE's application, for instance, showing that for weather data, engineered event-based features within the `event_rt` vector can be more effective for dynamic adaptation than direct MAFE processing of raw weather sequences in the current setup.

## Repository Structure

```
├── UrbanEV-main/data/        # Directory for the UrbanEV dataset CSV files (not included directly, link to dataset)
├── visualizations/           # Directory where generated plots will be saved
├── dymfusion_chargenet.py    # Main Python script with model implementation, data loading, training, evaluation, and visualization
├── README.md                 # This README file
└── requirements.txt          # (Recommended) Python dependencies
```

## Dataset

This project utilizes the **UrbanEV dataset**.
*   **Source**: Shenzhen, China
*   **Period**: September 1, 2022 - February 28, 2023
*   **Data Types**: Hourly EV charging occupancy (primary target), weather, Points of Interest (POI), station information (`inf.csv`), and an adjacency matrix (`adj.csv`).
*   **Graph Construction**: The provided `adj.csv` was found to have limited inter-zone connections. This implementation dynamically constructs a more effective graph based on Haversine distance between zones (derived from `inf.csv`) using a configurable threshold (e.g., 2.0 km).
*   **Dynamic Shocks**: The code includes functionality to simulate dynamic shock events (e.g., traffic jams, large local events) by modifying the target variable and adding corresponding indicator features to the `event_rt` vector for testing adaptive capabilities.

[**(Please include a link here to where users can download the UrbanEV dataset if it's publicly available, e.g., a Dryad link, or instructions on how to obtain it.)**](https://zenodo.org/records/14913966)

## Requirements

*   Python 3.x
*   PyTorch (e.g., 1.10 or newer)
*   NumPy
*   Pandas
*   Scikit-learn
*   Matplotlib
*   Seaborn
*   **PyTorch Geometric (PyG)**: Crucial for the GATConv layers in the STGA module. Ensure you install it according to the official instructions for your PyTorch and CUDA versions. If not installed, the STGA module will use a non-functional placeholder.

You can typically install these using pip:
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
# For PyTorch Geometric, follow official instructions:
# e.g., pip install torch_geometric
# And potentially for CUDA:
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-X.X.X+cuXXX.html
```
A `requirements.txt` file is recommended for easier environment setup.

## Running the Code

The main script `dymfusion_chargenet.py` (assuming you save the Python code with this name) handles data loading, preprocessing, model training, evaluation, and visualization for various ablation scenarios.

1.  **Prepare Data**:
    *   Download the UrbanEV dataset and place the CSV files into a directory structure like `UrbanEV-main/data/`.
    *   Update the `DATA_PATH` variable in the script if your data directory is different.

2.  **Configure Experiment**:
    *   Modify global variables at the top of the script for `EPOCHS`, `LR`, `BATCH_SIZE`, `DISTANCE_THRESHOLD_KM`, etc.
    *   Adjust the `ablation_scenarios` dictionary in the `if __name__ == "__main__":` block to select which model configurations to run. Each scenario is defined by an `ablation_cfg` dictionary that controls which modules/features are active.

3.  **Execute**:
    ```bash
    python dymfusion_chargenet.py
    ```

4.  **Outputs**:
    *   Training progress and evaluation metrics for each scenario will be printed to the console.
    *   Resulting plots (overall performance, ablation impacts, dynamic adaptation curves, shock vs. normal performance) will be saved as PDF files in a `visualizations/` directory created in the same location as the script.

## Key Code Components

*   **`load_and_preprocess_urban_ev()`**: Handles data loading, cleaning, feature engineering (including enhanced `event_rt` and dynamic shock simulation), graph construction, and sequence generation.
*   **`MAFESubEncoder`**: Basic MLP-based encoder for individual data sources.
*   **`STGA`**: Implements Spatiotemporal Graph Attention using GATConv and LSTM.
*   **`DCF`**: Implements the Dynamic Contextual Fusion mechanism using gating.
*   **`PredictiveDecoder`**: MLP for final predictions.
*   **`DyMFusionChargeNet`**: The main model class, integrating MAFE, STGA, DCF, and PD. It's adaptable via an `ablation_cfg` for running different experimental scenarios.
*   **`LSTMBaseline`**: A simple LSTM baseline model.
*   **Visualization Functions**: `plot_overall_performance`, `plot_ablation_bar`, `plot_dynamic_adaptation_curves` for generating result figures.
*   **Ablation Loop**: The main execution block iterates through predefined `ablation_scenarios` to test different model configurations.

## Interpreting Results

*   **Overall Performance Table/Plot**: Compare MAE/RMSE across all run scenarios.
*   **Ablation Plots**:
    *   **Impact of STGA**: Compare scenarios with and without STGA against a non-spatial baseline like LSTM.
    *   **Impact of DCF**: Compare "FullModel" (DCF enabled) against "StaticFusion" (DCF disabled).
    *   **Impact of MAFE Features**: Compare scenarios where specific MAFE inputs (POI, Weather MAFE) are ablated.
*   **Dynamic Adaptation Curves**: Visually inspect how well `DyMFusion_ChargeNet` (FullModel) tracks actual demand during simulated shock periods compared to `StaticFusion` and `LSTM`. The red shaded areas indicate periods where the shock signal in `event_rt` is active.
*   **Shock vs. Normal Performance Plot**: Quantitatively compare MAE during simulated shock periods versus normal periods for key models.

