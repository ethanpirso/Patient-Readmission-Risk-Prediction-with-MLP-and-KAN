# Patient-Readmission-Risk-Prediction-with-MLP-and-KAN
This project focuses on predicting patient readmission risk using two distinct neural network architectures: the traditional Multi-Layer Perceptron (MLP) and the novel Kolmogorov-Arnold Network (KAN). The project aims to compare the performance of these architectures in terms of accuracy and interpretability on a medical dataset.

## Dataset
The dataset `patient_data.csv` provides a detailed record of patient encounters in a healthcare setting, focusing on aspects relevant to hospital readmission and intensive care unit (ICU) stays. Each row corresponds to a unique patient encounter, with the columns containing the following information:

1. **patientID**: Unique identifier for each patient.
2. **female**: Binary indicator for the patient's gender, where 1 represents female and 0 represents male.
3. **age**: The age of the patient at the time of hospital admission.
4. **hospAdatetime**: The date and time the patient was admitted to the hospital.
5. **hospDdatetime**: The date and time the patient was discharged from the hospital.
6. **icuAdatetime**: The date and time the patient was admitted to the ICU.
7. **icuDdatetime**: The date and time the patient was discharged from the ICU.
8. **BinDaysfromHostoICUAdmit**: Binned number of days from hospital admission to ICU admission.
9. **hosplos**: Length of stay in the hospital, measured in days.
10. **iculos**: Length of stay in the ICU, measured in days.
11. **hospdeath**: Binary indicator for death in the hospital, where 1 indicates the patient died in the hospital.
12. **icudeath**: Binary indicator for death in the ICU, where 1 indicates the patient died in the ICU.
13. **apache2**: APACHE II score, a severity of disease classification system.
14. **apache4**: APACHE IV score, an updated severity of disease classification system.
15. **ADMITLOCATIONTXT**: Textual description of the patient's admission location (e.g., emergency department, transfer from another hospital).
16. **PrimaryDiagnosis**: The primary diagnosis for the patient's hospital stay.
17. **INITIALADMIT**: Binary indicator for whether this was the patient's initial admission.
18. **DISCHDISPOSITIONTXT**: Textual description of the patient's discharge disposition (e.g., discharged to home, transferred to another facility).
19. **Overflow**: Binary indicator for whether the patient was admitted during a time of overflow.
20. **Mechanical.Ventilation**: Binary indicator for whether the patient required mechanical ventilation.
21. **vasoactive**: Binary indicator for whether the patient received vasoactive drugs.
22. **Oncatheter**: Binary indicator for whether the patient was on a catheter.
23. **OnInsulin**: Binary indicator for whether the patient was on insulin.
24. **readmit72**: Binary indicator for readmission within 72 hours, where 1 indicates readmission.
25. **readmit72_next**: Binary indicator for the next readmission within 72 hours.
26. **teamtransfer**: Binary indicator for whether there was a team transfer during the patient's stay.

This dataset is structured to facilitate the analysis of factors influencing hospital readmission and ICU stays, including patient demographics, clinical interventions, and outcomes.

## Project Structure
This repository contains several key components essential for the execution and understanding of the Patient-Readmission-Risk-Prediction project using MLP (Multi-Layer Perceptron) and KAN (Kolmogorov-Arnold Network) models. Below is a detailed outline of the repository's structure and contents:

### `main.ipynb`
- **Overview**: This Jupyter notebook serves as the central hub for the project, where the models are trained, evaluated, and compared.
- **Contents**: It includes data preprocessing steps, model training and testing phases for both MLP and KAN models, and the visualization of results. Key results indicate that the KAN model offers a promising alternative to the traditional MLP, with specific metrics detailed in the notebook.

### `MLP.py`
- **Overview**: Contains the implementation of the Multi-Layer Perceptron model.
- **Contents**: This Python script defines the MLP class with its architecture, forward pass, training, and testing steps. It utilizes PyTorch and PyTorch Lightning for model operations.

### `KAN.py`
- **Overview**: Houses the implementation of the Kolmogorov-Arnold Network model.
- **Contents**: Similar to `MLP.py`, this script defines the KAN class, including its unique architecture and the necessary methods for training and testing the model using PyTorch and PyTorch Lightning.

### `preprocessing.py`
- **Overview**: Dedicated to data preprocessing steps required before feeding the data into the models.
- **Contents**: Implements functions for cleaning the data, handling missing values, encoding categorical variables, and normalizing the features. It ensures the data is in the right format and ready for model consumption.

### Results Found in `main.ipynb`
The notebook presents a comparative analysis of the MLP and KAN models in terms of accuracy, loss, and computational efficiency. Key findings include:
- The KAN model demonstrates a slightly higher accuracy compared to the MLP model in predicting patient readmission risk.
- Both models show competitive performance, but KAN offers better interpretability, which is crucial for medical applications.
- The detailed performance metrics and visualizations in the notebook provide insights into the strengths and weaknesses of each model.

This repository aims to contribute to the ongoing research in medical informatics by exploring advanced neural network architectures for predicting patient readmission risks. The comparative analysis between MLP and KAN models sheds light on the potential of using novel neural network designs in healthcare applications.

## Authors
Ethan Pirso