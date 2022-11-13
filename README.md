## A Tri-light Warning System for Hospitalized COVID-19 Patients: Credibility-based Risk Stratification under Data Shift

This is the code for the article "A Tri-light Warning System for Hospitalized COVID-19 Patients: Credibility-based Risk Stratification under Data Shift"

1. Environment and package requirements. Please refer to our previous repository for the package requirements. https://github.com/terryli710/COVID_19_Rapid_Triage_Risk_Predictor

2. Data Preparation. Please refer to our previous repository for the data prepocessing pipeline (https://github.com/terryli710/COVID_19_Rapid_Triage_Risk_Predictor). We have provided the example data file (.csv) that is needed for the model to make prediction. 
1) Prepare the radiomics data according to the previously published repository and name it as "radiomics_data.csv";
2) Fill in the available clinical_lab_test data based on the "example_lab_data.csv"; Please note that the clinica_lad_test items/values are listed as follows (partitioned by semi-colon):
age;sex 1=male;Coronary heart disease;Chronic liver disease;Chronic kidney disease;Chronic obstructive lung disease (COPD);Diabetes;Hypertension;Carcinoma;Fever;Cough;Myalgia;Fatigue;Headache;Nausea or vomiting;Diarrhoea;Abdominal pain;Dyspnea;WBC (White Blood Cell) Count × 10⁹/L;Neutrophil count × 10⁹/L;Lymphocyte count 0.8-3.5 × 10⁹/L;Hemoglobin 120-165 g/L;platelet 100-300 × 10⁹/L;Prothrombin time (PT);activated partial thromboplastin time (aPTT);D dimer;C-reactive protein (CRP);albumin;Alanine aminotransferase (ALT);Aspartate Aminotransferase (AST);Total bilirubin;Serum potassium;sodium;Creatinine;Creatine kinase (CK);Lactate dehydrogenase (LDH);α-Hydroxybutyrate dehydrogenase (HBDH)

3. Make prediction and output the reliability information.
Start the terminal in the directory of the model (Deployment). Run the model using the following command line.

```shell
python COVID-19_prediction --radiomics_data <save_root\final_merge_feature.csv> --lab_data <lab_input.csv>
```
