# A Tri-light Warning System for Hospitalized COVID-19 Patients: Credibility-based Risk Stratification under Data Shift

This is the code for the article "A Tri-light Warning System for Hospitalized COVID-19 Patients: Credibility-based Risk Stratification under Data Shift"

### Environment and package requirements. 
Please refer to our previous repository for the package requirements. https://github.com/terryli710/COVID_19_Rapid_Triage_Risk_Predictor

### Data Preparation. 
Please refer to our previous repository for the data prepocessing pipeline (https://github.com/terryli710/COVID_19_Rapid_Triage_Risk_Predictor). We have provided the example data file (.csv) that is needed for the model to make prediction. 
1) Prepare the radiomics data according to the previously published repository and name it as "example_radiomics_pipeline_output.csv";
2) Fill in the available clinical_lab_test data based on the "example_lab_data.csv"; 

### Make prediction and output the reliability information.
Start the terminal in the directory of the model (Deployment). Run the model using the following command line.

```shell
python COVID-19_prediction --radiomics_data <save_root\final_merge_feature.csv> --lab_data <lab_input.csv>
```
