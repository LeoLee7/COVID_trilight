# Convert input
import warnings
import pandas as pd
import numpy as np
import argparse
import joblib
import pickle
warnings.simplefilter('ignore', UserWarning)

# load file
# YOU MAY CHANGE THIS LINE
radiomics_data = pd.read_csv("example_radiomics_pipeline_output.csv")
# YOU MAY CHANGE THIS LINE
lab_data = pd.read_csv('example_lab_input.csv')

# YOU SHOULD NOT CHANGE THE FOLLOWING LINES
clinlab = pd.read_csv("example_median.csv")

del_col_name = ['ct_path','study_id','series_id','diagnostics_Versions_PyRadiomics','diagnostics_Versions_Numpy',
                'diagnostics_Versions_SimpleITK','diagnostics_Versions_PyWavelet','diagnostics_Versions_Python',
                'diagnostics_Configuration_Settings','diagnostics_Configuration_EnabledImageTypes',
                'diagnostics_Image-original_Hash','diagnostics_Image-original_Dimensionality',
                'diagnostics_Image-original_Spacing','diagnostics_Image-original_Size','diagnostics_Mask-original_Hash',
                'diagnostics_Mask-original_Spacing','diagnostics_Mask-original_Size',
                'diagnostics_Mask-original_BoundingBox','diagnostics_Mask-original_CenterOfMassIndex',
                'diagnostics_Mask-original_CenterOfMass']

# All the numeric columns convert to numeric
def lesion_to_patient(data):
    del_col_name.append("patient_id")
    Xcols = list(set(data.columns) - set(del_col_name))
    data[Xcols] = data[Xcols].apply(pd.to_numeric)
    feature_name_list = ['mean', 'std', 'medium', 'skew', '75', '25']
    X_mean = data.groupby('patient_id')[Xcols].mean()
    X_std = data.groupby('patient_id')[Xcols].std()[Xcols]
    X_medium = data.groupby('patient_id')[Xcols].median()[Xcols]
    X_skew = data.groupby('patient_id')[Xcols].skew()[Xcols]
    X_75 = data.groupby('patient_id')[Xcols].quantile(q=0.75)[Xcols]
    X_25 = data.groupby('patient_id')[Xcols].quantile(q=0.25)[Xcols]
    feature_list = [X_mean, X_std, X_medium, X_skew, X_75, X_25]
    for i in range(len(feature_name_list)):
        name_dict = {}
        for name in Xcols:
            name_dict[name] = name + '_' + feature_name_list[i]
        feature_list[i] = feature_list[i].rename(columns=name_dict)
    X_final_no_count = pd.concat(feature_list, axis=1)
    lesion_count = data.groupby('patient_id')['log-sigma-2-0-mm-3D_firstorder_Skewness'].count()
    lesion_count.name = 'lesion count'
    X_final = X_final_no_count.merge(lesion_count, on='patient_id')
    return X_final.fillna(0)


def radiomicsClinLab(radiomics_data, lab_data):
    radiomics_data = pd.read_csv(radiomics_data) #Load the radiomics data
    lab_data = pd.read_csv(lab_data) #Load the clinical_lab_data
    patient_input = lesion_to_patient(radiomics_data) #Group the radiomics features by patients

    ##RadioClinLab Model
    ss = joblib.load("feature_ss.joblib")
    feature_name = joblib.load("feature_names.joblib")

    # Imputation of the missing data with median value
    lab_median = pd.read_csv('example_median.csv')
    lab_median = lab_median.drop(['patient_id'], axis=1)
    lab_data = lab_data.set_index('patient_id')
    columns = lab_median.columns
    for item in columns:
        if np.isnan(lab_data[item].values[0]):
            lab_data[item].values[0] = lab_median[item].values[0]
            print('Impute: ', item)

    reordered = patient_input[feature_name]
    patient_input_radioclinlab = pd.concat([reordered, lab_data], axis=1, ignore_index=True)
    test_X_std = ss.transform(patient_input_radioclinlab)

    ICU_clfmodel = pickle.load("cpsc_predictor.pkl")
    ICU_fsmodel = joblib.load("feature_select_lasso.joblib")
    ICU_x = ICU_fsmodel.transform(test_X_std)

    # Make prediction based on CPSC
    ICU_prediction, ICU_probability = ICU_clfmodel.predict_SC(ICU_x)
    _, forced_prediction,credibility,_,p_value=cpsc.predict_CPSC(x_test_lasso)



    print('Model Prediction: \nICU: {}\n Model Predicted Probability: \n ICU: {%.3f} \n Model Prediction Credibility: \n ICU: {%.3f}'.format(ICU_prediction, ICU_probability, credibility))

def parse_opts():
    '''
    set the argument for the project
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--radiomics_data', default='', type=str, help='the path of radiomics data')
    parser.add_argument('--lab_data', default='', type=str, help='the path of clinical and lab data')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # command line
    opt = parse_opts()
    radiomics_data = opt.radiomics_data
    lab_data = opt.lab_data
    radiomicsClinLab(radiomics_data, lab_data)