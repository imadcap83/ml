import streamlit as st
from streamlit_pandas_profiling import st_profile_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import lime
from lime import lime_tabular
import joblib
import streamlit.components.v1 as components
import scipy.stats as stats
import pandas as pd
#import pandas_profiling
import datetime
from pathlib import Path
import os
import warnings
import ydata_profiling

#from pycaret.regression import setup, compare_model, pull, save_model

st.set_page_config(
    page_title="Machine learning optimization",
    page_icon="random")

with st.sidebar:

    st.image("logo.png")
    st.title("Hello GM ML Users")


    choice = st.radio("Exploratory Data Analysis",
                      ["Import a file", "Exploratory Analysis","Data Preprocessing","Complete Button"],index=0)


    reg = st.radio("Regression",
                   ["Select below radio button 👇","Train&Analyze Model",
                       "Prediction&Explaining", "ANOVA (ANalysis Of VAriance)",
                       "PCA (Principal Component Analysis) -TBD"])

    cfc = st.radio("Classification",
                   ["Select below radio button 👇", "Train&Analyze Model","Prediction&Explaining"])

    st.info("활성화 된 데이터 삭제")

    if st.button("Refresh"):

        files = ['anova_each.png','my_best.pkl','Feature Importance.png','Prediction Error.png','new_sourcedata.csv',
                 'sourcedata.csv','pred.csv','Confusion Matrix.png','Decision Boundary.png','final_model.pkl']
        for file in files:

            if os.path.isfile(file):
                os.remove(file)
            else :
                pass
        st.success('This is a success Refresh!', icon="✅")

#st.title('Machine Learning Application Quickview')

#st.markdown("""
#* Use the menu at left to select data and set plot parameters
# * Use your data as *.CSV
#""")

#st.text('This is some text.')

#st.write("hello")
if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Import a file":
    st.title("Import a file")
    st.header("You are able to upload CSV files")
    file = st.file_uploader("Upload a file")

    if file:
        df = pd.read_csv(file, index_col=None)

        for column in df.columns:

            if column == 'Unnamed: 0':
                df.drop(columns=[column], inplace=True)
                #df = pd.read_csv("sourcedata.csv")

        st.dataframe(df)
        df.to_csv("sourcedata.csv", index=None)

if choice == "Exploratory Analysis":

    if os.path.exists("sourcedata.csv"):
        df = pd.read_csv("sourcedata.csv", index_col=None)
        profile_report = df.profile_report()
        st_profile_report(profile_report)

#상관관계 차트
        dataset_corr = df.corr().round(4)
        mask = np.zeros_like(dataset_corr.round(4))
        mask[np.triu_indices_from(mask)] = True

        with sns.axes_style("whitegrid"):
            f, ax = plt.subplots(figsize=(12, 10))
            ax = sns.heatmap(dataset_corr.round(4), mask=mask, vmax=1, center=0, vmin=-1, square=True, cmap='PuOr',
                             linewidths=.5, annot=True, annot_kws={"size": 12}, fmt='.1f')
            plt.title('Heatmap (Correlations) of Features in the Dataset', fontsize=15)
            plt.xlabel('Features', fontsize=15)
            plt.ylabel('Features', fontsize=15)

        plt.show()

        st.pyplot(f)

    else:
        st.title("데이터 분석")
        st.info()

if choice == "Data Preprocessing":

    df = pd.read_csv("sourcedata.csv")

    #st.header("Check columns of dataset and view first few observations to ensure data loaded correctly")
    #st.dataframe(df.head(5))

    st.header("If you want to delete variables, Select variables")
    deleted_target = st.multiselect("Select Remove-Variables", df.columns)

    #st.write('You selected:', deleted_target)

    df.drop(labels=deleted_target, axis=1, inplace=True)

    st.dataframe(df.head(5))

    if st.button('Save Dataset'):

        st.write('Created your new dataset')

        df.to_csv("new_sourcedata.csv", index=None)

        if os.path.exists("new_sourcedata.csv"):
            df = pd.read_csv("new_sourcedata.csv", index_col=None)
            profile_report = df.profile_report()
            st_profile_report(profile_report)

if choice == "Complete Button":
    st.title('')

if reg == "Train&Analyze Model":

    from pycaret.regression import *

    if os.path.exists("new_sourcedata.csv"):
        df = pd.read_csv("new_sourcedata.csv")
    else :
        st.warning('This is a warning, Please return to "Data Preprocessing"', icon="⚠️")

    st.title("Training data (or a training dataset)")
    st.header("Target (Response Y) need to be selected")
    target = st.selectbox("Select Target(Response Y)",df.columns)

    size = st.number_input('Input Train Set [ 0.5 ~ 1.0]')
    st.write(size)

    normalization = st.selectbox(
        'Normalization is to rescale the values of numeric columns in the dataset without distorting differences in the ranges of values or losing information',
        ('False', 'True'))

    st.write('You selected:', normalization)

    if normalization == 'True':

        normalization = True

        #normalize_method = st.selectbox(
        #    " Defines the method to be used for normalization. Z score : calculated as z = (x – u) / s. Minmax : scales in the range of [0 , 1]. Maxabs : scales in the range of [-1 , 1].",
        #    ('zscore', 'minmax', 'maxabs'))

        #st.write('You selected:', normalize_method)

    else :
        normalization = False


    transformation = st.selectbox(
        'Transformation changes the shape of the distribution such that the transformed data can be represented by a normal or approximate normal distribution',
        ('False', 'True'))

    st.write('You selected:', transformation)

    if transformation == 'True':

        transformation = True

        #transformation_method = st.selectbox(
        #    "Defines the method for transformation. ",
        #    ('yeo-johnson', 'quantile'))

        #st.write('You selected:', transformation_method)

    else :
        transformation = False


    if st.button('Click Me'):
        setup(df, target=target, log_experiment = True, train_size=size, normalize = normalization, transformation = transformation)

        setup_df = pull()

        st.info("Setup Succesfully Completed")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        #save_model(best_model, 'best_model')

        st.info("The performance of all the model ")

        st.dataframe(compare_df)

        #if st.selectbox(best_model):

        fig = plt.figure(figsize=(9, 10))
        ax = fig.add_subplot(2, 1, 1)
        plot_model(best_model, plot='error', save=True, verbose=False)

        ax = fig.add_subplot(2, 1, 2)
        plot_model(best_model, plot='feature', save=True, verbose=False)
        #plt.savefig('plots_pred.png', dpi=300, pad_inches=0.25)
        plt.show()

        image = Image.open('Prediction Error.png')
        st.image(image, caption='Prediction Error')

        image = Image.open('Feature Importance.png')
        st.image(image, caption='Feature Importance')

        finalize_model = finalize_model(best_model)
        joblib.dump(finalize_model, 'final_model.pkl')

        save_model(best_model, 'my_best')



if reg == "Prediction&Explaining":

    from pycaret.regression import *

    warnings.filterwarnings('ignore')

    st.title("Import a prediction file")
    st.header("You are able to upload CSV files")
    pred_file = st.file_uploader("Upload a file")

    if pred_file :

        pred_df = pd.read_csv(pred_file, index_col=None)

        loaded_model = load_model('my_best')

        #dataset = pd.read_csv("new_sourcedata.csv", index_col=None)

        pred = predict_model(loaded_model, data=pred_df)

        st.dataframe(pred)
        pred_csv = pred.to_csv("pred.csv", index=None)

        with open('pred.csv', 'rb') as f:
            filename = 'prediction_' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M') + '.csv'
            st.download_button('Download Model', f, file_name=filename)

        col_num = st.text_input('Column Number', 'ex) 1 or 2 or 3... Please delete and input a number')
        st.write('The selected Column Number is', col_num)

        option = st.selectbox(
            'How would you like to be explaining the predictions',
            ('Select....','Classification','Regression'))

        #st.write('You selected:', option)
        if option == 'Regression':

            pred_data = pd.read_csv('pred.csv')
            X_train = pred_data.drop('Label', axis=1)
            y_train = pred_data['Label']
            y_test = X_train
            X_test = y_train

            f_model = joblib.load('final_model.pkl')

            explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                               feature_names=X_train.columns.values.tolist(),
                                                               class_names=['Label'], verbose=True,
                                                               mode='regression')

            model = f_model.fit(X_train, y_train)

            j = int(col_num)
            exp = explainer.explain_instance(X_train.values[j], model.predict, num_features=len(X_train.columns))

            exp.save_to_file('exp.html')

            HtmlFile = open("exp.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            #print(source_code)
            components.html(source_code, height =400, width=1000)

            #st.markdown('exp.html', unsafe_allow_html=True)

    else :
        st.warning('This is a warning, Please Upload a your file for prediction', icon="⚠️")

if reg == "ANOVA (ANalysis Of VAriance)":

    st.title("Analysis of Variance (ANOVA)")
    st.header("A statistical formula is used to compare variances across the means (or average) of different groups")

    if st.button('Click Me'):

        if os.path.exists("pred.csv"):

            anova_df = pd.read_csv('pred.csv')
            quantitative = [f for f in anova_df.columns if anova_df.dtypes[f] != 'object']
            quantitative.remove('prediction_label')

            qualitative = [f for f in anova_df.columns if anova_df.dtypes[f] != 'object']
            qualitative.remove('prediction_label')

            def anova(frame):
                anv = pd.DataFrame()
                anv['feature'] = qualitative
                pvals = []
                for c in qualitative:
                    samples = []
                    for cls in frame[c].unique():
                        s = frame[frame[c] == cls]['prediction_label'].values
                        samples.append(s)
                    pval = stats.f_oneway(*samples)[1]  # P-value 값
                    pvals.append(pval)
                anv['pval'] = pvals
                return anv.sort_values('pval')

            a = anova(anova_df)
            a['disparity'] = np.log(1. / a['pval'].values)
            fig = plt.figure(figsize=(10, 4))
            sns.barplot(data=a, x='feature', y='disparity')
            x = plt.xticks(rotation=90)

            st.subheader('Main effect plot')
            st.pyplot(fig)
            st.subheader('Main effect data frame')
            st.dataframe(a)

            def pairplot(x, y, **kwargs):
                ax = plt.gca()
                ts = pd.DataFrame({'time': x, 'val': y})
                ts = ts.groupby('time').mean()
                ts.plot(ax=ax)
                plt.xticks(rotation=90)

            f = pd.melt(anova_df, id_vars=['prediction_label'], value_vars=quantitative)
            fig_1 = plt.figure(figsize=(10, 4))
            g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False)
            g1 = g.map(pairplot, "value", "prediction_label")
            g1.savefig('anova_each.png')

            image_g1 = Image.open('anova_each.png')

            st.subheader('Main effect - One of each')
            st.image(image_g1)
    else :

        st.info('This is a message, Please click "Click Me"', icon="ℹ️")
    #file_path = Path("my_best.pkl")

if cfc == "Train&Analyze Model":
    from pycaret.classification import *

    if os.path.exists("new_sourcedata.csv"):
        df = pd.read_csv("new_sourcedata.csv")
    else :
        st.warning('This is a warning, Please return to "Data Preprocessing"', icon="⚠️")

    st.title("Training data (or a training dataset)")
    st.header("Target (Response Y) need to be selected")
    target = st.selectbox("Select Target(Response Y)",df.columns)

    if st.button('Click Me'):
        setup(df, target=target, log_experiment = True)
        setup_df = pull()

        st.info("Setup Succesfully Completed")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        #save_model(best_model, 'best_model')

        st.info("The performance of all the model ")

        st.dataframe(compare_df)

        #if st.selectbox(best_model):

        #fig = plt.figure(figsize=(9, 10))
        #ax = fig.add_subplot(2, 1, 1)
        plot_model(best_model, plot='confusion_matrix', save=True, verbose=False)

        #ax = fig.add_subplot(2, 1, 2)
        plot_model(best_model, plot='feature', save=True, verbose=False)
        #plt.savefig('plots_pred.png', dpi=300, pad_inches=0.25)

        plot_model(best_model, plot='boundary', save=True, verbose=False)

        #plt.show()

        image = Image.open('Confusion Matrix.png')
        st.image(image, caption='Confusion Matrix')

        image = Image.open('Feature Importance.png')
        st.image(image, caption='Feature Importance')

        image = Image.open('Decision Boundary.png')
        st.image(image, caption='Decision Boundary')

        finalize_model = finalize_model(best_model)
        joblib.dump(finalize_model, 'final_model.pkl')

        save_model(best_model, 'my_best')

if cfc == "Prediction&Explaining":

    from pycaret.classification import *

    warnings.filterwarnings('ignore')

    st.title("Import a prediction file")
    st.header("You are able to upload CSV files")
    pred_file = st.file_uploader("Upload a file")

    if pred_file :

        pred_df = pd.read_csv(pred_file, index_col=None)

        loaded_model = load_model('my_best')

        #dataset = pd.read_csv("new_sourcedata.csv", index_col=None)

        pred = predict_model(loaded_model, data=pred_df)

        st.dataframe(pred)
        pred_csv = pred.to_csv("pred.csv", index=None)

        with open('pred.csv', 'rb') as f:
            filename = 'prediction_' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M') + '.csv'
            st.download_button('Download Model', f, file_name=filename)

        col_num = st.text_input('Column Number', 'ex) 1 or 2 or 3... Please delete and input a number')
        st.write('The selected Column Number is', col_num)

        option = st.selectbox(
            'How would you like to be explaining the predictions',
            ('Select....','Classification','Regression'))

        #st.write('You selected:', option)
        if option == 'Classification':

            pred_data = pd.read_csv('pred.csv')
            X_train = pred_data.drop('Label', axis=1)
            y_train = pred_data['Label']
            y_test = X_train
            X_test = y_train

            f_model = joblib.load('final_model.pkl')

            explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                               feature_names=X_train.columns.values.tolist(),
                                                               class_names=['Label'],
                                                               mode='classification',feature_selection= 'auto',
                                                               kernel_width=None,discretize_continuous=True)

            model = f_model.fit(X_train, y_train)

            j = int(col_num)
            exp = explainer.explain_instance(X_train.values[j], model.predict, num_features=len(X_train.columns))

            exp.save_to_file('exp.html')

            HtmlFile = open("exp.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            #print(source_code)
            components.html(source_code, height =400, width=1000)

            #st.markdown('exp.html', unsafe_allow_html=True)

    else :
        st.warning('This is a warning, Please Upload a your file for prediction', icon="⚠️")



