import streamlit as st
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import os
from pycaret.regression import *
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import warnings


#from pycaret.regression import setup, compare_model, pull, save_model

with st.sidebar:

    st.image("logo.png")
    st.title("Hello GM ML Users")


    choice = st.radio("목록", ["Import a file", "Exploratory Analysis","Data Preprocessing","Train&Analyze Model","Prediction","파일 다운로드(TBD)","PCA 분석(TBD)"])
    st.info("활성화 된 데이터 삭제")

    if st.button("Refresh"):

        files = ['my_best.pkl','Feature Importance.png','Prediction Error.png','new_sourcedata.csv','sourcedata.csv']
        for file in files:

            if os.path.isfile(file):
                os.remove(file)
        st.success('This is a success Refresh!', icon="✅")

#st.write("hello")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Import a file":
    st.title("Import a file")
    st.header("You are able to upload CSV files")
    file = st.file_uploader("업로드 파일")

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
    deleted_target = st.multiselect("Select Target-Variables", df.columns)

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

if choice == "Train&Analyze Model":

    if os.path.exists("new_sourcedata.csv"):
        df = pd.read_csv("new_sourcedata.csv")
    else :
        st.warning('This is a warning, Please return to "Data Preprocessing"', icon="⚠️")

    st.title("ML 학습")
    st.header("Target 값을 선택 하세요")
    target = st.selectbox("Select Target(Y 값)",df.columns)

    if st.button('Click Me'):
        setup(df, target=target, silent=True, log_experiment = True)
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

        save_model(best_model, 'my_best')

if choice == "Prediction":

    #warnings.filterwarnings('ignore')

    loaded_model = load_model('my_best')

    dataset = pd.read_csv("new_sourcedata.csv", index_col=None)

    pred = predict_model(loaded_model, data=dataset)

    st.dataframe(pred)

    #file_path = Path("my_best.pkl")





