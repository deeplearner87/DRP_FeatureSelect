import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.io as pio
pio.renderers.default = 'iframe'
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.write("""
# T-ALL & B-ALL - DRP data analysis!
""")

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_stage(stage):
    st.session_state.stage = stage
    
import streamlit as st
import pandas as pd
from io import StringIO
import pandas as pd
import numpy as np

    
dir = 'D:/Dibyendu/Kerstin/'

# In[3]:
# # Drug response data
drp = pd.read_csv(dir+'Rank_drugs.csv', sep=';', header=0, index_col=1)
#print(drp.shape)
#drp.head()

#drp.head()

#drp['Labeling proteomics'].value_counts()

def create_groups(df):
    df['susceptibility_logAUC'] = pd.to_numeric(df['susceptibility_logAUC'])
    conditions = [
    (df['susceptibility_logAUC'] < 0.2),
    (df['susceptibility_logAUC'] >= 0.2) & (df['susceptibility_logAUC'] <= 0.75),
    (df['susceptibility_logAUC'] > 0.75)]

    # create a list of the values we want to assign for each condition
    values = ['Sensitive', 'Intermediate', 'Resistant']

    # create a new column and use np.select to assign values to it using our lists as arguments
    df['Class'] = np.select(conditions, values)
    return df

noOfdrugsPerSample = drp['Labeling proteomics'].value_counts()


lp = noOfdrugsPerSample[noOfdrugsPerSample==26].index
#lp

#Select samples treated with 26 drugs
drugs = drp.loc[drp['Labeling proteomics'].isin(lp)]
print(drugs.shape)
#drugs.head()


drugs['Labeling proteomics'] = drugs['Labeling proteomics'].astype('str')
drugs['Labeling proteomics'] = 'S'+drugs['Labeling proteomics']
drugs = create_groups(drugs)
#drugs.head()


unique_lp = np.unique(drugs['Labeling proteomics'])
unique_drugs = np.unique(drugs['drug'])
classes = np.unique(drugs['Class'])


drugs_pivoted = drugs.pivot(index='Labeling proteomics', columns='drug', values='susceptibility_logAUC')
#print(drugs_pivoted.shape)
#drugs_pivoted.head()

#drugs_pivoted.to_csv(dir+'samples_treated_with_26_drug_panel_susceptibility_logAUC.csv')

drug_class = drugs.pivot(index='Labeling proteomics', columns='drug', values='Class')
#drug_class.head()

#drug_class.to_csv(dir+'samples_treated_with_26_drug_panel_susceptibility_logAUC_classes.csv')


# In[18]:
# # Clinical data
clin = pd.read_excel(dir+'Clinical_data_proteomics_28012024_KR.xlsx', header=0)
clin['Sample ID Proteomics'] = clin['Sample ID Proteomics'].astype('str')
clin['Sample ID Proteomics'] = 'S'+clin['Sample ID Proteomics']
clin['Immunophenoytpe']=clin['Immunophenoytpe'].astype('str')
#clin = clin.loc[clin['Immunophenoytpe']!='Sample contaminated, needs to be excluded']
#clin.drop(clin[clin['Immunophenoytpe'] == 'Sample contaminated, needs to be excluded'].index, inplace=True)
clin.drop(clin[clin['Sample ID Proteomics'] == 'S126'].index, inplace=True) #dropping the contaminated sample
#clin.head()

clin.loc[clin['Immunophenoytpe'].isin(['T-ALL', 'T-ALL ', 'T-LBL/T-ALL']), 'Immunophenoytpe'] = 'T-ALL'
clin.loc[clin['Sample ID Proteomics']=='S108', 'Diagnosis/Relapse'] = 'Diagnosis' #information collected from protein expression data

clin['Immunophenoytpe'].value_counts()

clin['Diagnosis/Relapse'].value_counts()

B_ALL_samples = clin.loc[clin['Immunophenoytpe']== 'B-ALL', ['Sample ID Proteomics', 'Diagnosis/Relapse']]
T_ALL_samples = clin.loc[clin['Immunophenoytpe'] == 'T-ALL', ['Sample ID Proteomics', 'Diagnosis/Relapse']]

B_ALL_samples_primary = B_ALL_samples.loc[B_ALL_samples['Diagnosis/Relapse']=='Diagnosis', 'Sample ID Proteomics']
B_ALL_samples_relapse = B_ALL_samples.loc[B_ALL_samples['Diagnosis/Relapse']=='Relapse', 'Sample ID Proteomics']


T_ALL_samples_primary = T_ALL_samples.loc[T_ALL_samples['Diagnosis/Relapse']=='Diagnosis', 'Sample ID Proteomics']
T_ALL_samples_relapse = T_ALL_samples.loc[T_ALL_samples['Diagnosis/Relapse']=='Relapse', 'Sample ID Proteomics']

# In[26]:
# # Protein data

protein = pd.read_csv(dir+'Proteome_Atleast1validvalue_ImputedGD.txt', header=0, sep='\t')
#protein.head()

protein = protein.iloc[5:,:]
print(protein.shape)
#protein.head()

protein_copy = protein.copy()
protein.index = protein['Protein ID']
#protein.head()

protein = protein.iloc[:,0:128]
#protein.head()

#protein = protein.dropna()
#protein = protein.dropna(how='all')
#protein.dropna(axis=0, thresh=80, inplace=True)
#protein.shape

#Select samples treated with a 26 drugs panel
protein = protein[unique_lp]

#protein.shape

#protein.head()

protein2gene_mapping =  protein_copy[['Protein ID', 'Gene']]
#protein2gene_mapping.to_csv(dir+'protein2gene_mapping.csv')

B_ALL_primary = protein[protein.columns.intersection(B_ALL_samples_primary)].T
B_ALL_relapse = protein[protein.columns.intersection(B_ALL_samples_relapse)].T
T_ALL_primary = protein[protein.columns.intersection(T_ALL_samples_primary)].T
T_ALL_relapse = protein[protein.columns.intersection(T_ALL_samples_relapse)].T

B_ALL_df = protein[protein.columns.intersection(B_ALL_samples['Sample ID Proteomics'])].T
T_ALL_df = protein[protein.columns.intersection(T_ALL_samples['Sample ID Proteomics'])].T


    
#protein = protein.T
#protein.head()

# In[37]:
# # Find feature importance w.r.t. a particular drug

#from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
#from sklearn.model_selection import KFold, cross_val_score, cross_validate
#from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.metrics import accuracy_score, classification_report
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from xgboost.sklearn import XGBClassifier
#from lightgbm.sklearn import LGBMClassifier
#from sklearn.impute import SimpleImputer
#from sklearn.svm import SVC
#import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import shap

def preSelectFeatures(X, y, threshold, exp_name):
    import os
    X['Target'] = y
    corr_mat = pd.DataFrame(X.corr()['Target'])
    pd.DataFrame(corr_mat).to_csv(os.path.join(dir,'Results/ML/New/')+exp_name+'_correlation_with_target_DRP.csv')
    proteins = corr_mat.index[abs(corr_mat['Target']) >= threshold].tolist()   #consider both positive and negative correlations >=0.3 and <=-0.3
    #print(proteins)
    return proteins[:-1]


def protein2gene(df, cols, mapping):
    df = df[cols]
    genes = protein2gene_mapping.loc[protein2gene_mapping['Protein ID'].isin(df.columns), 'Gene']
    df.columns = genes
    df.columns = df.columns.astype(str)
    return df

def evaluateClassifiers(X, y):
    from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
    from sklearn.model_selection import KFold, cross_validate, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost.sklearn import XGBClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import Lasso
    
    kfold = KFold(n_splits=10, shuffle=True, random_state=42) 
    models = [LogisticRegression(solver='liblinear', random_state=42), DecisionTreeClassifier(random_state=42), 
              RandomForestClassifier(random_state=42), SVC(kernel='linear', random_state=42), 
              XGBClassifier(random_state=42), Lasso(alpha=0.00001, random_state=42)]
    names = ["LR", "DT", "RF", "SVM", "XGB", "Lasso"]

    X = X.loc[:,~X.columns.duplicated()]
    for model, name in zip(models, names):
        print()
        print(name)
        for score in ["accuracy", "f1_weighted", "roc_auc", "r2", "neg_mean_absolute_error", "normalized_mutual_info_score", "neg_root_mean_squared_error", "explained_variance"]:
            results = cross_val_score(model, X.values, y, cv = kfold, scoring = score)
            print(score,': {:.2f}'.format(results.mean()))

def importancePlot(feat_imp, exp_name):
    import matplotlib.pyplot as plt
    import os
    
    fig, ax = plt.subplots(figsize=(10,8))
    #feat_imp.plot.bar(yerr=std, ax=ax)
    feat_imp.plot.bar()
    ax.set_title("Feature importance_"+exp_name)
    ax.set_ylabel("Score")
    ax.set_xlabel('Gene')
    filename = exp_name+'_feature_importance_based_on_DRP.png'
    #plt.savefig(os.path.join(dir, 'Results/ML/New/')+filename, dpi = 300, format = 'png', bbox_inches="tight")
    
def differentialPlot(df, conditions, exp_name):
    import scanpy as sc
    import anndata
    import os
    from scipy import stats
    import matplotlib.pyplot as plt
    
    cols = df.columns.astype(str)
    samp = df.index
    X = pd.DataFrame(np.array(df, dtype=float))
    X.columns = cols
    X.index = samp
    X = stats.zscore(X)
    ad = anndata.AnnData(X)
    ad.obs = pd.DataFrame(conditions, columns=['class'])
    ad.var_names = X.columns
    ad.var_names_make_unique()
    filename = exp_name+'_heatmap_based_on_DRP.png'
    with plt.rc_context():
        ax = sc.pl.heatmap(ad, ad.var_names, groupby='class', swap_axes=True, show_gene_labels=True, cmap="PiYG", show=False)
        ax['heatmap_ax'].set_ylabel("Gene")
        st.pyplot(plt.gcf())
        #plt.savefig(os.path.join(dir,'Results/ML/New/')+filename, dpi = 300, format = 'png', bbox_inches="tight")

def majorityVoting(lof): #Input the list of features
    from collections import Counter
    cnt=Counter()
    for x in lof:
        cnt+= Counter(x)
    features = [k for k,v in dict(cnt).items() if v>=2] #selecting a feature if 3 or more classifiers agree
    return features


def selectFeatures(df, classifiers, exp_name, n):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from xgboost.sklearn import XGBClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import Lasso
    
    X = df[0]
    y = df[1]

    X = X.loc[:,~X.columns.duplicated()]
    
    #Train-test split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    
    feat_imp=[]
    acc = []
    if 'LR' in classifiers:
        #Train a Logistic Regression (LR) model
        lr = LogisticRegression(solver='liblinear', random_state=42)
        lr.fit(X, y)
        feat_imp_lr = pd.Series(lr.coef_[0], index=X.columns, name='LR')
        feat_imp_lr = feat_imp_lr.sort_values(ascending=False)[0:n]
        feat_imp.append(feat_imp_lr.index)
        acc.append(lr.score(X.values, y))
        importancePlot(feat_imp_lr, exp_name+'_LR') #Plot feature importances
    
    if 'DT' in classifiers:
        #Train a Decision Tree (DT) model
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X, y)
        feat_imp_dt = pd.Series(dt.feature_importances_, index=X.columns, name='DT')
        feat_imp_dt = feat_imp_dt.sort_values(ascending=False)[0:n]
        feat_imp.append(feat_imp_dt.index)
        acc.append(dt.score(X.values, y))
        importancePlot(feat_imp_dt, exp_name+'_DT') #Plot feature importances
    
    if 'RF' in classifiers:
        #Train a Random Forest (RF) model  
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X, y)
        feat_imp_rf = pd.Series(rf.feature_importances_, index=X.columns, name='RF')
        feat_imp_rf = feat_imp_rf.sort_values(ascending=False)[0:n]
        feat_imp.append(feat_imp_rf.index)
        acc.append(rf.score(X.values, y))
        importancePlot(feat_imp_rf, exp_name+'_RF') #Plot feature importances
        
    if 'SVC' in classifiers:
        #Train a SVM classifier model
        svc = SVC(kernel='linear', random_state=42)
        svc.fit(X, y)
        feat_imp_svc = pd.Series(svc.coef_[0], index=X.columns, name='SVC')
        feat_imp_svc = feat_imp_svc.sort_values(ascending=False)[0:n]
        feat_imp.append(feat_imp_svc.index)
        acc.append(svc.score(X.values, y))
        importancePlot(feat_imp_svc, exp_name+'_SVC') #Plot feature importances

    if 'XGB' in classifiers:
        #Train a XGBoost classifier model
        xgb = XGBClassifier(random_state=42)
        xgb.fit(X, y)
        feat_imp_xgb = pd.Series(xgb.feature_importances_, index=X.columns, name='XGB')
        feat_imp_xgb = feat_imp_xgb.sort_values(ascending=False)[0:n]
        feat_imp.append(feat_imp_xgb.index)
        acc.append(xgb.score(X.values, y))
        importancePlot(feat_imp_xgb, exp_name+'_XGB') #Plot feature importances

    if 'Lasso' in classifiers:
        #Train a Lasso model
        lasso = Lasso(alpha=0.00001, random_state=42)
        lasso.fit(X, y)
        feat_imp_lasso = pd.Series(lasso.coef_, index=X.columns, name='Lasso')
        feat_imp_lasso = feat_imp_lasso.sort_values(ascending=False)[0:n]
        feat_imp.append(feat_imp_lasso.index)
        acc.append(lasso.score(X.values, y))
        importancePlot(feat_imp_lasso, exp_name+'_Lasso') #Plot feature importances

    #print(f"Average training accuracy: {np.mean(acc):.2f}")
    selFeatures = majorityVoting(feat_imp)
    return selFeatures

def classify(data, drug_class, exp_name, drugOfInterest, classifiers, num_features, threshold):
    import os
    label = pd.DataFrame(drug_class[drugOfInterest])
    label = label[label[drugOfInterest]!='Intermediate']
    samples = data.index.intersection(label.index) #extracting sample IDs for drug classes 'Sensitive' and 'Resistant'
    X = data.loc[samples]
    label = label.loc[samples]
    le = LabelEncoder()
    y = le.fit_transform(label)
    cols = X.columns.astype(str)
    samples = X.index
    X = pd.DataFrame(np.array(X, dtype=float))
    X.columns = cols
    X.index = samples
    
    #preselect features based on correlation with target variable from entire protein expression data
    feat = preSelectFeatures(X, y, threshold, exp_name)
    print('{} proteins were found to have significant positive or negative correlation with the annotations.'.format(len(feat)))
    #evaluateClassifiers(X, y)
    X = X[feat]
    X = protein2gene(X, X.columns, protein2gene_mapping)
    #evaluateClassifiers(X,y)
    selFeatures = selectFeatures([X,y], classifiers, exp_name, num_features)
    X = X[selFeatures]
    differentialPlot(X, label.values, exp_name)
    X['Drug_Class']=label
    #X.to_csv(os.path.join(dir+'Results/ML/New/')+exp_name+'DRP_ML_selFeatures_with_annotations.csv')
    return selFeatures

cell_type = st.selectbox('Select cell-type: ', ['T-ALL', 'B-ALL'])
drugOfInterest = st.selectbox('Select drug', options=[opt.strip() for opt in unique_drugs])
#selected_class = st.radio("Which class are you interested in?", options=[opt.strip() for opt in classes])

if cell_type == 'B-ALL':
    data = B_ALL_df
elif cell_type == 'T-ALL':
    data = T_ALL_df

#num_features = st.slider('Select number of genes',0, protein.shape[1], 20))
num_features = st.slider('Select number of genes you want to select',1, 100, 50)
threshold = st.slider('Select threshold for correlation-based feature pre-selection', 0.00, 1.00, 0.55) #threshold for correlation-based preselection
classifiers = st.multiselect('Select models - You may choose multiple among the following: [Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Support Vector Machine Classifer, XG Boost Classifier and Lasso Regression]', ['LR', 'DT', 'RF', 'SVC', 'XGB', 'Lasso'])
#st.write(classifiers)

#cell_type = 'T-ALL'
#drugOfInterest = 'Dasatinib'
#num_features = 50
#threshold = 0.55 #threshold for correlation-based preselection
#classifiers = ['LR', 'DT', 'RF', 'SVC', 'XGB', 'Lasso']
#classifiers = ['LR', 'RF', 'SVC']


#Global explanation
analyze = st.button('Analyze', on_click=set_stage, args=(1,))
if analyze:
    #shap_importance = classify('Dexamethasone', 'Resistant', 20)
    #shap_importance = classify(selected_drug, selected_class, num_features)
    #st.write(st.session_state)
    path = 'D:/Dibyendu/Kerstin/'
    #shap_importance.to_csv(path+'shap_importance.csv')
    exp_name = cell_type+'_'+drugOfInterest+'_'
    selFeatures = classify(data, drug_class, exp_name, drugOfInterest, classifiers, num_features, threshold)

# def performLocalExplanation(model, explainer, shap_values, X, sample, class_ind, num_features):
#     #Local explanations
#     # ans = st.text_input(
#         #     "Do you want a local explanation (Y/N)? 👇", 'Y')
#         # if ans in ('y', 'Y'):
#         #     sample = st.text_input(
#         #         "Enter the sample name 👇", 'S1')
#             #print(X.index)
#     st.write("dibyendu")
#     ind = list(X.index).index(sample)
#     #local explanation
#     #shap.force_plot(explainer.expected_value[class_ind], shap_values[ind,:,class_ind], features= X.iloc[ind], link="logit", show=False)
#     #st.pyplot(plt.gcf())
#     shap.plots._waterfall.waterfall_legacy(explainer.expected_value[class_ind], shap_values[ind,:,class_ind], feature_names=X.columns, max_display=num_features, show=False)
#     st.pyplot(plt.gcf())


# def increment():
#     global count
#     count=1

# if "clicked" not in st.session_state:
#     st.session_state.clicked = False


# def estimateFeatureContributionsViaXAI(model, X, y, class_ind, num_features):
#     import matplotlib.pyplot as plt
#     shap.initjs() # Need to load JS vis in the notebook
#     #Check Shapley values for the last model
#     #explaining model
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X)
#     #print(X_train.shape)
#     #print(shap_values.shape)
#     #print('Expected Values: ', explainer.expected_value)
#     #Global explnations
#     shap.summary_plot(shap_values[:,:,class_ind], X, plot_type="bar", max_display=num_features, show=False)  
#     #plt.show()
#     st.pyplot(plt.gcf())
#     sh_imp = selectFeatures(shap_values[:,:,class_ind], X.columns, num_features)
#     #ans = st.text_input('Do you want a local explanation?','N')
#     # if st.button("Want a local explanation?") or st.session_state["clicked"]:
#     #     st.session_state["clicked"] = True
#     #     #if ans in ('y','Y'):
#     # if st.session_state["clicked"] == True:
#     #     sample = st.selectbox('Select sample id', options=[opt.strip() for opt in X.index]) #, on_change=increment())
#     #     #performLocalExplanation(model, explainer, shap_values, X, sample, class_ind, num_features)
#     #if sample=='None':
#     #    st.stop()
#     #else:
#     #    
#     #if count==1:
#         #performLocalExplanation(model, explainer, shap_values, X, sample, class_ind, num_features)
#     return sh_imp


# def classify(drugOfInterest, classOfInterest, num_features):
#     label = drug_class[drugOfInterest]
#     le = LabelEncoder()
#     y = le.fit_transform(label)
#     class_mapping = dict(zip(le.classes_,range(len(le.classes_))))
#     print(class_mapping)
#     class_ind = class_mapping[classOfInterest]
#     X = protein
#     cols = X.columns.astype(str)
#     samples = X.index
#     X = pd.DataFrame(np.array(X, dtype=float))
#     X.columns = cols
#     X.index = samples
#     #preselect features based on correlation with target variable from entire protein expression data
#     feat = preSelectFeatures(X, y)
#     print('{} proteins were found to have significant positive or negative correlation with the annotations.'.format(len(feat)))
#     #evaluateClassifiers(X, y)
#     X = X[feat]
#     X = protein2gene(X, X.columns, protein2gene_mapping)
    
#     #Train-test split
#     #X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
#     model = RandomForestClassifier(
#          n_estimators=100,
#          n_jobs=-1,
#          min_samples_leaf = 5,
#          oob_score=True,
#          random_state = 42)
#     #model.fit(X_train, y_train)
#     model.fit(X, y)
#     #print(f"RF train accuracy: {model.score(X, y):.2f}")
#     #print(f"RF train accuracy: {model.score(X_train, y_train):.2f}")
#     #print(f"RF test accuracy: {model.score(X_test, y_test):.2f}")
#     #explainModel(model, X_train, y_train, X_test, y_test)
#     sh_imp = estimateFeatureContributionsViaXAI(model, X, y, class_ind, num_features)
#     return sh_imp                                           


# def differentialPlot(df1, df2, conditions, shap_importance, exp_name, path):
#     import scanpy as sc
#     import anndata
#     from scipy import stats
    
#     data = pd.concat([df1, df2])
#     cols = data.columns.astype(str)
#     samples = data.index
#     X = pd.DataFrame(np.array(data, dtype=float))
#     X.columns = cols
#     X.index = samples
#     X = protein2gene(X, data.columns, protein2gene_mapping)
#     X = stats.zscore(X, axis=1)
#     X = X[shap_importance['feature']]
    
#     target = [conditions[0]]*df1.shape[0]+[conditions[1]]*df2.shape[0]


#     ad = anndata.AnnData(X)
#     ad.obs = pd.DataFrame(target, columns=['diagnosis'])
#     ad.var_names = X.columns
#     ad.var_names_make_unique()
#     filename = 'Results/ML/'+exp_name+'_heatmap_based_on_protein_expr_only.png'
#     with plt.rc_context():
#         ax = sc.pl.heatmap(ad, ad.var_names, groupby='diagnosis', swap_axes=True, show_gene_labels=True, cmap="magma", show=False)
#         ax['heatmap_ax'].set_ylabel("Gene")
#         st.pyplot(plt.gcf())
#         plt.savefig(path+filename, dpi = 300, format = 'png', bbox_inches="tight")




    

# if st.session_state.stage > 0:
#     path = 'D:/Dibyendu/Kerstin/'
#     #shap_importance = pd.read_csv(path+'shap_importance.csv')
#     #differential plot 
#     drug = selected_drug
#     clss = selected_class
#     cell_type = "T-ALL"
#     diagnosis = "Primary_vs_Relapse"
#     path = 'D:/Dibyendu/Kerstin/'
    
#     exp = st.selectbox('Select experiment', ['T-ALL_Primary_vs_Relapse', 'B-ALL_Primary_vs_Relapse', 'Primary_T-ALL_vs_B-ALL', 'Relapse_T-ALL_vs_B-ALL'])
#     if st.button('See heatmap'):
#         if exp=='T-ALL_Primary_vs_Relapse':
#             df1=T_ALL_primary
#             df2=T_ALL_relapse
#             conditions = ['primary', 'relapse']
#         elif exp=='B-ALL_Primary_vs_Relapse':
#             df1=B_ALL_primary
#             df2=B_ALL_relapse
#             conditions = ['primary', 'relapse']
#         elif exp=='Primary_T-ALL_vs_B-ALL':
#             df1=T_ALL_primary
#             df2=B_ALL_primary
#             conditions = ['T-ALL', 'B-ALL']
#         else:
#             df1=T_ALL_relapse
#             df2=B_ALL_relapse
#             conditions = ['T-ALL', 'B-ALL']
        
#         exp_name = 'DRP'+'_'+drug+'_'+clss+'_'+exp
#         differentialPlot(df1, df2, conditions, shap_importance, exp_name, path)


