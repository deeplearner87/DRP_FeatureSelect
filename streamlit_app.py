import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gdown

st.write("""
# High risk ALL - DRP data analysis!
""")

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_stage(stage):
    st.session_state.stage = stage
    

    
dir = 'https://raw.githubusercontent.com/deeplearner87/DRP_FeatureSelect/main/'


# Drug response data
def create_groups(df, drugOfInterest):
    df = df.loc[df['drug']==drugOfInterest]
    # Convert column to numeric if not already
    df = df.copy()
    df['susceptibility_logAUC'] = pd.to_numeric(df['susceptibility_logAUC'], errors='coerce')
    
    # Define conditions for classification
    conditions = [
        (df['susceptibility_logAUC'] < 0.2),
        (df['susceptibility_logAUC'] >= 0.2) & (df['susceptibility_logAUC'] <= 0.75),
        (df['susceptibility_logAUC'] > 0.75)
    ]
    
    # Define values for each condition
    values = ['Sensitive', 'Intermediate', 'Resistant']
    
    # Use np.select to create the 'Class' column safely
    df = df.copy()  # Ensure we're working with a copy to avoid the warning
    df.loc[:, 'Class'] = np.select(conditions, values, default='Unknown')
    df.index = df['Labeling proteomics']
    df.drop(columns=['Labeling proteomics', 'drug'], inplace=True)
    return df

def handlingDrugResponseData(drugOfInterest):
    drp = pd.read_csv(dir+'Rank_Drugs_cleaned_only25_drugs_10122024.csv', header=0)
    drp['Labeling proteomics'] = drp['Labeling proteomics'].astype(str)
    drp.loc[:, 'Labeling proteomics'] = 'S' + drp['Labeling proteomics']
    #Removing rows corresponding to the contaminated sample '128'
    drp = drp.loc[drp['Labeling proteomics']!='S128']
    #Drop duplicates and keep only the first entry - not necessary any more as data has been cleaned already!
    #drp = drp.drop_duplicates(subset=['Labeling proteomics', 'drug'], keep='first')
    #Filtering for 25 drugs - not necessary any more as data has been filtered already!
    #drp = drp.loc[drp['drug'].isin(drugs_of_interest)]
    ##Check how many drugs each sample is treated with
    #drp.groupby('Labeling proteomics')['drug'].nunique()
    ##Check how many samples were treated with the 25-drugs panel
    #drp.groupby('drug')['Labeling proteomics'].nunique()
    drug_class = create_groups(drp, drugOfInterest)
    return drug_class

# Clinical data
def handlingClinicalData():
    clin = pd.read_excel(dir+'Clinical_data_proteomics_28012024_KR.xlsx', header=0)
    clin['Sample ID Proteomics'] = clin['Sample ID Proteomics'].astype('str')
    clin['Sample ID Proteomics'] = 'S'+clin['Sample ID Proteomics']
    clin['Immunophenoytpe']=clin['Immunophenoytpe'].astype('str')
    clin.drop(clin[clin['Sample ID Proteomics'] == 'S126'].index, inplace=True) #dropping the contaminated sample
    #clin.head()
    clin.loc[clin['Immunophenoytpe'].isin(['T-ALL', 'T-ALL ', 'T-LBL/T-ALL']), 'Immunophenoytpe'] = 'T-ALL'
    clin.loc[clin['Sample ID Proteomics']=='S108', 'Diagnosis/Relapse'] = 'Diagnosis' #information collected from protein expression data
    ##Check number of T-ALL and B-ALL samples
    #clin['Immunophenoytpe'].value_counts()
    ##Check number of Primary and Relapse samples
    #clin['Diagnosis/Relapse'].value_counts()
    B_ALL_samples = clin.loc[clin['Immunophenoytpe']== 'B-ALL', ['Sample ID Proteomics', 'Diagnosis/Relapse']]
    T_ALL_samples = clin.loc[clin['Immunophenoytpe'] == 'T-ALL', ['Sample ID Proteomics', 'Diagnosis/Relapse']]
    
    #B_ALL_samples_primary = B_ALL_samples.loc[B_ALL_samples['Diagnosis/Relapse']=='Diagnosis', 'Sample ID Proteomics']
    #B_ALL_samples_relapse = B_ALL_samples.loc[B_ALL_samples['Diagnosis/Relapse']=='Relapse', 'Sample ID Proteomics']
    #T_ALL_samples_primary = T_ALL_samples.loc[T_ALL_samples['Diagnosis/Relapse']=='Diagnosis', 'Sample ID Proteomics']
    #T_ALL_samples_relapse = T_ALL_samples.loc[T_ALL_samples['Diagnosis/Relapse']=='Relapse', 'Sample ID Proteomics']

    #Loading the protein data
    protein = pd.read_csv(dir+'Proteome_Atleast1validvalue_ImputedGD.txt', header=0, sep='\t', low_memory=False)
    protein = protein.iloc[5:,:]
    protein_copy = protein.copy()
    protein.index = protein['Protein ID']

    protein = protein.iloc[:,0:127]
    #protein2gene_mapping =  protein_copy[['Protein ID', 'Gene']]

    #B_ALL_primary = protein[protein.columns.intersection(B_ALL_samples_primary)].T
    #B_ALL_relapse = protein[protein.columns.intersection(B_ALL_samples_relapse)].T
    #T_ALL_primary = protein[protein.columns.intersection(T_ALL_samples_primary)].T
    #T_ALL_relapse = protein[protein.columns.intersection(T_ALL_samples_relapse)].T

    B_ALL_df = protein[protein.columns.intersection(B_ALL_samples['Sample ID Proteomics'])].T
    T_ALL_df = protein[protein.columns.intersection(T_ALL_samples['Sample ID Proteomics'])].T

    TALL_protein_samples = T_ALL_df.index
    BALL_protein_samples = B_ALL_df.index
    return [T_ALL_df, B_ALL_df, TALL_protein_samples, BALL_protein_samples]

# Mapping DRP against Transcriptomics data
def mapping_DRP_Transcriptomics(drug_class, TALL_protein_samples, BALL_protein_samples):
    mapping_df = pd.read_excel(dir+'T-ALL_Samples_Consolidated_Dibyendu_301024_051124_151124.xlsx', sheet_name='Proteomics', header=1)
    mapping_df = mapping_df[['Sample ID Submitted', 'Remarks (Dibyendu)', 'Protein Sample ID']].dropna()
    #Ensure the column is treated as strings and NaN values are properly handled
    mapping_df['Protein Sample ID'] = mapping_df['Protein Sample ID'].astype(str).replace('nan', '')

    #Split by commas and handle rows properly
    mapping_df = mapping_df.assign(
        protein_sample_id=mapping_df['Protein Sample ID'].str.split(',')
        ).explode('protein_sample_id')

    #Strip any leading/trailing whitespace and remove empty values
    mapping_df['protein_sample_id'] = mapping_df['protein_sample_id'].str.strip()
    mapping_df = mapping_df.loc[mapping_df['protein_sample_id'] != '']
    mapping_df.drop(columns=['Protein Sample ID'], inplace=True)
    mapping_df['protein_sample_id'] = 'S'+mapping_df['protein_sample_id']
    mapping_df['Sample ID Submitted'] = 'OE0583_T-ALL_'+mapping_df['Sample ID Submitted'].astype(str)
    #Filter by 'Remarks (Dibyendu)'
    mapping_df = mapping_df.loc[mapping_df['Remarks (Dibyendu)'] == 'Available']

    drug_class_df = drug_class.reset_index()
    joined_df = mapping_df.merge(drug_class_df, how='inner', left_on='protein_sample_id', right_on='Labeling proteomics')
    #print(joined_df.shape)
    
    TALL_RNA_samples = joined_df.loc[joined_df['protein_sample_id'].isin(TALL_protein_samples)]['Sample ID Submitted']
    BALL_RNA_samples = joined_df.loc[joined_df['protein_sample_id'].isin(BALL_protein_samples)]['Sample ID Submitted']
    #print(len(TALL_protein_samples))
    #print(len(BALL_protein_samples))
    #print(len(TALL_RNA_samples))
    #print(len(BALL_RNA_samples))
    #print('T-ALL RNA samples')
    #print(TALL_RNA_samples)
    #print('B-ALL RNA samples')
    #print(BALL_RNA_samples)
    
    joined_df.index = joined_df['Sample ID Submitted']
    drug_class_rna = joined_df.drop(columns = ['Sample ID Submitted', 'Remarks (Dibyendu)', 'protein_sample_id', 'Labeling proteomics'])

    #Loading Transcriptomics data
    file_url = "https://drive.google.com/uc?id=1GfEkMNic_H6pGahtpB14X9iTgvcAHdf5"

    # Read the CSV file from Google Drive
    rna = pd.read_csv(file_url, index_col=0)
    B_ALL_rna_df = rna.loc[BALL_RNA_samples]
    T_ALL_rna_df = rna.loc[TALL_RNA_samples]
    return [drug_class_rna, T_ALL_rna_df, B_ALL_rna_df]

# Feature Selection
def preSelectFeatures(X, y, threshold, exp_name):
    import os
    X['Target'] = y
    corr_mat = pd.DataFrame(X.corr()['Target'])
    #pd.DataFrame(corr_mat).to_csv(os.path.join(dir,'Results/ML/New_05122024/')+exp_name+'_correlation_with_target_DRP.csv')
    features = corr_mat.index[abs(corr_mat['Target']) >= threshold].tolist()   #consider both positive and negative correlations >=0.3 and <=-0.3
    #print(features)
    return features[:-1]

def protein2gene(df, cols):
    #Loading the protein data
    protein = pd.read_csv(dir+'Proteome_Atleast1validvalue_ImputedGD.txt', header=0, sep='\t', low_memory=False)
    protein = protein.iloc[5:,:]
    protein_copy = protein.copy()
    protein.index = protein['Protein ID']
    protein = protein.iloc[:,0:127]
    protein2gene_mapping =  protein_copy[['Protein ID', 'Gene']]
    df = df[cols]
    genes = protein2gene_mapping.loc[protein2gene_mapping['Protein ID'].isin(df.columns), 'Gene']
    df.columns = genes
    df.columns = df.columns.astype(str)
    return df

# def setProxy():
#   #Set proxy
#   import os
# 
# proxy = 'http://www-int2.dkfz-heidelberg.de:3128/'
# 
# os.environ['http_proxy'] = proxy 
# os.environ['HTTP_PROXY'] = proxy
# os.environ['https_proxy'] = proxy
# os.environ['HTTPS_PROXY'] = proxy
# os.environ['ftp_proxy'] = proxy
# os.environ['FTP_PROXY'] = proxy
# 
# def ensembleID2Gene(X, cols):
#   import mygene
# import pandas as pd
# 
# setProxy()
# #Initialize MyGeneInfo client
# mg = mygene.MyGeneInfo()
# 
# def remove_version(ensg_ids):
#   #Remove the version number from ENSG IDs.
#   return [ensg.split('.')[0] for ensg in ensg_ids]
# 
# def map_ensg_to_gene_symbol(ensg_ids):
#   #Map ENSG IDs (without version numbers) to gene symbols.
#   cleaned_ensg_ids = remove_version(ensg_ids)
# #Query mygene.info to get gene symbols for the cleaned ENSG IDs
# result = mg.querymany(cleaned_ensg_ids, scopes="ensembl.gene", fields="symbol", species="human")
# #Extract the mapping as a dictionary
# mapping = {item['query']: item.get('symbol', 'NA') for item in result}
# return mapping
# 
# mapping = map_ensg_to_gene_symbol(cols)
# #mapping_df = pd.DataFrame(list(mapping.items()), columns=["ENSG_ID", "Gene_Symbol"])
# print(mapping)
# result = [value if value != "NA" else key for key, value in mapping.items()]
# X.columns = result
# return X

def ensemblID2Gene(df, cols):
    import pandas as pd
    gene_mapping = pd.read_csv('Data/gene_mapping.csv')
    #cols = remove_version(cols)
    #print(cols)
    df = df[cols]
    #print(gene_mapping.head())
    print(gene_mapping.columns)
    mapping_dict = dict(zip(gene_mapping['Ensembl_ID'], gene_mapping['Gene_Symbol']))
    # Map the column names using the mapping dictionary
    df.columns = [mapping_dict.get(col, col) for col in cols]
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
    filename = exp_name+'_feature_importance_based_on_DRP.pdf'
    #plt.savefig(os.path.join(dir, 'Results/')+filename, dpi = 300, format = 'pdf', bbox_inches="tight")

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
    filename = exp_name+'_heatmap_based_on_DRP.pdf'
    with plt.rc_context():
        ax = sc.pl.heatmap(ad, ad.var_names, groupby='class', swap_axes=True, show_gene_labels=True, cmap="PiYG_r", show=False)
        ax['heatmap_ax'].set_ylabel("Gene")
        st.pyplot(plt.gcf())
        #plt.savefig(os.path.join(dir,'Results/')+filename, dpi = 300, format = 'pdf', bbox_inches="tight")

def majorityVoting(lof): #Input the list of features
    from collections import Counter
    cnt=Counter()
    for x in lof:
        cnt+= Counter(x)
    features = [k for k,v in dict(cnt).items() if v>=2] #selecting a feature if 2 or more classifiers agree
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

def classify(data, drug_class, exp_name, classifiers, num_features, threshold, omics_type):
    import os
    from sklearn.preprocessing import LabelEncoder
    #print(data.head())
    
    
    label = pd.DataFrame(drug_class['Class'])
    label = label[label['Class']!='Intermediate']
    samples = data.index.intersection(label.index) #extracting sample IDs for drug classes 'Sensitive' and 'Resistant'
    #print(label)
    #print(samples)
    X = data.loc[samples]
    label = label.loc[samples]
    le = LabelEncoder()
    y = le.fit_transform(np.ravel(label))
    cols = X.columns.astype(str)
    samples = X.index
    X = pd.DataFrame(np.array(X, dtype=float))
    X.columns = cols
    X.index = samples
    #print(X.shape)
    #preselect features based on correlation with target variable from entire protein expression data
    feat = preSelectFeatures(X, y, threshold, exp_name)
    if feat:
        X = X[feat]
    if omics_type == 'Proteomics':
        print('{} proteins were found to have significant positive or negative correlation with the annotations.'.format(len(feat)))
        X = protein2gene(X, X.columns)
    elif omics_type == 'Transcriptomics':
        print('{} genes were found to have significant positive or negative correlation with the annotations.'.format(len(feat)))
        X = ensemblID2Gene(X, X.columns)

    #evaluateClassifiers(X,y)
    selFeatures = selectFeatures([X,y], classifiers, exp_name, num_features)
    X = X[selFeatures]
    differentialPlot(X, label.values, exp_name)
    X['Drug_Class']=label
    #X.to_csv(os.path.join(dir+'Results/ML/New_05122024/')+exp_name+'DRP_ML_selFeatures_with_annotations.csv')
    return selFeatures

#omics_type = st.selectbox('Select omics-type', ['Proteomics', 'Transcriptomics'])
omics_type = 'Proteomics'
cell_type = st.selectbox('Select cell-type', ['T-ALL', 'B-ALL'])
drugs_of_interest = ['Idarubicin', 'Dasatinib', 'Ponatinib', 'Venetoclax', 'Navitoclax', 'Doxorubicin', 'Birinapant', 'Bortezomib', 'CB-103', 'Dexamethasone', 'Cytarabine', 'Etoposide', 'Methotrexate', 'Selinexor', 'Vincristine', 'Nilotinib', 'Temsirolimus', 'Bosutinib', 'Panobinostat', 'Trametinib', 'Ruxolitinib', 'Dinaciclib', 'A1331852', 'S-63845', 'Nelarabine']
drugOfInterest = st.selectbox('Select drug', options=[opt.strip() for opt in drugs_of_interest])
#selected_class = st.radio("Which class are you interested in?", options=[opt.strip() for opt in classes])
#num_features = st.slider('Select number of genes',0, protein.shape[1], 20))
num_features = st.slider('Select number of features (genes/proteins) you want to select',1, 100, 50)
threshold = st.slider('Select threshold for correlation-based feature pre-selection', 0.00, 1.00, 0.55) #threshold for correlation-based preselection
classifiers = st.multiselect('Select models - You may choose multiple among the following: [Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Support Vector Machine Classifer, XG Boost Classifier and Lasso Regression]', ['LR', 'DT', 'RF', 'SVC', 'XGB', 'Lasso'])
st.write(classifiers)

drug_class = handlingDrugResponseData(drugOfInterest)
l1 = handlingClinicalData()
l2 = mapping_DRP_Transcriptomics(drug_class, l1[2], l1[3])

if omics_type == 'Transcriptomics':
    drug_data = l2[0]
elif omics_type == 'Proteomics':
    drug_data = drug_class

if cell_type == 'B-ALL':
    if omics_type == 'Transcriptomics':
        data = l2[2]
    elif omics_type == 'Proteomics':
        data = l1[1]
elif cell_type == 'T-ALL':
    if omics_type == 'Transcriptomics':
        data = l2[1]
    elif omics_type == 'Proteomics':
        data = l1[0]

analyze = st.button('Analyze', on_click=set_stage, args=(1,))
if analyze:
    if len(classifiers) < 2:
        st.write('Please select at least 2 classifiers')
    else:
        #shap_importance = classify('Dexamethasone', 'Resistant', 20)
        #shap_importance = classify(selected_drug, selected_class, num_features)
        #st.write(st.session_state)
        path = 'D:/Dibyendu/Kerstin/'
        #shap_importance.to_csv(path+'shap_importance.csv')
        exp_name = cell_type+'_'+drugOfInterest+'_'
        selFeatures = classify(data, drug_data, exp_name, classifiers, num_features, threshold, omics_type)
