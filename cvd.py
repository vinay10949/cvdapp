#id;age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active;cardio

#['gender_1','alco','active','blood_pressure_level','ap_lo','weight','bmi','age_bucket','cholesterol','ap_hi']
import streamlit as st,shap
import pandas as pd
import matplotlib, matplotlib.pyplot as pl
import feature_engine
import numpy as np
from PIL import Image
import pickle
#loaded_model = pickle.load(open("xgboost_cv_best_pickle.dat", "rb"))
#st.write('Note that this model is previously fitted and loaded here, due to performance reasons')
import catboost
from sklearn.externals import joblib

joblib_file = "joblib_catboost_Model.pkl"  
catboost_calibrated_model = joblib.load(joblib_file)

ohe_file = "ohe.pkl"  
ohe_enc = joblib.load(ohe_file)

jamesStienenc_file = "jamesStienenc.pkl"  
jamesStienenc_enc = joblib.load(jamesStienenc_file)


ageDiscretizer_file = "ageDiscretizer.pkl"  
ageDiscretizer = joblib.load(ageDiscretizer_file)
test=pd.read_csv("test.csv", index_col=None)
test.drop('Unnamed: 0',axis=1,inplace=True)
explainer = shap.TreeExplainer(catboost_calibrated_model.base_estimator)
testShapValues=explainer.shap_values(test)		



def calculateBloodPressureLevel(data):
    if (data['ap_hi'] < 120) and (data['ap_lo'] < 80):
        return 'Normal'
    if (data['ap_hi'] >= 120 and data['ap_hi'] <=129) and (data['ap_lo'] < 80):
        return 'Elevated'
    if (data['ap_hi'] >= 130 and data['ap_hi'] <=139) | (data['ap_lo'] >= 80 and data['ap_lo'] <=89):
        return 'Stage1HyperTension'
    if (data['ap_hi'] >= 140) | (data['ap_lo'] >= 90):
        return 'Stage2HyperTension'
    if (data['ap_hi'] >= 180) | (data['ap_lo'] >= 120):
        return 'HypertensiveCrisis'

def BMI(data):
    return data['weight'] / (data['height']/100)**2
 
def main():
	img=Image.open("images/heart.png")
	me=Image.open("images/vinay.jpg")
	st.sidebar.image(me,width=200)	
	st.sidebar.subheader("Name : Vinay Sawant")
	st.sidebar.subheader("Email : vinay10949@gmail.com")
	st.image(img,width=200,caption='Save Lives')
	st.subheader('Created by: Vinay Sawant')

	st.title("Cardio Vascular Detection")

	st.header("Whats your gender ? ")
	gender=st.radio("Gender",[1,0])


	st.header("Whats your height ? ")
	height=st.slider("Height(cms)",152,192)


	st.header("Whats your Weight ? ")
	weight=st.slider("Weight in Kgs",55,150)


	st.header("Whats your Age ? ")
	age=st.slider("Age in days",10585,36500)


	st.header("What is your Systolic blood pressure? ")
	apHi=st.slider("Sbp",100,170)

	st.header("What is your Diastolic blood pressure ")
	apLo=st.slider("Dbp",60,100)

	st.header("Whats your cholesterol level ?")
	cholesterol=st.radio("Cholesterol Level",["Normal","AboveNormal","WellAboveNormal"])
	
	st.header("Whats your glucose level ?")
	gluc=st.radio("Glucose level",["Normal","AboveNormal","WellAboveNormal"])

	st.header("Do you smoke ?")
	smoke=st.radio("    ",["Yes","No"])

	st.header("Do you consume alcohol ?")
	alco=st.radio("  ",["Yes","No"])

	st.header("Are you physically active,Do you work out ? ")
	active=st.radio(" ",["Yes","No"])
	st.title('Explaining the  model')
	st.write('below, all seperate decision trees that have been build by training the model can be reviewed')		
	st.write('To handle this inconsitency, SHAP values give robust details, among which is feature importance')  
	st.write(catboost_calibrated_model.base_estimator.plot_tree(tree_idx=0))
	b=st.button("Submit", key=None)
	if b:  
		age=age/365
		#id;age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active
		d={'age':age,'gender':gender,'weight':weight,'height':height,'ap_hi':apHi,'ap_lo':apLo,'cholesterol':cholesterol,'gluc':gluc,
'smoke':smoke,'alco':alco,'active':active}
		data=pd.DataFrame(d,index=[0])
		data=ageDiscretizer.transform(data)
		data['age_bucket']=data['age'].round(2)
		data['bmi'] = data.apply(BMI, axis=1)
		# bucket boundaries
		buckets = [0, 18.5, 24.9, 29.9, 1000]
		# bucket labels
		labels = ['Underweight','Healthy','Overweight','Obese']
		# discretisation
		data['bmi_category'] = pd.cut(data['bmi'], bins=buckets, labels=labels, include_lowest=True)
		data['blood_pressure_level'] = data.apply(calculateBloodPressureLevel, axis=1)
		data=ohe_enc.transform(data)
		data=jamesStienenc_enc.transform(data)
		data['smoke'] = data['smoke'].map({"Yes":1,"No":0})
		data['alco'] = data['alco'].map({"Yes":1,"No":0})
		data['active'] = data['active'].map({"Yes":1,"No":0})
		data['cholesterol'] = data['cholesterol'].map({"Normal":1,"AboveNormal":2,"WellAboveNormal":3})	
		pred=catboost_calibrated_model.predict(data[['gender_1','alco','active','blood_pressure_level','ap_lo','weight','bmi','age_bucket'
,'cholesterol','ap_hi']])
		newData=data[['gender_1','alco','active','blood_pressure_level','ap_lo','weight','bmi','age_bucket'
,'cholesterol','ap_hi']]		
		if ["Yes" if pred[0]==1 else "No"][0]=="Yes":
			st.warning("You have high probablitily of Cardio Disease,go visit Doctor immediately")
		else:
			st.success("You are safe,you dont have Cardio disease")	
		st.title('Explaining the  model')
		st.write('below, all seperate decision trees that have been build by training the model can be reviewed')		
		st.write('To handle this inconsitency, SHAP values give robust details, among which is feature importance')  
		st.write(catboost_calibrated_model.base_estimator.plot_tree(tree_idx=0))
		shap_values = explainer.shap_values(newData)		
		pl.title('Assessing feature importance based on Shap values')
		shap.summary_plot(shap_values,newData,plot_type="bar",show=False)
		st.pyplot(bbox_inches='tight')
		pl.clf()
		st.write('SHAP values can also be used to represent the distribution of the training set of the respectable SHAP value in relation with the Target value, in this case the Cardio Disease')
		pl.title('Total distribution of observations based on Shap values, colored by Target value')
		shap.summary_plot(shap_values,newData,show=False)
		st.pyplot(bbox_inches='tight')
		pl.clf()
		st.write('Which features caused this specific prediction? features in red increased the prediction, in blue decreased them')
		pred_prob=catboost_calibrated_model.predict_proba(newData)
		expectedValue=explainer.expected_value.round(4)
		sumShap=sum(shap_values[0]).round(3)
		st.write('The real probablitiy value for this individual record is: '+str(pred_prob))
		st.write('The predicted label is : '+["Yes" if pred[0]==1 else "No"][0])
		st.write('This prediction is calculated as follows: '+'The average cardio disease probablity is : ('+str(expectedValue)+')'+' + the sum of the SHAP values. ')
		st.write( 'For this individual record the sum of the SHAP values is: '+str(sumShap))
		st.write( 'This yields to a predicted value of cardio:'+str(expectedValue)+' + '+str(sumShap)+'= '+str(expectedValue+sumShap))
		st.write('Which features caused this specific prediction? features in red increased the prediction, in blue decreased them')
		shap.force_plot(explainer.expected_value, shap_values[0],newData,matplotlib=True,show=False,figsize=(16,5))
		st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
		pl.clf()
		st.write('In the plot above, the feature values are shown. The SHAP values are represented by the length of the specific bar.'
'However, it is not quite clear what each single SHAP value is exactly, this can be seen below, if wanted.')
		st.title('Developing a deeper understanding of the data using SHAP: Interaction effects')
		shap.dependence_plot('ap_hi', testShapValues, test, interaction_index="age_bucket")
		st.pyplot()
		pl.clf()
		shap.dependence_plot('bmi', testShapValues, test, interaction_index="ap_hi")
		st.pyplot()
		pl.clf()
		st.write('Conclusion: There is interaction between bmi and ap_hi')
    
if __name__== '__main__':
    main()

	
