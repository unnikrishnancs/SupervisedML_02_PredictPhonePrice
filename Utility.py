
import pandas as pd
from sklearn.preprocessing  import OneHotEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler



if __name__=="__main__":
	print("This program (UtilityFunctions.py) was called directly from python command prompt: python <file.py>")
	print()
else:
	print("This program (UtilityFunctions.py) was called by another program")
	print()

		
def data_preprocessing_regr(data,cat_cols,oth_numeric_cols):	
	
	#Handle NaN value
	data=data.dropna()
	
	cat_df=pd.DataFrame()
	
	#loop through all categorical col and one-hot-encode
	for col in cat_cols:
		ohe=OneHotEncoder()
		ohe.fit(data[[col]])
		#print("-----------Identified Categories: \n",ohe.categories_,"\n Feature Names : \n",ohe.get_feature_names_out())
		data_new=ohe.transform(data[[col]]).toarray()
		data_newdf=pd.DataFrame(data=data_new,columns=ohe.get_feature_names_out())	
		#print("col ",col," transformed as \n",data_newdf.head())
		#print()
		
		#join caterories
		cat_df=pd.concat([cat_df,data_newdf],axis=1)
		
	print("joined categries...\n",cat_df.head())
	
	#remaining columns
	df_oth=data[oth_numeric_cols]
	#to make it concat friendly		
	df_oth=df_oth.reset_index(drop=True)
	
	#join input features
	inp_feat=pd.concat([cat_df,df_oth],axis=1)	
		
	return inp_feat
	


#function to predict new data
def predict_price(model,brand,cam,screensize,RAM,battery):
	try:
		# initialize one-hot vector for "Locality" (all 0s)
		inp={
		'Brand Name_Apple':0,'Brand Name_Google':0,'Brand Name_Micromax':0,'Brand Name_Motorola':0,'Brand Name_Nokia':0,
		'Brand Name_OnePlus':0,'Brand Name_Oppo':0,'Brand Name_Realme':0,'Brand Name_Samsung':0,'Brand Name_Vivo':0,'Brand Name_Xiaomi':0,
		'camera (in no./mp)_0.3MP':0,'camera (in no./mp)_12.2MP':0,'camera (in no./mp)_12MP':0,'camera (in no./mp)_13MP':0,
		'camera (in no./mp)_5MP':0,'camera (in no./mp)_8MP':0,'camera (in no./mp)_Dual':0,'camera (in no./mp)_Quad':0,'camera (in no./mp)_Triple':0
		}        
		
		#pass NEW data for prediction...in this case ---------- locality='Yelahanka', MinPrice=15000, MaxPrice=25000, AvgRent=20000 ----------
		inp["Brand Name_"+brand]=1
		inp["camera (in no./mp)_"+cam]=1
		inp["screensize (in inches)"]=screensize
		inp["RAM (in GB)"]=RAM
		inp["battery (in mah)"]=battery 
		print(inp)
		#print()
		   	
		#create datafrane
		inp_df=pd.DataFrame(data=[inp])
		print(inp_df)
		print()
		
		#predict the data
		print("Predicted Phone Price (BrandName=",brand,", camera=",cam,", screensize=",screensize,", RAM=", RAM," battery=",battery,") : INR ",model.predict(inp_df))
		print()

	except  Exception as ex:
		print("Error occured :", ex)
		print()
	

#scale the feature vector X (inputs)
def scale_features(data):
	ss=StandardScaler()
	data=pd.DataFrame(data=ss.fit_transform(data),columns=ss.feature_names_in_)
	return data
	

