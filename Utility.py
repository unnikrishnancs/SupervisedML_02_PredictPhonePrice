
import pandas as pd
from sklearn.preprocessing  import OneHotEncoder
import numpy as np


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
def predict_price(model,BrandName,camera,screensize,RAM,battery):
	try:
		# initialize one-hot vector for "Locality" (all 0s)
		inp={'Locality_BTM Layout':0, 'Locality_Bagaluru Near Yelahanka':0,
		'Locality_Banashankari':0, 'Locality_Banaswadi':0, 'Locality_Battarahalli':0,
		'Locality_Begur':0, 'Locality_Bellandur':0, 'Locality_Bommanahalli':0,
		'Locality_Brookefield':0, 'Locality_Budigere Cross':0,
		'Locality_CV Raman Nagar':0, 'Locality_Chandapura':0,
		'Locality_Dasarahalli on Tumkur Road':0,
		'Locality_Electronic City Phase 1':0, 'Locality_Electronics City':0,
		'Locality_Gottigere':0, 'Locality_HSR Layout':0, 'Locality_Harlur':0,
		'Locality_Hebbal':0, 'Locality_Hennur':0, 'Locality_Horamavu':0,
		'Locality_Hosa Road':0, 'Locality_Hoskote':0, 'Locality_Hulimavu':0,
		'Locality_Indira Nagar':0, 'Locality_J. P. Nagar':0,
		'Locality_JP Nagar Phase 7':0, 'Locality_Jakkur':0, 'Locality_Jayanagar':0,
		'Locality_Jigani':0, 'Locality_Kalyan Nagar':0,
		'Locality_Kannur on Thanisandra Main Road':0, 'Locality_Kasavanahalli':0,
		'Locality_Koramangala':0, 'Locality_Krishnarajapura':0,
		'Locality_Kumbalgodu':0, 'Locality_Mahadevapura':0, 'Locality_Marathahalli':0,
		'Locality_Marsur':0, 'Locality_Murugeshpalya':0, 'Locality_Nagarbhavi':0,
		'Locality_Narayanapura on Hennur Main Road':0, 'Locality_RR Nagar':0,
		'Locality_Rajajinagar':0, 'Locality_Ramamurthy Nagar':0,
		'Locality_Sarjapur':0, 'Locality_Sarjapur Road Post Railway Crossing':0,
		'Locality_Subramanyapura':0, 'Locality_Talaghattapura':0,
		'Locality_Thanisandra':0, 'Locality_Varthur':0, 'Locality_Vidyaranyapura':0,
		'Locality_Whitefield':0, 'Locality_Whitefield Hope Farm Junction':0,
		'Locality_Yelahanka':0}        
		
		#pass NEW data for prediction...in this case ---------- locality='Yelahanka', MinPrice=15000, MaxPrice=25000, AvgRent=20000 ----------
		inp["Locality_"+loc]=1	
		inp["MinPrice"]=min_p
		inp["MaxPrice"]=max_p
		inp["AvgRent"]=avg 
		#print(inp)
		#print()
		   	
		#create datafrane
		inp_df=pd.DataFrame(data=[inp])
		print(inp_df)
		print()
		
		#predict the data
		#print("House Type Prediction (locality=",loc,", min_price=",min_p,", max_price=",max_p,", avg_rent=", avg,") :",model.predict(inp_df))
		print("House Type Prediction (locality=",loc,", min_price=",min_p,", max_price=",max_p,", avg_rent=", avg,") :",l2n.inverse_transform(model.predict(inp_df)))
		print()

	except  Exception as ex:
		print("Error occured :", ex)
		print()
	


