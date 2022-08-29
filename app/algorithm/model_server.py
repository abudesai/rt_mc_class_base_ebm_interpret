import numpy as np, pandas as pd
import os, sys
import json
import pprint
import algorithm.utils as utils
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.mc_classifier as mc_classifier


# get model configuration parameters 
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path, data_schema): 
        self.model_path = model_path
        self.data_schema = data_schema
        self.preprocessor = None
        self.model = None
        self.has_local_explanations = True
        self.MAX_LOCAL_EXPLANATIONS = 3
        self.id_field_name = self.data_schema["inputDatasets"]["multiClassClassificationBaseMainInput"]["idField"]  
       
    
    def _get_preprocessor(self): 
        if self.preprocessor is None: 
            try: 
                self.preprocessor = pipeline.load_preprocessor(self.model_path)
                return self.preprocessor
            except: 
                print(f'Could not load preprocessor from {self.model_path}. Did you train the model first?')
                return None
        else: return self.preprocessor
    
    def _get_model(self): 
        if self.model is None: 
            try: 
                self.model = mc_classifier.load_model(self.model_path)
                return self.model
            except: 
                print(f'Could not load model from {self.model_path}. Did you train the model first?')
                return None
        else: return self.model
        
    
    def _get_predictions(self, data, return_probs = True):  
        preprocessor = self._get_preprocessor()
        model = self._get_model()
        
        if preprocessor is None:  raise Exception("No preprocessor found. Did you train first?")
        if model is None:  raise Exception("No model found. Did you train first?")
                    
        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data)          
        # Grab input features for prediction
        pred_X, pred_ids = proc_data['X'].astype(np.float), proc_data['ids']    
        # make predictions
        if return_probs:
            preds = model.predict_proba( pred_X )
        else: 
            preds = model.predict( pred_X )
        return preds, pred_ids
    
    
    def predict_proba(self, data):          
        preds, pred_ids = self._get_predictions(data, return_probs=True)
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)        
        id_df = pd.DataFrame(pred_ids, columns=[self.id_field_name])
        
        # return the prediction df with the id and class probability fields        
        preds_df = pd.concat( [ id_df, pd.DataFrame(preds, columns = class_names)], axis=1 )
        return preds_df     
    
    
    def predict(self, data):        
        preds_df = self.predict_proba(data)
        class_names = [ str(c) for c in preds_df.columns[1:] ]          
        preds_df["prediction"] = pd.DataFrame(preds_df[class_names], columns = class_names).idxmax(axis=1)     
        preds_df.drop(class_names, axis=1, inplace=True) 
        return preds_df


    def explain_local(self, data):
        if data.shape[0] > self.MAX_LOCAL_EXPLANATIONS:
            msg = f'''Warning!
            Maximum {self.MAX_LOCAL_EXPLANATIONS} explanation(s) allowed at a time. 
            Given {data.shape[0]} samples. 
            Selecting top {self.MAX_LOCAL_EXPLANATIONS} sample(s) for explanations.'''
            print(msg)
        
        data = data.head(self.MAX_LOCAL_EXPLANATIONS)
        print(f"Now generating local explanations for {data.shape[0]} sample(s).")     
        # ------------------------------------------------------------------------------
        preprocessor = self._get_preprocessor()        
        proc_data = preprocessor.transform(data)  
        pred_X, ids = proc_data['X'].astype(np.float), proc_data['ids']  
        
        model = self._get_model()
        
        local_explanations = model.explain_local(pred_X)        
        
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)
        all_explanations = []        
        for i in range(pred_X.shape[0]):
            local_expl_data = local_explanations.data(i)            
            sample_expl_dict = {}
            sample_expl_dict[self.id_field_name] = ids[i]
            sample_expl_dict['predicted_class'] = str(class_names[int(local_expl_data["perf"]["predicted"])] )
            sample_expl_dict['predicted_class_prob'] = np.round(local_expl_data["perf"]["predicted_score"],4)
            
            sample_expl_dict["feature_impacts_per_class"] = {}
            for j, c in enumerate(class_names): 
                class_exp_dict = {}   
                class_exp_dict['Intercept'] =  local_expl_data['extra']['scores'][0][j]
                
                feature_impacts = {
                    f:np.round(v[j],4) for f,v in zip(local_expl_data["names"], local_expl_data["scores"])
                }
                class_exp_dict["feature_impacts"] = feature_impacts
                
                sample_expl_dict["feature_impacts_per_class"][str(c)] = class_exp_dict
            
            all_explanations.append(sample_expl_dict)            
            # pprint.pprint(sample_expl_dict)
        # ------------------------------------------------------  
        all_explanations = json.dumps(all_explanations, cls=utils.NpEncoder, indent=2)
        return all_explanations