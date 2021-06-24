# scalp_EEG_classification
This is a repo of scalp EEG classification.


*data description* 
this data resample from ~300k raw EEG segments by keeping the type ratios.
  
    1.type_0 refer to background  
    2.type_1 refer to IED  
    3.type_2 refer to artifacts 

*parameters description*

    1.train params, include number of iteration, maxing learning rate 
    2.architecture elecments params, inlude layers, filter sizes, dropout...


*before running* 
    
    1.pip install pytorch
    2.pip install fastai
    3.pip install scikit,numpy
    4.pip install tasi


