# hydro-auto-od

app.py is a connector to the hydro serving. There are two main functions: status and launch. 
- Status returns the availablity of existing outlier detection methods and their status:(in queue, training, deploying etc)
- launch_ method gets id of model we want to track and id of the outlier method to train. Checks that everything is ok with statuses and start preparing model

in auto_od.py all steps are performed:
- get data
- get model (from tabular_od_methods.py)
- train model
- pack and upload model with metafiles
- delete folder

generate_monitoring_modelspec is responsible for creating outlier model spec. And there could be troubles there.
models requirements are stated in tabular_od_methods.py
models func_main.py are in a form of simple py file and is copied to tar during packing