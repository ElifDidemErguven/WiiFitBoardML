import pickle
import pandas as pd

#we use this piece of code after the model training to store it in a pickle file format (python object file)
#pickle.dump(trained_model_name,open("logreg_model_test.pkl","wb"))

#We use this to load our saved model with this
model = pickle.load(open("logreg_model_test.pkl","rb"))

#a random point in time from an excersize (assuming we can get it form the wii board)
keys = ['ms', 'Sensor 1 (kg)', 'Sensor 2 (kg)', 'Sensor 3 (kg)', 'Sensor 4 (kg)', 'COP distance X', 'COP distance Y', 'Total Force (kg)']
values = [1.3383787994366367,-1.0442495077938023,-0.49926221900396767,-1.2898608856121723,-1.3267618039110662,1.3801891049920814,-3.7608521094475584,-1.4441541145899572]
dictionary = dict(zip(keys, values))
#which is then saved to a dataframe
ph_single_data_point = pd.DataFrame(data=dictionary, index=[0])

#on which we run a prediction on
print(model.predict(ph_single_data_point))



