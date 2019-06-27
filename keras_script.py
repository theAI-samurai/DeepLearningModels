import numpy as np

from keras.models import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
path = 'D:/myProject/DeepLearningModels/'
os.chdir(path)



df_data = pd.read_csv(path+'map_traininglabels.csv')
X_train, X_test, y_train, y_test = train_test_split(df_data['image_id'], df_data['Class'], test_size=0.25)


# Parameters
params = {'dim': (32,32,32),
          'batch_size': 64,
          'n_classes': 6,
          'n_channels': 1,
          'shuffle': True}

# Datasets
partition = {'train':X_train, 'validation':X_test}
df_=df_data
df_=df_.drop(['score'],1)
labels = df_.set_index('image_id').T.to_dict('list')

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
model = Sequential()
[...] # Architecture
model.compile()

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)