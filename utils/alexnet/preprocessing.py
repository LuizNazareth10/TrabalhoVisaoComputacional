import os
from tensorflow.keras.models import load_model

def save_model(self, path = 'models/alexNet/saved_model/alexnet.h5'):##rever qual modelo deve ser salvo (modelo provavelmente vai ser .keras)
        self.model.save(path)
        
def load_model(self, path = 'models/alexNet/saved_model/alexnet.h5'):
    if not os.path.exists(path):
        print('Não existe modelo desse arquivo salvo no caminho passado')
        # return None
    else:
        self.model = load_model(path)