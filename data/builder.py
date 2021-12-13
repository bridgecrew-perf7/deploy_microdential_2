from wtforms.fields.core import Label
from . import model_builder
import pandas as pd
import pathlib
import os

#class yang gunanya untuk melakukan proses masukan data ke model
class svm_model:
    def __init__(self):
        #kita definisikan encoder labelnya terlebih dahulu
        path = "data_bersih.csv"
        self.__label_encoder = {"BAIK" : 0, "SEDANG" : 1, "TIDAK SEHAT" : 2}
        self.__label_decoder = {0 : "BAIK", 1 : "SEDANG", 2 : "TIDAK SEHAT"}

        #baca data csv yang telah dibersihkan, lalu simpan di varibel local x dan y
        self.__df = pd.read_csv(path)
        self.__df = self.__df.drop(['index', 'co'], axis=1)

        self.__df = self.__encoder(self.__df)
        self.__x = self.__df[self.__df.columns[:-1]]
        self.__y = self.__df['categori']

        #encode data label
        self.__y = self.__y.astype(int)

        #deklarasi kelas model yang menampung sklearn model SVM
        self.__model = model_builder.model(self.__x)

        #buat model
        self.__model.fit(self.__x.to_numpy(), self.__y.to_numpy(), test_size=0.38)

    #fungsi yang gunanya untuk mengubah bentuk label dari yang tadinya object menjadi int
    def __encoder(self, data):
        data = data.replace({'categori' : self.__label_encoder})
        
        return data

    #fungsi untuk memberikan prediksi data
    def predict(self, x):
        return self.__label_decoder[self.__model.predict(x)[0]]

    #fungsi untuk mendapat akurasi dari model yang dilatih
    def report_accuracy(self):
        return self.__model.report_accuracy()