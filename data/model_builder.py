from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

#class yang kita gunakan untuk membentuk model
class model:
    def __init__(self, x):
        #definisikan model dan scaler
        self.__model = SVC()
        self.__scaler = MinMaxScaler()
        self.__scaler.fit(x)

    def fit(self, x, y, test_size=0.3):
        #split data menjadi data training dan data test
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(x, y, test_size=test_size)

        #ubah skala data 
        self.__x_train = self.__scaler_data(self.__x_train)
        self.__x_test = self.__scaler_data(self.__x_test)

        #latih data training ke model
        self.__model.fit(self.__x_train, self.__y_train)

    #fungsi untuk mengskalakan data
    def __scaler_data(self, data):
        return self.__scaler.transform(data)

    #fungsi untuk memberikan prediksi berdasarkan nilai
    def predict(self, x):
        x = self.__scaler.transform(x)
        return self.__model.predict(x)

    def report_accuracy(self):
        return accuracy_score(self.__y_test, self.__model.predict(self.__x_test))
