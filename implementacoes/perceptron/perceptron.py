from typing import List
import numpy as np
from numpy.typing import ArrayLike

print("nada")

class Perceptron:
    """
    Classificador Perceptron

    Parameters
    ----------
    learing_rate: float
        taxa de aprendizado (0 <= learning_rate <=1)
    epochs: int
        passagens pelos datasets de treino
    random_state: int
        Seed para o gerador de números aleatórios usado na
        inicialização dos pesos

    Attributes
    ---------
    w_: 1d-array
        pesos depois do treino
    b_: scalar
        bias de valor unitário
    err: list
        número de classificações erradas em cada epoch

    Methods
    ------
    __init__: None
        método de inicialização da classe
    fit: object
        treina os dados de treino
    net_input: scalar
        calcula a rede de entrada
    predict: class label
        determina a classe depois do pass unidade

    """
    def __init__(self, learing_rate: float=0.001, epochs: int=50, random_state: int=42):
        super(Perceptron, self).__init__()
        self.learing_rate = learing_rate
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> object:
        """
        Fit training data

        Parameters
        ---------
        X: [array-like], shape = [n_examples, n_features]
            Vetores de treino onde n_examples é o número de exemplos
            e n_feature é o número de features
        y: [array-like], shape = [n_examples]
            Valores alvo

        Return
        ------
        self: object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_: np.ndarray = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_: np.float_ = np.float_(0.)
        self.err_: List = []

        for _ in range(self.epochs):
            err = 0

            for xi, target in zip(X, y):
                update = self.learing_rate * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                err += np.int_(update !=0)
            self.err_.append(err)
        return self


    def net_input(self, X):
        "calcula os valores na rede de entrada"
        return np.dot(X, self.w_) + self.b_


    def predict(self, X):
        """
        Retorna o classe após a função passo
        """
        return np.where(self.net_input(X) >= 0., 1, 0)








X = np.array([1, 2, 3, 4, 5])
y = np.array([-1, 0, 1, 2, 3])

model = Perceptron()
model.fit(X, y)
