import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import preproceso
from sklearn.metrics import accuracy_score, f1_score, precision_score
import scikitplot.metrics as skplt
import matplotlib.pyplot as plt


numtopics = [20, 22, 24, 26, 28, 30]
best_fscore = -1
fscores_todos = []

for numtopic in numtopics:

    fscores = []
    f="datasets/train.csv"
    df = pd.read_csv(f)
    df, diccionario = preproceso.topicosTrain(df, numtopic)

    for semilla in range(10):

        # items
        x = []
        for i in range(len(df)):
            b = df.iloc[i]
            x.append(b["Topicos"])


        # etiquetas
        Y = df["Chapter"]
        y = Y.to_numpy()

        # split train(0.8) test(0.2) con semilla
        print("semilla: " +str(semilla))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=semilla)


        # PERCEPTRON OPTIMIZADO
        ppn = Perceptron(penalty=None, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, eta0=1.0, n_jobs=None, random_state=0, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False)

        # MLP OPTIMIZADO
        #ppn = MLPClassifier(hidden_layer_sizes=(300,200,100), max_iter=300,activation = 'relu',solver='adam',alpha=0.0001,random_state=1)

        # Train the perceptron
        ppn.fit(x_train, y_train)
        # Apply the trained perceptron on the X data to make predicts for the y test data
        y_pred = ppn.predict(x_test)


        print("La accuracy es:", accuracy_score(y_pred, y_test))
        print("La precision es:", precision_score(y_pred, y_test, average='weighted'))
        fscore = f1_score(y_pred, y_test, average='weighted')
        print("El f1 score es:", str(fscore))
        fscores.append(fscore)

        error = {"aciertos": 0, "errores": 0}
        for y, label in zip(list(y_test), y_pred):
            if y - label == 0:
                error["aciertos"] = error["aciertos"] + 1
            else:
                error["errores"] = error["errores"] + 1
        print(error)
        errorTotal = error["errores"] / (error["errores"] + error["aciertos"])
        print("El error es de: " + str(errorTotal))

        if fscore > best_fscore:
            best_fscore = fscore
            best_ypred = y_pred
            ntop = numtopic
            sem = semilla

    fscores_todos.append(fscores)

#crear boxplot conjunto
print(fscores_todos)
fig = plt.figure()
fig.suptitle('Bonanza del modelo', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
ax.boxplot(fscores_todos)
plt.xticks([1, 2, 3, 4, 5, 6], ['20', '22', '24', '26', '28', '30'])
ax.set_title('según el número de tópicos')
ax.set_xlabel('NumTopics')
ax.set_ylabel('Weighted Fscore')
plt.show()


#printear la matriz del mejor
skplt.plot_confusion_matrix(best_ypred, y_test)
plt.xlabel("True label")
plt.ylabel("Predicted label")
plt.savefig('Imagenes/matrizDeConfusion.png')
plt.show()



print("La mejor representacion ha sido:")
print("Fscore: " +str(best_fscore))
print("NumTopics: " +str(ntop))
print("Semilla: " +str(sem))