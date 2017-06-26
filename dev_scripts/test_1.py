import numpy as np
r = np.random.RandomState(2)

from rbclassifier.genData import genData
X,y = genData(n_samples=100, n_features=10,strRel=2, n_redundant=4,
                    n_repeated=0, flip_y=0,random_state=r)

from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

from rbclassifier.rbclassifier import RelevanceBoundsClassifier
fri = RelevanceBoundsClassifier(shadow_features=False,n_resampling=40,random_state=r)


fri.fit(X_scaled,y)


fri.interval_

print("L1 {}\n C {}\n {}\n {}\n".format(fri._svm_L1,fri._hyper_C,fri.interval_,fri.allrel_prediction_))




