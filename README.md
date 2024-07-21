# Objective:
The objective of this assessment is to evaluate the understanding and ability to apply supervised learning techniques to a real-world dataset.

# Dataset:
Breast cancer dataset available in the sklearn library.

# Data Preprocessing :
Firstly, the data is loaded from sklearn site.

      from sklearn.datasets import load_breast_cancer

Features and targets are assigned to variables x and y respectively. 

* For data preprocessing, missing values are calculated.
  
      missing_values=X.isnull().sum()
      print(missing_values)

* Duplicate values are removed.
  
      X.drop_duplicates(inplace=True)

* Encoding By Feature Scaling :

        from sklearn.preprocessing import StandardScaler
        scaler=StandardScaler()

Data preprocessing plays a vital role in machine learning. 
It helps in missing values.
It helps in detecting and removing outliers.
It helps in normalising and standardising features.

# Algorithms used :
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)
5. k-Nearest Neighbors (k-NN)
6. Naive Bayes

# 1. Logistic Regression
Logistic regression is suitable when the target variable is binary or categorical.
Here, the target variables are ['malignant', 'benign']
So, it is suitable for this model.

# Code
Importing libraries

      from sklearn.linear_model import LogisticRegression
      model=LogisticRegression()
      from sklearn.model_selection import train_test_split

Split the dataset into training and testing sets and train the regresser.

      X_train, X_test, y_train, y_test=train_test_split(cancer.data, cancer.target, test_size=0.2)
      model.fit(X_train, y_train)
      
Accuracy of the model is calculated :

      model.score(X_test, y_test)

Accuracy of Logistic regression model is 94.7%

# 2. Decision Tree Classifier

A Decision Tree classifier is used with binary or multiclass target variables. Decision Trees can handle both categorical and numerical features
This model is easy to understand and visualize.

# Code
Importing libraries

      from sklearn.tree import DecisionTreeClassifier
      tree_model1=DecisionTreeClassifier()

Train the classifier

      tree_model1.fit(X_train, y_train)

The tree is plotted using the following code :

      from sklearn import tree
      plt.figure(figsize=(25,20))
      tree.plot_tree(tree_model1, filled=True)

  When the classification report is printed, accuracy is 93% and precision is 91%

  # Post pruning
  In decision tree algorithms, post-pruning, also known as cost-complexity pruning or just pruning, is typically done using parameters like the maximum depth of the tree, minimum samples per leaf, or minimum samples per split. These parameters control the growth of the tree during training and can help prevent overfitting.

  # Pre pruning
  Pre-pruning, also known as early stopping, works by setting constraints on the tree-building process during training. Unlike post-pruning, which involves growing a full tree and then removing nodes, pre-pruning prevents the tree from becoming overly complex in the first place.
  
# 3. Random Forest Classifier

This model gives high accuracy and reduces overfitting. 
It can handle dataset with many features.

# Code
Importing libraries :
        from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier
        rclf=RandomForestClassifier()
        
Training Classifier :
        rclf.fit(X_train,y_train)

Prediction :
rclf_pred=rclf.predict(X_test)

Accuracy :
print(accuracy_score(y_test,rclf_pred))

# Various Ensemble Techniques
* Bagging Classifier
* AdaBoost Classifier
* Gadient Boosting Classifier
* XG Boost

Comparing Random forest model, adaBoost, Gradient Boost, XG boost and Bagging classifier :

Highest Accuracy : XG Boost (97%), Random Forest (96%), Gradient Boost (96%)

Highest Precision : XG Boost (97%), Random Forest (97%)

Highest recall : Ada Boost, XG Boost

Highest f1 score : XG Boost
# Comparing the above models, XG boost shows best result with 97% accuracy and 97% precision.

# 4. Support Vector Machine (SVM)
SVMs are effective in high-dimensional spaces.
It helps reducing overfitting.
By choosing appropriate kernels, SVMs can model complex decision boundaries, making them suitable for datasets where the decision boundary between classes is not linear.

# Code
Importing libraries

             from sklearn.svm import SVC
            svc_model=SVC(kernel='rbf',C=3)
            svc_model.fit(X_train, y_train)

Accuracy, precision, recall and f1 score is calculated.
Accuracy is 94.7% and f1 score is 96%

After cross validation and hyperparameter tuning, best f1 score is 91%

# 5. k-Nearest Neighbors (k-NN)
It classifies a data point based on the majority class among its k-nearest neighbors.
It makes no assumptions about the underlying data distribution.

# Code
Importing libraries

            from sklearn.neighbors import KNeighborsClassifier

Accuracy is 95.6%

After Parameter tuning, accuracy is 96% and number of neighbours = 9

# 6. Naive Bayes
They require a small amount of training data to estimate the parameters.
Handles irrelevant features.
Naive Bayes can handle missing data by simply ignoring the missing values during probability estimation.

# Types of Naive Bayes
* Gaussian Naive Bayes : Suitable for continuous data.
* Multinomial Naive Bayes : Suitable for discrete data.
* Bernoulli Naive Bayes : Suitable for boolean or binary features.

# Code
Importing libraries

              from sklearn.naive_bayes import GaussianNB
              classifier = GaussianNB()

Accuracy is 97.3%

# Comparing Performance of the models
Accuracies : Logistic Regression Model : 94.7 % , Decision Tree Model : 95.7%

Comparing Random forest model, adaBoost, Gradient Boost, XG boost and Bagging classifier :

Highest Accuracy : XG Boost (97%), Random Forest (96%), Gradient Boost (96%)

Highest Precision : XG Boost (97%), Random Forest (97%)

Highest recall : Ada Boost, XG Boost

Highest f1 score : XG Boost

SVM : 95%

KNN : 95%

Naive Bayes : 97%

Comparing all the models, XG Boost and Naive Bayes performs best and Logistic Regression Model performs worse.

