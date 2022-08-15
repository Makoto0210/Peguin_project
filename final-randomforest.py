import numpy as np
import pandas as pd
import sklearn
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
#---------------------------------------------------------------------------------
simplify = pd.read_csv('penguins_size.csv') #7 344
#-----------------------------------------------------------------------------------------------
#train:test = 2 : 1---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
test_simplify = simplify.iloc[2::3].copy() # 2 5 8............
##print(test_simplify.index.tolist())
train_index = list(range(0,len(simplify),3)) + list(range(1,len(simplify),3))
train_index = sorted(train_index)
train_simplify = simplify.iloc[train_index].copy() # 0 1 3 4...............
#--------------------------------------------------------------------------------
factor = train_simplify.columns.tolist()
##print(factor)
train_simplify = train_simplify[train_simplify["sex"].isin(["MALE","FEMALE"])] #把sex不是 "MALE","FEMALE"的清掉，同時也清掉NA
train_simplify_x = train_simplify[factor[1:]]
train_simplify_y = train_simplify[factor[0]]
Std = StandardScaler()
train_simplify_x[factor[2:6]] = Std.fit_transform(train_simplify_x[factor[2:6]]) #標準化
train_simplify_x_dummies = pd.get_dummies(train_simplify_x)
##print(train_simplify_x_dummies)
#-------------------------------------------------------------------------------
test_simplify = test_simplify[test_simplify["sex"].isin(["MALE","FEMALE"])] #把sex不是 "MALE","FEMALE"的清掉，同時也清掉NA
test_simplify_x = test_simplify[factor[1:]]
test_simplify_y = test_simplify[factor[0]]
##print(test_simplify_y)
##print(test_simplify_x)
test_simplify_x[factor[2:6]] = Std.fit_transform(test_simplify_x[factor[2:6]]) #標準化
test_simplify_x_dummies = pd.get_dummies(test_simplify_x)
##print(test_simplify_x_dummies)
#----------------------------------------------------------------------------------
dummy = test_simplify_x_dummies.columns.tolist()
print(dummy)
#-------------------------------------------------------------------------------------------------
# 建立 random forest 模型
max_sample = [0.7]
max_feature = [round(sqrt(len(dummy)))]
print(max_feature)
#fit--------------------------------------------------------------------------------------------------------------------
count = 0
i = 0
while i < len(max_sample):
    j = 0
    while j < len(max_feature):
        forest = ensemble.RandomForestClassifier(criterion="entropy",random_state=100,max_samples=max_sample[i],max_features=max_feature[j])
        model0 = forest.fit(train_simplify_x_dummies, train_simplify_y)
        # 預測
        test_simplify_y_predicted = model0.predict(test_simplify_x_dummies)
        predict = test_simplify_y_predicted.tolist()
        # -------------------------------------------------------
        test_simplify['predict'] = predict
        answer = {"species": test_simplify[factor[0]], "predict": predict}
        answer_df = pd.DataFrame(answer)
        print(answer_df)
        # -------------------------------------------------------------------------
        result0 = classification_report(test_simplify[factor[0]], test_simplify['predict'], output_dict=True)
        result0 = pd.DataFrame(result0)
        result0 = result0.round(4)
        print(result0)
        result0['accuracy'] = ["", "", result0.at['support', 'accuracy'], result0.at['support', 'macro avg']]
        result0 = result0.transpose()
        result0s = np.split(result0, [3])
        result0 = pd.concat([result0s[0], pd.DataFrame([[np.NaN] * 4], columns=result0.columns), result0s[1]])
        result0 = result0.rename(index={0: ""})
        #0000000000000000000000000000000000000000000000000000000000000000000--------------------------------------------
        result0.to_csv("randomforest-result0" + str(count) + "-penguin.csv")
        #隨機森林視覺化圖-----------------------------------------------------------------------------------------------
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 2), dpi=900)
        for index in range(0, 5):
            plot_tree(forest.estimators_[index],feature_names=train_simplify_x_dummies.columns,filled=True,ax=axes[index])
            axes[index].set_title('Estimator: ' + str(index))
        fig.savefig("split0visual-rf" + str(count) + ".png")
        plt.cla()
        #混淆矩陣圖----------------------------------------------------------------------------------------------------
        plot_confusion_matrix(forest, test_simplify_x_dummies, test_simplify_y, cmap="Purples")
        plt.savefig("split0confmatr-rf" + str(count) + ".png")
        plt.cla()
        count = count + 1
        j = j + 1
    i = i + 1
#---------------------------------------------------------------------------------------------------------------------
#train:test = 3: 1---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
test_simplify = simplify.iloc[3::4].copy() # 3 7 11............
#print(test_simplify.index.tolist())
train_index = list(range(0,len(simplify),4)) + list(range(1,len(simplify),4)) + list(range(2,len(simplify),4))
train_index = sorted(train_index)
train_simplify = simplify.iloc[train_index].copy() # 0 1 2 4 5 6 8 9 10...............
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
factor = train_simplify.columns.tolist()
##print(factor)
train_simplify = train_simplify[train_simplify["sex"].isin(["MALE","FEMALE"])] #把sex不是 "MALE","FEMALE"的清掉，同時也清掉NA
train_simplify_x = train_simplify[factor[1:]]
train_simplify_y = train_simplify[factor[0]]
Std = StandardScaler()
train_simplify_x[factor[2:6]] = Std.fit_transform(train_simplify_x[factor[2:6]]) #標準化
train_simplify_x_dummies = pd.get_dummies(train_simplify_x)
##print(train_simplify_x_dummies)
#-------------------------------------------------------------------------------
test_simplify = test_simplify[test_simplify["sex"].isin(["MALE","FEMALE"])] #把sex不是 "MALE","FEMALE"的清掉，同時也清掉NA
test_simplify_x = test_simplify[factor[1:]]
test_simplify_y = test_simplify[factor[0]]
##print(test_simplify_y)
##print(test_simplify_x)
test_simplify_x[factor[2:6]] = Std.fit_transform(test_simplify_x[factor[2:6]]) #標準化
test_simplify_x_dummies = pd.get_dummies(test_simplify_x)
##print(test_simplify_x_dummies)
#----------------------------------------------------------------------------------
dummy = test_simplify_x_dummies.columns.tolist()
print(dummy)
#-------------------------------------------------------------------------------------------------
# 建立 random forest 模型
max_sample = [0.7]
max_feature = [round(sqrt(len(dummy)))]
print(max_feature)
#fit--------------------------------------------------------------------------------------------------------------------
count = 0
i = 0
while i < len(max_sample):
    j = 0
    while j < len(max_feature):
        forest = ensemble.RandomForestClassifier(criterion="entropy",random_state=100,max_samples=max_sample[i],max_features=max_feature[j])
        model0 = forest.fit(train_simplify_x_dummies, train_simplify_y)
        # 預測
        test_simplify_y_predicted = model0.predict(test_simplify_x_dummies)
        predict = test_simplify_y_predicted.tolist()
        # -------------------------------------------------------
        test_simplify['predict'] = predict
        answer = {"species": test_simplify[factor[0]], "predict": predict}
        answer_df = pd.DataFrame(answer)
        print(answer_df)
        #-------------------------------------------------------------------------
        result0 = classification_report(test_simplify[factor[0]], test_simplify['predict'], output_dict=True)
        result0 = pd.DataFrame(result0)
        result0 = result0.round(4)
        print(result0)
        result0['accuracy'] = ["", "", result0.at['support', 'accuracy'], result0.at['support', 'macro avg']]
        result0 = result0.transpose()
        result0s = np.split(result0, [3])
        result0 = pd.concat([result0s[0], pd.DataFrame([[np.NaN] * 4], columns=result0.columns), result0s[1]])
        result0 = result0.rename(index={0: ""})
        #11111111111111111111111111111111111111111111111---------------------------------------------------------------
        result0.to_csv("randomforest-result1" + str(count) +"-penguin.csv")
        #隨機森林視覺化圖-----------------------------------------------------------------------------------------------
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 2), dpi=900)
        for index in range(0, 5):
            plot_tree(forest.estimators_[index],feature_names=train_simplify_x_dummies.columns,filled=True,ax=axes[index])
            axes[index].set_title('Estimator: ' + str(index))
        fig.savefig("split1visual-rf" + str(count) + ".png")
        plt.cla()
        #混淆矩陣圖----------------------------------------------------------------------------------------------------
        plot_confusion_matrix(forest, test_simplify_x_dummies, test_simplify_y, cmap="Purples")
        plt.savefig("split1confmatr-rf" + str(count) + ".png")
        plt.cla()
        count = count + 1
        j = j + 1
    i = i + 1
    #---------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#train:test = 1 : 1---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
test_simplify = simplify.iloc[1::2].copy() #1 3 5 7............
#print(test_simplify.index.tolist())
train_index = list(range(0,len(simplify),2))
train_simplify = simplify.iloc[train_index].copy() # 2 4 6 8 10...............
#--------------------------------------------------------------------------------
factor = train_simplify.columns.tolist()
##print(factor)
train_simplify = train_simplify[train_simplify["sex"].isin(["MALE","FEMALE"])] #把sex不是 "MALE","FEMALE"的清掉，同時也清掉NA
train_simplify_x = train_simplify[factor[1:]]
train_simplify_y = train_simplify[factor[0]]
Std = StandardScaler()
train_simplify_x[factor[2:6]] = Std.fit_transform(train_simplify_x[factor[2:6]]) #標準化
train_simplify_x_dummies = pd.get_dummies(train_simplify_x)
#print(train_simplify_x_dummies)
#-------------------------------------------------------------------------------
test_simplify = test_simplify[test_simplify["sex"].isin(["MALE","FEMALE"])] #把sex不是 "MALE","FEMALE"的清掉，同時也清掉NA
test_simplify_x = test_simplify[factor[1:]]
test_simplify_y = test_simplify[factor[0]]
#print(test_simplify_y)
#print(test_simplify_x)
test_simplify_x[factor[2:6]] = Std.fit_transform(test_simplify_x[factor[2:6]]) #標準化
test_simplify_x_dummies = pd.get_dummies(test_simplify_x)
#print(test_simplify_x_dummies)
#--------------------------------------------------------------------------------
dummy = test_simplify_x_dummies.columns.tolist()
print(dummy)
#-------------------------------------------------------------------------------------------------
# 建立 random forest 模型
max_sample = [0.7]
max_feature = [round(sqrt(len(dummy)))]
print(max_feature)
#fit--------------------------------------------------------------------------------------------------------------------
count = 0
i = 0
while i < len(max_sample):
    j = 0
    while j < len(max_feature):
        forest = ensemble.RandomForestClassifier(criterion="entropy",random_state=100,max_samples=max_sample[i],max_features=max_feature[j])
        model0 = forest.fit(train_simplify_x_dummies, train_simplify_y)
        # 預測
        test_simplify_y_predicted = model0.predict(test_simplify_x_dummies)
        predict = test_simplify_y_predicted.tolist()
        # -------------------------------------------------------
        test_simplify['predict'] = predict
        answer = {"species": test_simplify[factor[0]], "predict": predict}
        answer_df = pd.DataFrame(answer)
        print(answer_df)
        #-------------------------------------------------------------------------
        result0 = classification_report(test_simplify[factor[0]], test_simplify['predict'], output_dict=True)
        result0 = pd.DataFrame(result0)
        result0 = result0.round(4)
        print(result0)
        result0['accuracy'] = ["", "", result0.at['support', 'accuracy'], result0.at['support', 'macro avg']]
        result0 = result0.transpose()
        result0s = np.split(result0, [3])
        result0 = pd.concat([result0s[0], pd.DataFrame([[np.NaN] * 4], columns=result0.columns), result0s[1]])
        result0 = result0.rename(index={0: ""})
        #2222222222222222222222222222222222222222222222222222222222222-------------------------------------------------
        result0.to_csv("randomforest-result2" + str(count) +"-penguin.csv")
        #隨機森林視覺化圖-----------------------------------------------------------------------------------------------
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 2), dpi=900)
        for index in range(0, 5):
            plot_tree(forest.estimators_[index],feature_names=train_simplify_x_dummies.columns,filled=True,ax=axes[index])
            axes[index].set_title('Estimator: ' + str(index))
        fig.savefig("split2visual-rf" + str(count) + ".png")
        plt.cla()
        #混淆矩陣圖----------------------------------------------------------------------------------------------------
        plot_confusion_matrix(forest, test_simplify_x_dummies, test_simplify_y, cmap="Purples")
        plt.savefig("split2confmatr-rf" + str(count) + ".png")
        plt.cla()
        count = count + 1
        j = j + 1
    i = i + 1
    #---------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#train:test = 1 : 3---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
train_simplify = simplify.iloc[3::4].copy() # 3 7 11............
#print(test_simplify.index.tolist())
test_index = list(range(0,len(simplify),4)) + list(range(1,len(simplify),4)) + list(range(2,len(simplify),4))
test_index = sorted(test_index)
test_simplify = simplify.iloc[test_index].copy() # 0 1 2 4 5 6 8 9 10...............
#--------------------------------------------------------------------------------
factor = train_simplify.columns.tolist()
##print(factor)
train_simplify = train_simplify[train_simplify["sex"].isin(["MALE","FEMALE"])] #把sex不是 "MALE","FEMALE"的清掉，同時也清掉NA
train_simplify_x = train_simplify[factor[1:]]
train_simplify_y = train_simplify[factor[0]]
Std = StandardScaler()
train_simplify_x[factor[2:6]] = Std.fit_transform(train_simplify_x[factor[2:6]]) #標準化
train_simplify_x_dummies = pd.get_dummies(train_simplify_x)
#print(train_simplify_x_dummies)
#-------------------------------------------------------------------------------
test_simplify = test_simplify[test_simplify["sex"].isin(["MALE","FEMALE"])] #把sex不是 "MALE","FEMALE"的清掉，同時也清掉NA
test_simplify_x = test_simplify[factor[1:]]
test_simplify_y = test_simplify[factor[0]]
#print(test_simplify_y)
#print(test_simplify_x)
test_simplify_x[factor[2:6]] = Std.fit_transform(test_simplify_x[factor[2:6]]) #標準化
test_simplify_x_dummies = pd.get_dummies(test_simplify_x)
#print(test_simplify_x_dummies)
#--------------------------------------------------------------------------------
dummy = test_simplify_x_dummies.columns.tolist()
print(dummy)
#-------------------------------------------------------------------------------------------------
# 建立 random forest 模型
max_sample = [0.7]
max_feature = [round(sqrt(len(dummy)))]
print(max_feature)
#fit--------------------------------------------------------------------------------------------------------------------
count = 0
i = 0
while i < len(max_sample):
    j = 0
    while j < len(max_feature):
        forest = ensemble.RandomForestClassifier(criterion="entropy",random_state=100,max_samples=max_sample[i],max_features=max_feature[j])
        model0 = forest.fit(train_simplify_x_dummies, train_simplify_y)
        # 預測
        test_simplify_y_predicted = model0.predict(test_simplify_x_dummies)
        predict = test_simplify_y_predicted.tolist()
        # -------------------------------------------------------
        test_simplify['predict'] = predict
        answer = {"species": test_simplify[factor[0]], "predict": predict}
        answer_df = pd.DataFrame(answer)
        print(answer_df)
        #-------------------------------------------------------------------------
        result0 = classification_report(test_simplify[factor[0]], test_simplify['predict'], output_dict=True)
        result0 = pd.DataFrame(result0)
        result0 = result0.round(4)
        print(result0)
        result0['accuracy'] = ["", "", result0.at['support', 'accuracy'], result0.at['support', 'macro avg']]
        result0 = result0.transpose()
        result0s = np.split(result0, [3])
        result0 = pd.concat([result0s[0], pd.DataFrame([[np.NaN] * 4], columns=result0.columns), result0s[1]])
        result0 = result0.rename(index={0: ""})
        #333333333333333333333333333333333333333333333333333333333333333333333------------------------------------------
        result0.to_csv("randomforest-result3" + str(count) +"-penguin.csv")
        #隨機森林視覺化圖-----------------------------------------------------------------------------------------------
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 2), dpi=900)
        for index in range(0, 5):
            plot_tree(forest.estimators_[index],feature_names=train_simplify_x_dummies.columns,filled=True,ax=axes[index])
            axes[index].set_title('Estimator: ' + str(index))
        fig.savefig("split3visual-rf" + str(count) + ".png")
        plt.cla()
        #混淆矩陣圖----------------------------------------------------------------------------------------------------
        plot_confusion_matrix(forest, test_simplify_x_dummies, test_simplify_y, cmap="Purples")
        plt.savefig("split3confmatr-rf" + str(count) + ".png")
        plt.cla()
        count = count + 1
        j = j + 1
    i = i + 1
#-----------------------------------------------------------------------------------------------
#train:test = 1 : 9---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
train_simplify = simplify.iloc[9::10].copy() # 9 19 29............
train_index = train_simplify.index.tolist()
test_index = list(range(len(simplify)))
print(len(train_index))
print(len(test_index))
i = len(test_index) - 1
while i >= 0:
    if test_index[i] in train_index:
        test_index.remove(test_index[i])
    i = i - 1
print(len(test_index))
test_simplify = simplify.iloc[test_index].copy() # 0 1 2 3 4 5 6 7 8 10...............
#--------------------------------------------------------------------------------
factor = train_simplify.columns.tolist()
##print(factor)
train_simplify = train_simplify[train_simplify["sex"].isin(["MALE","FEMALE"])] #把sex不是 "MALE","FEMALE"的清掉，同時也清掉NA
train_simplify_x = train_simplify[factor[1:]]
train_simplify_y = train_simplify[factor[0]]
Std = StandardScaler()
train_simplify_x[factor[2:6]] = Std.fit_transform(train_simplify_x[factor[2:6]]) #標準化
train_simplify_x_dummies = pd.get_dummies(train_simplify_x)
#print(train_simplify_x_dummies)
#-------------------------------------------------------------------------------
test_simplify = test_simplify[test_simplify["sex"].isin(["MALE","FEMALE"])] #把sex不是 "MALE","FEMALE"的清掉，同時也清掉NA
test_simplify_x = test_simplify[factor[1:]]
test_simplify_y = test_simplify[factor[0]]
#print(test_simplify_y)
#print(test_simplify_x)
test_simplify_x[factor[2:6]] = Std.fit_transform(test_simplify_x[factor[2:6]]) #標準化
test_simplify_x_dummies = pd.get_dummies(test_simplify_x)
#print(test_simplify_x_dummies)
dummy = test_simplify_x_dummies.columns.tolist()
print(dummy)
#-------------------------------------------------------------------------------------------------
# 建立 random forest 模型
max_sample = [0.7]
max_feature = [round(sqrt(len(dummy)))]
print(max_feature)
#fit--------------------------------------------------------------------------------------------------------------------
count = 0
i = 0
while i < len(max_sample):
    j = 0
    while j < len(max_feature):
        forest = ensemble.RandomForestClassifier(criterion="entropy",random_state=100,max_samples=max_sample[i],max_features=max_feature[j])
        model0 = forest.fit(train_simplify_x_dummies, train_simplify_y)
        # 預測
        test_simplify_y_predicted = model0.predict(test_simplify_x_dummies)
        predict = test_simplify_y_predicted.tolist()
        # -------------------------------------------------------
        test_simplify['predict'] = predict
        answer = {"species": test_simplify[factor[0]], "predict": predict}
        answer_df = pd.DataFrame(answer)
        print(answer_df)
        #-------------------------------------------------------------------------
        result0 = classification_report(test_simplify[factor[0]], test_simplify['predict'], output_dict=True)
        result0 = pd.DataFrame(result0)
        result0 = result0.round(4)
        print(result0)
        result0['accuracy'] = ["", "", result0.at['support', 'accuracy'], result0.at['support', 'macro avg']]
        result0 = result0.transpose()
        result0s = np.split(result0, [3])
        result0 = pd.concat([result0s[0], pd.DataFrame([[np.NaN] * 4], columns=result0.columns), result0s[1]])
        result0 = result0.rename(index={0: ""})
        #444444444444444444444444444444444444444444444444444444444444444444444------------------------------------------
        result0.to_csv("randomforest-result4" + str(count) +"-penguin.csv")
        #隨機森林視覺化圖-----------------------------------------------------------------------------------------------
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 2), dpi=900)
        for index in range(0, 5):
            plot_tree(forest.estimators_[index],feature_names=train_simplify_x_dummies.columns,filled=True,ax=axes[index])
            axes[index].set_title('Estimator: ' + str(index))
        fig.savefig("split4visual-rf" + str(count) + ".png")
        plt.cla()
        #混淆矩陣圖----------------------------------------------------------------------------------------------------
        plot_confusion_matrix(forest, test_simplify_x_dummies, test_simplify_y, cmap="Purples")
        plt.savefig("split4confmatr-rf" + str(count) + ".png")
        plt.cla()
        count = count + 1
        j = j + 1
    i = i + 1
#-----------------------------------------------------------------------------------------------
#只取10個train--------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
train_simplify = simplify.iloc[34::32].copy() # 只取10個train............
train_index = train_simplify.index.tolist()
test_index = list(range(len(simplify)))
print(len(train_index))
print(len(test_index))
i = len(test_index) - 1
while i >= 0:
    if test_index[i] in train_index:
        test_index.remove(test_index[i])
    i = i - 1
print(len(test_index))
test_simplify = simplify.iloc[test_index].copy() #...............
#--------------------------------------------------------------------------------
factor = train_simplify.columns.tolist()
##print(factor)
train_simplify = train_simplify[train_simplify["sex"].isin(["MALE","FEMALE"])] #把sex不是 "MALE","FEMALE"的清掉，同時也清掉NA
train_simplify_x = train_simplify[factor[1:]]
train_simplify_y = train_simplify[factor[0]]
Std = StandardScaler()
train_simplify_x[factor[2:6]] = Std.fit_transform(train_simplify_x[factor[2:6]]) #標準化
train_simplify_x_dummies = pd.get_dummies(train_simplify_x)
#print(train_simplify_x_dummies)
#-------------------------------------------------------------------------------
test_simplify = test_simplify[test_simplify["sex"].isin(["MALE","FEMALE"])] #把sex不是 "MALE","FEMALE"的清掉，同時也清掉NA
test_simplify_x = test_simplify[factor[1:]]
test_simplify_y = test_simplify[factor[0]]
#print(test_simplify_y)
#print(test_simplify_x)
test_simplify_x[factor[2:6]] = Std.fit_transform(test_simplify_x[factor[2:6]]) #標準化
test_simplify_x_dummies = pd.get_dummies(test_simplify_x)
#print(test_simplify_x_dummies)
dummy = test_simplify_x_dummies.columns.tolist()
print(dummy)
#-------------------------------------------------------------------------------------------------
# 建立 random forest 模型
max_sample = [0.7]
max_feature = [round(sqrt(len(dummy)))]
print(max_feature)
#fit--------------------------------------------------------------------------------------------------------------------
count = 0
i = 0
while i < len(max_sample):
    j = 0
    while j < len(max_feature):
        forest = ensemble.RandomForestClassifier(criterion="entropy",random_state=100,max_samples=max_sample[i],max_features=max_feature[j])
        model0 = forest.fit(train_simplify_x_dummies, train_simplify_y)
        # 預測
        test_simplify_y_predicted = model0.predict(test_simplify_x_dummies)
        predict = test_simplify_y_predicted.tolist()
        # -------------------------------------------------------
        test_simplify['predict'] = predict
        answer = {"species": test_simplify[factor[0]], "predict": predict}
        answer_df = pd.DataFrame(answer)
        print(answer_df)
        #-------------------------------------------------------------------------
        result0 = classification_report(test_simplify[factor[0]], test_simplify['predict'], output_dict=True)
        result0 = pd.DataFrame(result0)
        result0 = result0.round(4)
        print(result0)
        result0['accuracy'] = ["", "", result0.at['support', 'accuracy'], result0.at['support', 'macro avg']]
        result0 = result0.transpose()
        result0s = np.split(result0, [3])
        result0 = pd.concat([result0s[0], pd.DataFrame([[np.NaN] * 4], columns=result0.columns), result0s[1]])
        result0 = result0.rename(index={0: ""})
        #5555555555555555555555555555555555555555555555555555555555555555555555------------------------------------------
        result0.to_csv("randomforest-result5" + str(count) +"-penguin.csv")
        #隨機森林視覺化圖-----------------------------------------------------------------------------------------------
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 2), dpi=900)
        for index in range(0, 5):
            plot_tree(forest.estimators_[index],feature_names=train_simplify_x_dummies.columns,filled=True,ax=axes[index])
            axes[index].set_title('Estimator: ' + str(index))
        fig.savefig("split5visual-rf" + str(count) + ".png")
        plt.cla()
        #混淆矩陣圖----------------------------------------------------------------------------------------------------
        plot_confusion_matrix(forest, test_simplify_x_dummies, test_simplify_y, cmap="Purples")
        plt.savefig("split5confmatr-rf" + str(count) + ".png")
        plt.cla()
        count = count + 1
        j = j + 1
    i = i + 1