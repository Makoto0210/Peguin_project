import pandas as pd
from sklearn import tree
import sklearn
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
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
#--------------------------------------------------------------------------------
# 建立分類器
clf = tree.DecisionTreeClassifier(random_state=2)
species_clf = clf.fit(train_simplify_x_dummies, train_simplify_y)
cclf = tree.DecisionTreeClassifier(criterion = "entropy",random_state=2)
species_cclf = cclf.fit(train_simplify_x_dummies, train_simplify_y)
# 預測
test_simplify_y_predicted = species_clf.predict(test_simplify_x_dummies)
#---------------------------------------------------
test_simplify_y_predicted1 = species_cclf.predict(test_simplify_x_dummies)
predict = test_simplify_y_predicted.tolist()
predict1 = test_simplify_y_predicted1.tolist()
#-------------------------------------------------------
test_simplify['predict'] = predict
test_simplify['predict1'] = predict1
#print(test_simplify[[factor[0],'predict','predict1']])
#00000000000000000000000000000000000000---------------------------------------------------------------------------------
answer = {"species": test_simplify[factor[0]],"predict": predict,"predict1": predict1}
answer_df = pd.DataFrame(answer)
answer_df.to_csv("penguintree0.csv")
#視覺化決策樹圖-----------------------------------------------------------------------------------------------------------
plt.figure(figsize=(11,11),dpi=200)
plot_tree(clf,feature_names=train_simplify_x_dummies.columns, filled=True)
plt.savefig("split0visual-dtree0.png")
plt.cla()
plt.figure(figsize=(11,11),dpi=200)
plot_tree(cclf,feature_names=train_simplify_x_dummies.columns, filled=True)
plt.savefig("split0visual-dtree1.png")
plt.cla()
#混淆矩陣圖-----------------------------------------------------------------------------------------------------------
plot_confusion_matrix(clf,test_simplify_x_dummies,test_simplify_y,cmap="Purples")
plt.savefig("split0confmatr-dtree0.png")
plt.cla()

plot_confusion_matrix(cclf,test_simplify_x_dummies,test_simplify_y,cmap="Purples")
plt.savefig("split0confmatr-dtree1.png")
plt.cla()
#------------------------------------------------------------------------------------------------------
result0 = classification_report(test_simplify[factor[0]],test_simplify['predict'],output_dict = True)
result0 = pd.DataFrame(result0)
result0 = result0.round(4)
print(result0)
result0['accuracy'] = ["","",result0.at['support', 'accuracy'],result0.at['support', 'macro avg']]
result0 = result0.transpose()
result0s = np.split(result0,[3])
result0 = pd.concat([result0s[0],pd.DataFrame([[np.NaN]*4],columns = result0.columns),result0s[1]])
result0 = result0.rename(index={0: ""})
result0.to_csv("decisiontree-result00-penguin.csv")
#------------------------------------------------------------------------------------------------------------
result1 = classification_report(test_simplify[factor[0]],test_simplify['predict1'],output_dict = True)
result1 = pd.DataFrame(result1)
result1 = result1.round(4)
print(result1)
result1['accuracy'] = ["","",result1.at['support', 'accuracy'],result1.at['support', 'macro avg']]
result1 = result1.transpose()
result1s = np.split(result1,[3])
result1 = pd.concat([result1s[0],pd.DataFrame([[np.NaN]*4],columns = result1.columns),result1s[1]])
result1 = result1.rename(index={0: ""})
result1.to_csv("decisiontree-result01-penguin.csv")
#-----------------------------------------------------------------------------------------------
#train:test = 3: 1---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
test_simplify = simplify.iloc[3::4].copy() # 3 7 11............
#print(test_simplify.index.tolist())
train_index = list(range(0,len(simplify),4)) + list(range(1,len(simplify),4)) + list(range(2,len(simplify),4))
train_index = sorted(train_index)
train_simplify = simplify.iloc[train_index].copy() # 0 1 2 4 5 6 8 9 10...............
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
# 建立分類器
clf = tree.DecisionTreeClassifier(random_state=11)
species_clf = clf.fit(train_simplify_x_dummies, train_simplify_y)
cclf = tree.DecisionTreeClassifier(criterion = "entropy",random_state=11)
species_cclf = cclf.fit(train_simplify_x_dummies, train_simplify_y)
# 預測
test_simplify_y_predicted = species_clf.predict(test_simplify_x_dummies)
#---------------------------------------------------
test_simplify_y_predicted1 = species_cclf.predict(test_simplify_x_dummies)
predict = test_simplify_y_predicted.tolist()
predict1 = test_simplify_y_predicted1.tolist()
#-------------------------------------------------------
test_simplify['predict'] = predict
test_simplify['predict1'] = predict1
#print(test_simplify[[factor[0],'predict','predict1']])
#111111111111111111111111111111111111111111111111111111111111111111111--------------------------------------------------
answer = {"species": test_simplify[factor[0]],"predict": predict,"predict1": predict1}
answer_df = pd.DataFrame(answer)
answer_df.to_csv("penguintree1.csv")
#視覺化決策樹圖-----------------------------------------------------------------------------------------------------------
plt.figure(figsize=(11,11),dpi=200)
plot_tree(clf,feature_names=train_simplify_x_dummies.columns, filled=True)
plt.savefig("split1visual-dtree0.png")
plt.cla()
plt.figure(figsize=(11,11),dpi=200)
plot_tree(cclf,feature_names=train_simplify_x_dummies.columns, filled=True)
plt.savefig("split1visual-dtree1.png")
plt.cla()
#混淆矩陣圖-----------------------------------------------------------------------------------------------------------
plot_confusion_matrix(clf,test_simplify_x_dummies,test_simplify_y,cmap="Purples")
plt.savefig("split1confmatr-dtree0.png")
plt.cla()

plot_confusion_matrix(cclf,test_simplify_x_dummies,test_simplify_y,cmap="Purples")
plt.savefig("split1confmatr-dtree1.png")
plt.cla()
#------------------------------------------------------------------------------------------------------
result0 = classification_report(test_simplify[factor[0]],test_simplify['predict'],output_dict = True)
result0 = pd.DataFrame(result0)
result0 = result0.round(4)
print(result0)
result0['accuracy'] = ["","",result0.at['support', 'accuracy'],result0.at['support', 'macro avg']]
result0 = result0.transpose()
result0s = np.split(result0,[3])
result0 = pd.concat([result0s[0],pd.DataFrame([[np.NaN]*4],columns = result0.columns),result0s[1]])
result0 = result0.rename(index={0: ""})
result0.to_csv("decisiontree-result10-penguin.csv")
#------------------------------------------------------------------------------------------------------------
result1 = classification_report(test_simplify[factor[0]],test_simplify['predict1'],output_dict = True)
result1 = pd.DataFrame(result1)
result1 = result1.round(4)
print(result1)
result1['accuracy'] = ["","",result1.at['support', 'accuracy'],result1.at['support', 'macro avg']]
result1 = result1.transpose()
result1s = np.split(result1,[3])
result1 = pd.concat([result1s[0],pd.DataFrame([[np.NaN]*4],columns = result1.columns),result1s[1]])
result1 = result1.rename(index={0: ""})
result1.to_csv("decisiontree-result11-penguin.csv")
#-----------------------------------------------------------------------------------------------
#train:test = 1 : 1---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
test_simplify = simplify.iloc[1::2].copy() #1 3 5 7............
#print(test_simplify.index.tolist())
train_index = list(range(0,len(simplify),2))
train_simplify = simplify.iloc[train_index].copy() # 0 2 4 6 8 10...............
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
# 建立分類器
clf = tree.DecisionTreeClassifier(random_state=11)
species_clf = clf.fit(train_simplify_x_dummies, train_simplify_y)
cclf = tree.DecisionTreeClassifier(criterion = "entropy",random_state=11)
species_cclf = cclf.fit(train_simplify_x_dummies, train_simplify_y)
# 預測
test_simplify_y_predicted = species_clf.predict(test_simplify_x_dummies)
#---------------------------------------------------
test_simplify_y_predicted1 = species_cclf.predict(test_simplify_x_dummies)
predict = test_simplify_y_predicted.tolist()
predict1 = test_simplify_y_predicted1.tolist()
#-------------------------------------------------------
test_simplify['predict'] = predict
test_simplify['predict1'] = predict1
#print(test_simplify[[factor[0],'predict','predict1']])
#22222222222222222222222222222222222222222222222222222222----------------------------------------------------------------
answer = {"species": test_simplify[factor[0]],"predict": predict,"predict1": predict1}
answer_df = pd.DataFrame(answer)
answer_df.to_csv("penguintree2.csv")
#視覺化決策樹圖-----------------------------------------------------------------------------------------------------------
plt.figure(figsize=(11,11),dpi=200)
plot_tree(clf,feature_names=train_simplify_x_dummies.columns, filled=True)
plt.savefig("split2visual-dtree0.png")
plt.cla()
plt.figure(figsize=(11,11),dpi=200)
plot_tree(cclf,feature_names=train_simplify_x_dummies.columns, filled=True)
plt.savefig("split2visual-dtree1.png")
plt.cla()
#混淆矩陣圖-----------------------------------------------------------------------------------------------------------
plot_confusion_matrix(clf,test_simplify_x_dummies,test_simplify_y,cmap="Purples")
plt.savefig("split2confmatr-dtree0.png")
plt.cla()

plot_confusion_matrix(cclf,test_simplify_x_dummies,test_simplify_y,cmap="Purples")
plt.savefig("split2confmatr-dtree1.png")
plt.cla()
#------------------------------------------------------------------------------------------------------
result0 = classification_report(test_simplify[factor[0]],test_simplify['predict'],output_dict = True)
result0 = pd.DataFrame(result0)
result0 = result0.round(4)
print(result0)
result0['accuracy'] = ["","",result0.at['support', 'accuracy'],result0.at['support', 'macro avg']]
result0 = result0.transpose()
result0s = np.split(result0,[3])
result0 = pd.concat([result0s[0],pd.DataFrame([[np.NaN]*4],columns = result0.columns),result0s[1]])
result0 = result0.rename(index={0: ""})
result0.to_csv("decisiontree-result20-penguin.csv")
#------------------------------------------------------------------------------------------------------------
result1 = classification_report(test_simplify[factor[0]],test_simplify['predict1'],output_dict = True)
result1 = pd.DataFrame(result1)
result1 = result1.round(4)
print(result1)
result1['accuracy'] = ["","",result1.at['support', 'accuracy'],result1.at['support', 'macro avg']]
result1 = result1.transpose()
result1s = np.split(result1,[3])
result1 = pd.concat([result1s[0],pd.DataFrame([[np.NaN]*4],columns = result1.columns),result1s[1]])
result1 = result1.rename(index={0: ""})
result1.to_csv("decisiontree-result21-penguin.csv")
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
# 建立分類器
clf = tree.DecisionTreeClassifier(random_state=11)
species_clf = clf.fit(train_simplify_x_dummies, train_simplify_y)
cclf = tree.DecisionTreeClassifier(criterion = "entropy",random_state=11)
species_cclf = cclf.fit(train_simplify_x_dummies, train_simplify_y)
# 預測
test_simplify_y_predicted = species_clf.predict(test_simplify_x_dummies)
#---------------------------------------------------
test_simplify_y_predicted1 = species_cclf.predict(test_simplify_x_dummies)
predict = test_simplify_y_predicted.tolist()
predict1 = test_simplify_y_predicted1.tolist()
#-------------------------------------------------------
test_simplify['predict'] = predict
test_simplify['predict1'] = predict1
#print(test_simplify[[factor[0],'predict','predict1']])
#333333333333333333333333333333333333333333333333333333333333333--------------------------------------------------------
answer = {"species": test_simplify[factor[0]],"predict": predict,"predict1": predict1}
answer_df = pd.DataFrame(answer)
answer_df.to_csv("penguintree3.csv")
#視覺化決策樹圖-----------------------------------------------------------------------------------------------------------
plt.figure(figsize=(11,11),dpi=200)
plot_tree(clf,feature_names=train_simplify_x_dummies.columns, filled=True)
plt.savefig("split3visual-dtree0.png")
plt.cla()
plt.figure(figsize=(11,11),dpi=200)
plot_tree(cclf,feature_names=train_simplify_x_dummies.columns, filled=True)
plt.savefig("split3visual-dtree1.png")
plt.cla()
#混淆矩陣圖-----------------------------------------------------------------------------------------------------------
plot_confusion_matrix(clf,test_simplify_x_dummies,test_simplify_y,cmap="Purples")
plt.savefig("split3confmatr-dtree0.png")
plt.cla()

plot_confusion_matrix(cclf,test_simplify_x_dummies,test_simplify_y,cmap="Purples")
plt.savefig("split3confmatr-dtree1.png")
plt.cla()
#------------------------------------------------------------------------------------------------------
result0 = classification_report(test_simplify[factor[0]],test_simplify['predict'],output_dict = True)
result0 = pd.DataFrame(result0)
result0 = result0.round(4)
print(result0)
result0['accuracy'] = ["","",result0.at['support', 'accuracy'],result0.at['support', 'macro avg']]
result0 = result0.transpose()
result0s = np.split(result0,[3])
result0 = pd.concat([result0s[0],pd.DataFrame([[np.NaN]*4],columns = result0.columns),result0s[1]])
result0 = result0.rename(index={0: ""})
result0.to_csv("decisiontree-result30-penguin.csv")
#------------------------------------------------------------------------------------------------------------
result1 = classification_report(test_simplify[factor[0]],test_simplify['predict1'],output_dict = True)
result1 = pd.DataFrame(result1)
result1 = result1.round(4)
print(result1)
result1['accuracy'] = ["","",result1.at['support', 'accuracy'],result1.at['support', 'macro avg']]
result1 = result1.transpose()
result1s = np.split(result1,[3])
result1 = pd.concat([result1s[0],pd.DataFrame([[np.NaN]*4],columns = result1.columns),result1s[1]])
result1 = result1.rename(index={0: ""})
result1.to_csv("decisiontree-result31-penguin.csv")
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
#--------------------------------------------------------------------------------
# 建立分類器
clf = tree.DecisionTreeClassifier(random_state=11)
species_clf = clf.fit(train_simplify_x_dummies, train_simplify_y)
cclf = tree.DecisionTreeClassifier(criterion = "entropy",random_state=11)
species_cclf = cclf.fit(train_simplify_x_dummies, train_simplify_y)
# 預測
test_simplify_y_predicted = species_clf.predict(test_simplify_x_dummies)
#---------------------------------------------------
test_simplify_y_predicted1 = species_cclf.predict(test_simplify_x_dummies)
predict = test_simplify_y_predicted.tolist()
predict1 = test_simplify_y_predicted1.tolist()
#-------------------------------------------------------
test_simplify['predict'] = predict
test_simplify['predict1'] = predict1
#print(test_simplify[[factor[0],'predict','predict1']])
#444444444444444444444444444444444444444444444444444444444444444444444--------------------------------------------------
answer = {"species": test_simplify[factor[0]],"predict": predict,"predict1": predict1}
answer_df = pd.DataFrame(answer)
answer_df.to_csv("penguintree4.csv")
#視覺化決策樹圖-----------------------------------------------------------------------------------------------------------
plt.figure(figsize=(11,11),dpi=200)
plot_tree(clf,feature_names=train_simplify_x_dummies.columns, filled=True)
plt.savefig("split4visual-dtree0.png")
plt.cla()
plt.figure(figsize=(11,11),dpi=200)
plot_tree(cclf,feature_names=train_simplify_x_dummies.columns, filled=True)
plt.savefig("split5visual-dtree1.png")
plt.cla()
#混淆矩陣圖-----------------------------------------------------------------------------------------------------------
plot_confusion_matrix(clf,test_simplify_x_dummies,test_simplify_y,cmap="Purples")
plt.savefig("split4confmatr-dtree0.png")
plt.cla()

plot_confusion_matrix(cclf,test_simplify_x_dummies,test_simplify_y,cmap="Purples")
plt.savefig("split4confmatr-dtree1.png")
plt.cla()
#------------------------------------------------------------------------------------------------------
result0 = classification_report(test_simplify[factor[0]],test_simplify['predict'],output_dict = True)
result0 = pd.DataFrame(result0)
result0 = result0.round(4)
print(result0)
result0['accuracy'] = ["","",result0.at['support', 'accuracy'],result0.at['support', 'macro avg']]
result0 = result0.transpose()
result0s = np.split(result0,[3])
result0 = pd.concat([result0s[0],pd.DataFrame([[np.NaN]*4],columns = result0.columns),result0s[1]])
result0 = result0.rename(index={0: ""})
result0.to_csv("decisiontree-result40-penguin.csv")
#------------------------------------------------------------------------------------------------------------
result1 = classification_report(test_simplify[factor[0]],test_simplify['predict1'],output_dict = True)
result1 = pd.DataFrame(result1)
result1 = result1.round(4)
print(result1)
result1['accuracy'] = ["","",result1.at['support', 'accuracy'],result1.at['support', 'macro avg']]
result1 = result1.transpose()
result1s = np.split(result1,[3])
result1 = pd.concat([result1s[0],pd.DataFrame([[np.NaN]*4],columns = result1.columns),result1s[1]])
result1 = result1.rename(index={0: ""})
result1.to_csv("decisiontree-result41-penguin.csv")
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
#--------------------------------------------------------------------------------
# 建立分類器
clf = tree.DecisionTreeClassifier(random_state=11)
species_clf = clf.fit(train_simplify_x_dummies, train_simplify_y)
cclf = tree.DecisionTreeClassifier(criterion = "entropy",random_state=11)
species_cclf = cclf.fit(train_simplify_x_dummies, train_simplify_y)
# 預測
test_simplify_y_predicted = species_clf.predict(test_simplify_x_dummies)
#---------------------------------------------------
test_simplify_y_predicted1 = species_cclf.predict(test_simplify_x_dummies)
predict = test_simplify_y_predicted.tolist()
predict1 = test_simplify_y_predicted1.tolist()
#-------------------------------------------------------
test_simplify['predict'] = predict
test_simplify['predict1'] = predict1
#print(test_simplify[[factor[0],'predict','predict1']])
#5555555555555555555555555555555555555555555555555555555555555555555----------------------------------------------------
answer = {"species": test_simplify[factor[0]],"predict": predict,"predict1": predict1}
answer_df = pd.DataFrame(answer)
answer_df.to_csv("penguintree5.csv")
#視覺化決策樹圖-----------------------------------------------------------------------------------------------------------
plt.figure(figsize=(11,11),dpi=200)
plot_tree(clf,feature_names=train_simplify_x_dummies.columns, filled=True)
plt.savefig("split5visual-dtree0.png")
plt.cla()
plt.figure(figsize=(11,11),dpi=200)
plot_tree(cclf,feature_names=train_simplify_x_dummies.columns, filled=True)
plt.savefig("split5visual-dtree1.png")
plt.cla()
#混淆矩陣圖-----------------------------------------------------------------------------------------------------------
plot_confusion_matrix(clf,test_simplify_x_dummies,test_simplify_y,cmap="Purples")
plt.savefig("split5confmatr-dtree0.png")
plt.cla()

plot_confusion_matrix(cclf,test_simplify_x_dummies,test_simplify_y,cmap="Purples")
plt.savefig("split5confmatr-dtree1.png")
plt.cla()
#------------------------------------------------------------------------------------------------------
result0 = classification_report(test_simplify[factor[0]],test_simplify['predict'],output_dict = True)
result0 = pd.DataFrame(result0)
result0 = result0.round(4)
print(result0)
result0['accuracy'] = ["","",result0.at['support', 'accuracy'],result0.at['support', 'macro avg']]
result0 = result0.transpose()
result0s = np.split(result0,[3])
result0 = pd.concat([result0s[0],pd.DataFrame([[np.NaN]*4],columns = result0.columns),result0s[1]])
result0 = result0.rename(index={0: ""})
result0.to_csv("decisiontree-result50-penguin.csv")
#------------------------------------------------------------------------------------------------------------
result1 = classification_report(test_simplify[factor[0]],test_simplify['predict1'],output_dict = True)
result1 = pd.DataFrame(result1)
result1 = result1.round(4)
print(result1)
result1['accuracy'] = ["","",result1.at['support', 'accuracy'],result1.at['support', 'macro avg']]
result1 = result1.transpose()
result1s = np.split(result1,[3])
result1 = pd.concat([result1s[0],pd.DataFrame([[np.NaN]*4],columns = result1.columns),result1s[1]])
result1 = result1.rename(index={0: ""})
result1.to_csv("decisiontree-result51-penguin.csv")
