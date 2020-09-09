import numpy as np
from sklearn.metrics import roc_curve
from sklearn import metrics
#=============================================================================#
################################ test function ################################
#=============================================================================#

def test_CNN(model,x_test,y_test):  
    y_pred = model.predict(x_test)  
    cls_pre = np.argmax(y_pred,axis=1)
    cls_true = np.argmax(y_test,axis=1)
    
    indics_false = np.where(cls_pre!=cls_true)[0]
    indics_true = np.where(cls_pre==cls_true)[0]
    
    population = len(cls_true)
    # positive = bad = wheel flat # negative = good = intact wheel
    # TP = coorectly diagnosis # FP = false alarm # FN = non-success diagnosis
    # recall reveals the cablability of diagnosis
    # precision indicates the avoidancy of false alarm
    T = len(indics_true)
#    F = len(indics_false)
#    TP = len(cls_pre[indics_true]) - np.count_nonzero(cls_pre[indics_true])
#    FP = len(cls_pre[indics_false])- np.count_nonzero(cls_pre[indics_false]) #
#    FN = F-FP   
    acc = T/population
#    recall = TP/(TP+FN)
#    precision = TP/(TP+FP)
#    f1 = 2*recall*precision/(recall+precision)
#    fpr, tpr, _ = roc_curve(cls_true, cls_pre)
#    auc= metrics.auc(fpr, tpr) 
    print('acc=%.2f' %(acc*100)+'%')
#    print('recall=%.2f' %(recall*100)+'%')
#    print('precision=%.2f' %(precision*100)+'%')
##    print('f1 score=%.2f' %(f1*100)+'%')
#    print('AUC=%.2f' %(auc*100)+'%')
    return acc,cls_pre,cls_true,indics_false,indics_true

def test_GBDT(model,x_test,y_test):     
    y_pred = model.predict_proba(x_test)    
    cls_pre = np.argmax(y_pred,axis=1)
    cls_true = np.argmax(y_test,axis=1)
    
    indics_false = np.where(cls_pre!=cls_true)[0]
    indics_true = np.where(cls_pre==cls_true)[0]
    
    population = len(cls_true)
    # positive = bad = wheel flat # negative = good = intact wheel
    # TP = coorectly diagnosis # FP = false alarm # FN = non-success diagnosis
    # recall reveals the cablability of diagnosis
    # precision indicates the avoidancy of false alarm
    T = len(indics_true)
#    F = len(indics_false)
#    TP = len(cls_pre[indics_true]) - np.count_nonzero(cls_pre[indics_true])
#    FP = len(cls_pre[indics_false])- np.count_nonzero(cls_pre[indics_false]) #
#    FN = F-FP   
    acc = T/population
#    recall = TP/(TP+FN)
#    precision = TP/(TP+FP)
#    f1 = 2*recall*precision/(recall+precision)
#    fpr, tpr, _ = roc_curve(cls_true, cls_pre)
#    auc= metrics.auc(fpr, tpr) 
    print('acc=%.2f' %(acc*100)+'%')
#    print('recall=%.2f' %(recall*100)+'%')
#    print('precision=%.2f' %(precision*100)+'%')
##    print('f1 score=%.2f' %(f1*100)+'%')
#    print('AUC=%.2f' %(auc*100)+'%')
    return acc,cls_pre,cls_true,indics_false,indics_true