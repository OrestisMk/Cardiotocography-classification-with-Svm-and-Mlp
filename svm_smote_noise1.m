
tic
clear;
clc;

%import data
opts = detectImportOptions('shuflenoise.csv','NumHeaderLines',1);
ctg_smoted_noise = readtable('shuflenoise.csv',opts);
ctg_smoted_noise.Properties.VariableNames([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27]) = {'A','LB','AC','FM','UC','DL','DS','DP','ASTV','MSTV','ALTV','MLTV','Width','Min','Max','Nmax','Nzeros','Mode','Mean','Median','Variance','Tendency','0','1','2','3','status'}



summary(ctg_smoted_noise);


% size table (2644 rows, 27 columns)
[r c] = size(ctg_smoted_noise);


% seperation of depentent and independent variables
Xsmt = table2array(ctg_smoted_noise(:,2:26));
Ysmt = table2array(ctg_smoted_noise(:,27));

%save all variables
dep_variables = ctg_smoted_noise.Properties.VariableNames;
Xsmt_variables = ctg_smoted_noise(:,2:26);



 
 %define classes
count_y0_y1 = tabulate(Ysmt)% 
 


%seperate data into training and test sets using cv partition to account
cvsmt = cvpartition(Ysmt,'holdout',0.30);
Xsmt_Train = Xsmt(training(cvsmt,1),:);
ysmt_Train = Ysmt(training(cvsmt,1));
Xsmt_Test = Xsmt(test(cvsmt,1),:);
ysmt_Test = Ysmt(test(cvsmt,1),:);



rng(1)
CVSVMModel = fitcsvm(Xsmt_Train,ysmt_Train,'Holdout',0.3,'ClassNames',{'0','1'},...
    'Standardize',false);
CompactSVMModel = CVSVMModel.Trained{1}; % Extract trained, compact classifier
testInds = test(CVSVMModel.Partition);   % Extract the test indices
classLoss = kfoldLoss(CVSVMModel)%Estimate the out-of-sample misclassification rate



[svm_yPrd, svm_scr,conio] = predict(CompactSVMModel,Xsmt_Test);% returns predicted class labels based on the trained classification model
svm_loss = loss(CompactSVMModel,Xsmt_Test, ysmt_Test); %classification effective of training data based on model predictions


%%% confusion matrix
figure(figure('Name', 'svm'))
[svm_cm, order] = confusionmat(ysmt_Test, svm_yPrd)
cm1chart_svm = confusionchart( ysmt_Test, svm_yPrd)

%%roc curve
[X1,Y1] = perfcurve(ysmt_Test, svm_yPrd,1)

%accurcy of knn model with bayesopt tuning.
Accuracy_svm = 100*(svm_cm(1,1)+svm_cm(2,2))./(svm_cm(1,1)+svm_cm(2,2)+svm_cm(1,2)+svm_cm(2,1))

%Precision defines the accuracy of judgment.
svm_precision = svm_cm(1,1)./(svm_cm(1,1)+svm_cm(1,2));


%Recall is the ability to identify the number of samples that would really count positive for fetal distress.
svm_recall = svm_cm(1,1)./(svm_cm(1,1)+svm_cm(2,1));

%F1-score means a statistical measure of the accuracy. Also , F1 score is used because FN and TN are crusial for our results. 
f1_Scores_simple = 2*(svm_precision.*svm_recall)./(svm_precision+svm_recall)




% Identify misclassified patient
testheight = size(Xsmt,1)
trainheight = size(Ysmt,1)
misClass_svm = (svm_cm(1,2)+svm_cm(2,1));
errFT = 100*misClass_svm/testheight;

%%
%Bayes Optimisation
 
%Index and partition for second cvpartitioned dataset
foldIdx = size(Xsmt_Train,1)
cv1 = cvpartition(foldIdx, 'kfold', 10)

opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',cv1,...
    'AcquisitionFunctionName','expected-improvement-plus');
svmmod = fitcsvm(Xsmt_Train,ysmt_Train,'KernelFunction','rbf',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)

lossnew = kfoldLoss(fitcsvm(Xsmt_Train,ysmt_Train,'CVPartition',cv1,'KernelFunction','rbf',...
    'BoxConstraint',svmmod.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint,...
    'KernelScale',svmmod.HyperparameterOptimizationResults.XAtMinObjective.KernelScale))

[svm_yPrd_bayesopt, svm_scr_bayesopt,conio] = predict(svmmod,Xsmt_Test);% returns predicted class labels based on the trained classification model
knn_loss = loss(svmmod,Xsmt_Test, ysmt_Test); %classification effective of training data based on model predictions
knn_rloss = resubLoss(svmmod); %Misclassifications from the predictions above 

%%% confusion matrix
figure('Name', 'svm_bayesopt_cm')
[svm_bayesopt_cm, order] = confusionmat(ysmt_Test, svm_yPrd_bayesopt)
cm1chart_bayesopt = confusionchart( ysmt_Test, svm_yPrd_bayesopt)

%roc curve
[X2,Y2] = perfcurve(ysmt_Test, svm_yPrd_bayesopt,1)


%accurcy of knn model with bayesopt tuning.
Accuracy_svm_bayesopt = 100*(svm_bayesopt_cm(1,1)+svm_bayesopt_cm(2,2))./(svm_bayesopt_cm(1,1)+svm_bayesopt_cm(2,2)+svm_bayesopt_cm(1,2)+svm_bayesopt_cm(2,1))

%Precision defines the accuracy of judgment.
svm_precision = svm_bayesopt_cm(1,1)./(svm_bayesopt_cm(1,1)+svm_bayesopt_cm(1,2));


%Recall is the ability to identify the number of samples that would really count positive for tumor.
svm_bayeopt_recall = svm_bayesopt_cm(1,1)./(svm_bayesopt_cm(1,1)+svm_bayesopt_cm(2,1));

%F1-score means a statistical measure of the accuracy. Also , F1 score is used because FN and TN are crusial for our results. 
f1_Scores_bayesopt = 2*(svm_precision.*svm_bayeopt_recall)./(svm_precision+svm_bayeopt_recall)


plot(X1,Y1)
hold on
plot(X2,Y2)
legend('SVM','SVM bayes_opt')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for SVM')
hold off

% Identify misclassified tumours
testheight_bayes = size(Xsmt,1)
trainheight_bayes = size(Ysmt,1)
misClass_bayesopt = (svm_bayesopt_cm(1,2)+svm_bayesopt_cm(2,1));
errFT = 100*misClass_bayesopt/testheight;
toc;