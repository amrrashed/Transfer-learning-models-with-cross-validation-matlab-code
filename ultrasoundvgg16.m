clc;close all;clear;%delete(findall(0));
%%cross validation
%load images
digitDatasetPath = fullfile('D:\CIT project\datasets\ultrasonic images\us-dataset\modified 224');
 imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
% Determine the split up
total_split=countEachLabel(imds)
% Number of Images
num_images=length(imds.Labels);
% Visualize random images
perm=randperm(num_images,6);

% for idx=1:length(perm)
%     
%     subplot(2,3,idx);
%     imshow(imread(imds.Files{perm(idx)}));
%     title(sprintf('%s',imds.Labels(perm(idx))))
%     
% end

%%K-fold Validation
% Number of folds
num_folds=5;

% Loop for each fold
for fold_idx=1:num_folds
    
    fprintf('Processing %d among %d folds \n',fold_idx,num_folds);
    
   % Test Indices for current fold
    test_idx=fold_idx:num_folds:num_images;

    % Test cases for current fold
    imdsTest = subset(imds,test_idx);
    
    % Train indices for current fold
    train_idx=setdiff(1:length(imds.Files),test_idx);
    
    % Train cases for current fold
    imdsTrain = subset(imds,train_idx);
 
    % alexNet Architecture 
    net=vgg16;
     % Replacing the last layers with new layers
    layersTransfer = net.Layers(1:end-3);
    clear net;
 % Number of categories
    numClasses = numel(categories(imdsTrain.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

    
    %rmsprop,sgdm,adam
    % Training Options, we choose a small mini-batch size due to limited images 
   options = trainingOptions('sgdm',...
        'ExecutionEnvironment','auto',...
        'MaxEpochs',20,'MiniBatchSize',8,...
        'Shuffle','every-epoch', ...
        'InitialLearnRate',1e-4, ...
        'Verbose',false, ...
        'Plots','training-progress',...
        'OutputFcn',@(info)savetrainingplot(info));
%      'L2Regularization',1e-10,...
% 'OutputFcn',@(info)savetrainingplot(info)
%           'LearnRateSchedule','piecewise', ...
    % Data Augumentation
    augmenter = imageDataAugmenter( ...
        'RandRotation',[-5 5],'RandXReflection',1,...
        'RandYReflection',1,'RandXShear',[-0.05 0.05],'RandYShear',[-0.05 0.05]);
    
    % Resizing all training images to [224 224] for ResNet architecture
    auimds = augmentedImageDatastore([224 224],imdsTrain,'DataAugmentation',augmenter);
    
    % Training
     netTransfer = trainNetwork(auimds,layers,options);
%     netTransfer = trainNetwork(imdsTrain,layers,options);
    % Resizing all testing images to [224 224] for ResNet architecture   
    augtestimds = augmentedImageDatastore([224 224],imdsTest);
    
   
    % Testing and their corresponding Labels and Posterior for each Case
   % [predicted_labels(test_idx),posterior(test_idx,:)] = classify(netTransfer,augtestimds);
    [predicted_labels(test_idx),posterior(test_idx,:)] = classify(netTransfer,augtestimds,'ExecutionEnvironment','cpu');
    % Save the Independent ResNet Architectures obtained for each Fold
    save(sprintf('vgg16_%d_among_%d_folds',fold_idx,num_folds),'netTransfer','test_idx','train_idx');
     delete(findall(0))
    % Clearing unnecessary variables 
    clearvars -except fold_idx num_folds num_images predicted_labels posterior imds netTransfer;
    
end
analyzeNetwork(netTransfer)
%%Performance Study
% Actual Labels
actual_labels=imds.Labels;

% Confusion Matrix
f  = confusionmat(actual_labels,predicted_labels')
figure;
plotconfusion(actual_labels,predicted_labels')
title('Confusion Matrix: vgg16 ');
%ROC CURVE
test_labels=double(nominal(imds.Labels));

% ROC Curve - Our target class is the first class in this scenario 
[fp_rate,tp_rate,T,AUC]=perfcurve(test_labels,posterior(:,1),1);
figure;
plot(fp_rate,tp_rate,'b-');
grid on;
xlabel('False Positive Rate');
ylabel('Detection Rate');
% Area under the ROC curve value
AUC
%evaluation
%Evaluate(YValidation,YPred)
ACTUAL=actual_labels;
PREDICTED=predicted_labels';
idx = (ACTUAL()=='benign');
%disp(idx)
p = length(ACTUAL(idx));
n = length(ACTUAL(~idx));
N = p+n;
tp = sum(ACTUAL(idx)==PREDICTED(idx));
tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
fp = n-tn;
fn = p-tp;

tp_rate = tp/p;
tn_rate = tn/n;

accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));
gmean = sqrt(tp_rate*tn_rate);

disp(['accuracy=' num2str(accuracy)])
disp(['sensitivity=' num2str(sensitivity)])
disp(['specificity=' num2str(specificity)])
disp(['precision=' num2str(precision)])
disp(['recall=' num2str(recall)])
disp(['f_measure=' num2str(f_measure)])
disp(['gmean=' num2str(gmean)])
title={'''AUC','''accuracy','''sensitivity','''specificity','''precision','''recall','''f_measure','''gmean'};
VALUES=[AUC,accuracy,sensitivity,specificity,precision,recall,f_measure,gmean];
filename='performance.xlsx';
xlswrite(filename,title,'Sheet1','A1')
xlswrite(filename,VALUES,'Sheet1')
winopen(filename);