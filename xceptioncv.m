clc;close all;clear;%delete(findall(0));
%d = gpuDevice
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
figure;
for idx=1:length(perm)
    
    subplot(2,3,idx);
    imshow(imread(imds.Files{perm(idx)}));
    title(sprintf('%s',imds.Labels(perm(idx))))
    
end
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
 
    % xception Architecture 
    net=xception;
    lgraph = layerGraph(net);
    clear net;
    
    % Number of categories
    numClasses = numel(categories(imdsTrain.Labels));
    
    % Replacing the last layers with new layers

newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,'predictions',newFCLayer);
newClassLayer = softmaxLayer('Name','new_softmax');
lgraph = replaceLayer(lgraph,'predictions_softmax',newClassLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassLayer);
    
    % Preprocessing Technique
    %imdsTrain.ReadFcn = @(filename)preprocess_Xray(filename);
    %imdsTest.ReadFcn = @(filename)preprocess_Xray(filename);
    %rmsprop,sgdm,adam
    % Training Options, we choose a small mini-batch size due to limited images 
    options = trainingOptions('rmsprop',...
        'MaxEpochs',20,'MiniBatchSize',8,...
        'Shuffle','every-epoch', ...
        'InitialLearnRate',1e-4, ...
        'Verbose',false, ...
        'Plots','training-progress');
    
    % Data Augumentation
    augmenter = imageDataAugmenter( ...
        'RandRotation',[-5 5],'RandXReflection',1,...
        'RandYReflection',1,'RandXShear',[-0.05 0.05],'RandYShear',[-0.05 0.05]);
    
    % Resizing all training images to [229 229] for ResNet architecture
    auimds = augmentedImageDatastore([299 299],imdsTrain,'DataAugmentation',augmenter);
    
    % Training
    netTransfer = trainNetwork(auimds,lgraph,options);
    
    % Resizing all testing images to [299 299] for ResNet architecture   
    augtestimds = augmentedImageDatastore([299 299],imdsTest);
   
    % Testing and their corresponding Labels and Posterior for each Case
    [predicted_labels(test_idx),posterior(test_idx,:)] = classify(netTransfer,augtestimds);
    
    % Save the Independent ResNet Architectures obtained for each Fold
    save(sprintf('xception_%d_among_%d_folds',fold_idx,num_folds),'netTransfer','test_idx','train_idx');
    
    % Clearing unnecessary variables 
    clearvars -except fold_idx num_folds num_images predicted_labels posterior imds netTransfer;
    
end
analyzeNetwork(netTransfer)
%%Performance Study
% Actual Labels
actual_labels=imds.Labels;

% Confusion Matrix
figure;
plotconfusion(actual_labels,predicted_labels')
title('Confusion Matrix: xception');
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
    
