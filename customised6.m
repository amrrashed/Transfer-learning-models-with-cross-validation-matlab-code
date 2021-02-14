clc;clear;%delete(findall(0))
digitDatasetPath = fullfile('D:\CIT project\datasets\Dataset_BUSI\old_with_BUSI');
 imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

[imds,imdstest] = splitEachLabel(imds,0.8,'randomized');
inputSize=[100,100,3];
imdsTrain = augmentedImageDatastore(inputSize,imds);
imdstest1 = augmentedImageDatastore(inputSize,imdstest);
%100,100 79%
%150 150 78%
%200 200 77%
numClasses = 3;
layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(5,20)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
% % Data Augumentation
%      augmenter = imageDataAugmenter('RandRotation', [-20 20],'RandXShear',[-0.05 0.05],'RandYShear',[-0.05 0.05],...
%      'RandScale',[0.5 1],'RandXTranslation',[-3 3], ...
%      'RandYTranslation',[-3 3]);
% augmenter = imageDataAugmenter('RandXReflection',1,...
%         'RandYReflection',1);
% % %  % Resizing all training images to [227 227] for ResNet architecture
%    imds = augmentedImageDatastore([224 224],imds,'DataAugmentation',augmenter);
%    imdsValidation1 = augmentedImageDatastore([224 224],imdsValidation,'DataAugmentation',augmenter);
options = trainingOptions('rmsprop', ...
    'ExecutionEnvironment','auto',...
    'MaxEpochs',20,'MiniBatchSize',8,...
    'Shuffle','every-epoch', ...
    'InitialLearnRate',1e-5, ...
    'Plots','training-progress',...
    'Verbose',false);
%   'ValidationData',imdsValidation1, ...
%     'ValidationFrequency',30, ...
trainedNet = trainNetwork(imdsTrain,layers,options);

[YPred,probs] = classify(trainedNet,imdstest1,'ExecutionEnvironment','cpu');
YValidation = imdstest.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation)%90.78