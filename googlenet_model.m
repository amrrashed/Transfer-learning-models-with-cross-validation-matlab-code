clc;clear;%delete(findall(0))
%alex net,resnet,darknet importer,google net,cnn,tensorflow and keras models
%deepNetworkDesigner
digitDatasetPath = fullfile('D:\CIT project\datasets\Dataset_BUSI\old_with_BUSI');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdstest] = splitEachLabel(imds,0.8,'randomized');
numClasses = numel(categories(imdsTrain.Labels));
 % alexNet Architecture 
    net=vgg16;
    
     % Replacing the last layers with new layers
    layersTransfer = net.Layers(1:end-3);
     inputSize=[224,224,3];%100 100
     %90.2 end-4,3 layer
     %93.69 end-3,6 layer
layers = [
     layersTransfer
      fullyConnectedLayer(500)
      reluLayer
       dropoutLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
%inputSize = net.Layers(1).InputSize;
% inputSize= [224 224];
 imdsTrain = augmentedImageDatastore(inputSize,imdsTrain);
 imdstest1 = augmentedImageDatastore(inputSize,imdstest);
%learning functions sgdm,rmsprop,adam
%VGG16                    VGG19 
%ADAM 91.26%               ADAM 87.86%
%rmsprop 92.72 %           RMSPROP 89.81
%sgdm 88.83%               SGDM  89.32
options = trainingOptions('rmsprop',...
        'ExecutionEnvironment','gpu',...
        'MaxEpochs',20,'MiniBatchSize',8,...%20,8
        'Shuffle','every-epoch', ...
        'InitialLearnRate',1e-5, ...
        'Verbose',false, ...
        'Plots','training-progress');
trainedNet = trainNetwork(imdsTrain,layers,options);

[YPred,probs] = classify(trainedNet,imdstest,'ExecutionEnvironment','cpu');
YValidation = imdstest.Labels;
cf  = confusionmat(YValidation, YPred)
accuracy = sum(YPred == YValidation)/numel(YValidation)%78%
%accuracy = mean(YPred == imdsValidation.Labels)
%%save Network
%save simpleDL.mat trainedNet lgraph
%% Try to classify something else
% img = readimage(imds,100);
% actualLabel = imds.Labels(100);
% predictedLabel = trainedNet.classify(img);
% imshow(img);
% title(['Predicted: ' char(predictedLabel) ', Actual: ' char(actualLabel)])
