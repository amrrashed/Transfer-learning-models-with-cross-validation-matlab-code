 %% Try to classify something else
 clear;clc
 %format bank
 load covidnet79images1.mat trainedNet lgraph
%img = readimage(imds,100);
[filerootd, pathname1, filterindex1] = uigetfile({'*.jpg';'*.png';'*.jpeg'}, ...
   'Select an image');
x=imresize(imread([pathname1, filerootd]),[224 224]);
% x=imresize(imread('G:\covid project\matlabdb80\bacteria\person1_bacteria_1.jpeg'),[224 224]);
%x=imresize(imread('G:\covid project\matlabdb80\covid\01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg'),[224 224]);
[a,b,c]=size(x);
if c==1
  img=cat(3,x,x,x);
else
    img=x;
end
%actualLabel = imds.Labels(100);
actualLabel='--';
%predictedLabel = trainedNet.classify(img);
%YPred=classify(trainedNet,img);
%[YPred,scores] = classify(trainedNet,img)
[YPred,scores] =trainedNet.classify(img);
%YPred = predict(trainedNet,img);
switch(YPred)
    case 'bacteria'
       score=scores(1); 
    case 'covid'
        score=scores(2);
     case 'normal'
      score=scores(3);
end
imshow(img);
title(['Predicted: ' char(YPred) mat2str(floor(score*100)) '%',' Actual: ' char(actualLabel)])



