function Iout = preprocess_Xray(filename)
% This function preprocesses the given X-ray image by converting it into
% grayscale if required and later converting to 3-channel image to adapt to
% existing deep learning architectures 
%
% Author: Barath Narayanan
% Date: 3/17/2020

% Read the Filename
I = imread(filename);

% Some images might be RGB, convert them to Grayscale
if ~ismatrix(I)
    I=rgb2gray(I); 
end

% Replicate the image 3 times to create an RGB image
Iout = cat(3,I,I,I);

end