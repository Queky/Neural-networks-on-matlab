close all
clear all
clc

net = googlenet;

inputSize = net.Layers(1).InputSize

classNames = net.Layers(end).ClassNames;
numClasses = numel(classNames);
disp(classNames(randperm(numClasses,10)))

I = imread('./images/ClassifyImageUsingGoogLeNetExample_01.png');
I = imresize(I,inputSize(1:2));
[label,scores] = classify(net,I);
label

figure
imshow(I)
title(string(label) + ", " + num2str(100*scores(classNames == label),3) + "%");

% Top predictions
[~,idx] = sort(scores,'descend');
idx = idx(5:-1:1);
classNamesTop = net.Layers(end).ClassNames(idx);
scoresTop = scores(idx);

figure
barh(scoresTop)
xlim([0 1])
title('Top 5 Predictions')
xlabel('Probability')
yticklabels(classNamesTop)