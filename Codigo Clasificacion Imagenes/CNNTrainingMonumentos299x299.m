clear all; close all;

%% Training para inceptionresnetv2, inceptionv3

% https://es.mathworks.com/help/deeplearning/ug/train-deep-learning-network-to-classify-new-images.html

imds = imageDatastore('..\DATASET299x299',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

%% Redes con dimensiones de imagen 299x299x3: inceptionresnetv2, inceptionv3
%red ='inceptionresnetv2';
red ='inceptionv3';

switch red
    case 'inceptionresnetv2'
        net = inceptionresnetv2;
    case 'inceptionv3'
        net = inceptionv3;
end

net.Layers(1)
inputSize = net.Layers(1).InputSize;

analyzeNetwork(net)

if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

[learnableLayer,classLayer] = findLayersToReplace(lgraph);

numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

analyzeNetwork(lgraph)

pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

numaugsValidationImages = augimdsValidation.NumObservations;
idx = randperm(numaugsValidationImages,16);

figure
for i = 1:16
    subplot(4,4,i)
    I = imread(augimdsValidation.Files{idx(i),1});
    imshow(I)
end

miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,lgraph,options);

%% Guardar la red entrenada con el nombre del fichero apropiado
NewDir = '..\ModelosEntrenados\';
if exist(NewDir, 'dir') ~= 7
    mkdir(NewDir);
end
S1 = 'netTransferMonumentos';
S2 = red;
fichero = [NewDir,S1,S2];
fichero = convertCharsToStrings(fichero);
save (fichero, "netTransfer");

%% Validación del proceso de entrenamiento

[YValidationPred,probs] = classify(netTransfer,augimdsValidation);
validationAccuracy = mean(YValidationPred == imdsValidation.Labels);
validationError = mean(YValidationPred ~= imdsValidation.Labels);

YTrainPred = classify(netTransfer,augimdsTrain);
trainError = mean(YTrainPred ~= imdsTrain.Labels);
disp("Error Entrenamiento: " + trainError*100 + "%")
disp("Error Validacion: " + validationError*100 + "%")

%% Plot the confusion matrix. Display the precision and recall for each class by using column and row summaries. Sort the classes of the confusion matrix. The largest confusion is between unknown words and commands, up and off, down and no, and go and no.
figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
cm = confusionchart(YValidationPred,imdsValidation.Labels, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized');
A1 = 'Matriz Confusion Validacion: ';
A2 = red;
cm.Title = [A1,A2];
%sortClasses(cm)


idx = randperm(numel(imdsValidation.Files),8);
figure
for i = 1:8
    subplot(4,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YValidationPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end

%% Visualización de los pesos

% Get the network weights for the second convolutional layer
w1 = netTransfer.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5);

% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
figure
montage(w1)
title('Pesos primera capa convolucional')


