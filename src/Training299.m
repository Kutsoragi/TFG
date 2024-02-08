function Training299(red, solver, batchSize,maxEpochs, initialLearnRate)

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
    
    valFrequency = floor(numel(augimdsTrain.Files)/batchSize);
    options = trainingOptions(solver, ...
        'MiniBatchSize',batchSize, ...
        'MaxEpochs',maxEpochs, ...
        'InitialLearnRate',initialLearnRate, ...
        'Shuffle','every-epoch', ...
        'ValidationData',augimdsValidation, ...
        'ValidationFrequency',valFrequency, ...
        'ValidationPatience',Inf, ...
        'Verbose',false, ...
        'Plots','training-progress');
    
    netTransfer = trainNetwork(augimdsTrain,lgraph,options);
    
    %% Guardar la red entrenada con el nombre del fichero apropiado
    NewDir = fullfile('..\ModelosEntrenados\',red);

    if exist(NewDir, 'dir') ~= 7
        mkdir(NewDir);
    end
    S1 = '\netTransferMonumentos';
    S2 = red;
    redPath = [NewDir,S1,S2];
    redPath = convertCharsToStrings(redPath);
    %imdsValidation, augimdsValidation, imdsTrain, augimdsTrain
    imdsValPath = [NewDir, '\imdsValidation'];
    augImdsValPath = [NewDir, '\augimdsValidation'];
    imdsTrainPath = [NewDir, '\imdsTrain'];
    augImdsTrainPath = [NewDir, '\augimdsTrain'];
    save (redPath, "netTransfer");
    save (imdsValPath, "imdsValidation");
    save (augImdsValPath, "augimdsValidation");
    save (imdsTrainPath, "imdsTrain");
    save (augImdsTrainPath, "augimdsTrain");
end

