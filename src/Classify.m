function Classify(red, netTransfer, imdsValidation, augimdsValidation, imdsTrain, augimdsTrain)
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
end