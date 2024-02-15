function [label, I] = Search(red, netTransfer, Img, size)
    filesPath = fullfile('..\ModelosEntrenados\', red);
    netTransfer = fullfile(filesPath, netTransfer);
    load(netTransfer);
    I = imread(Img);
    I = imresize(I, [size size]);

    [YValidationPred] = classify(netTransfer,I);


    label = YValidationPred(1);

    
end
