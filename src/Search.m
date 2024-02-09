function Search(red, netTransfer, Img, size)
    filesPath = fullfile('..\ModelosEntrenados\', red);
    netTransfer = fullfile(filesPath, netTransfer);
    load(netTransfer);
    I = imread(Img);
    I = imresize(I, [size size]);

    [YValidationPred,probs] = classify(netTransfer,I);
z
    figure
    imshow(I)
    label = YValidationPred(1);
    title(string(label) + ", " + num2str(100*max(probs(1, :)),3) + "%");
    
end