%% Redimensionado de las imágenes
% Dimensión: 224x224x3: GoogleNet, VGG16, VGG19, ResNet18, ResNet50, ResNet101, densenet201
% Dimensión: 227x227x3: AlexNet, squeezenet
% Dimensión: 299x299x3: inceptionresnetv2, inceptionv3


%cd 'C:\TFG'
imds = imageDatastore('..\Monumentos',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

idx = size(imds.Files,1);

for j=1:1:3
    if j == 1
        maxHeight = 299;
        maxWidth = 299;
        S2 = 'DATASET299x299';
    elseif j == 2
        maxHeight = 227;
        maxWidth = 227;
        S2 = 'DATASET227x227';
    elseif j == 3
        maxHeight = 224;
        maxWidth = 224;
        S2 = 'DATASET224x224';
    end
    for i=1:1:idx
      D = cell2mat(imds.Files(i));
      Img = imread(D);
    
      I = imresize(Img, [maxHeight maxWidth]);
    
      [a,b] = find(D =='\');
    
      S1 = D(1:b(numel(b)-2));
      %S2 = 'DATASET224x224';
      %S2 = 'DATASET227x227';
      S3 = D(b(numel(b)-1):size(D,2));
      [pathstr, name, ext] = fileparts(S3);
      NewDir = [S1, S2, pathstr];
      if exist(NewDir, 'dir') ~= 7
        mkdir(NewDir);
      end
      fichero = [S1,S2,S3];
      imwrite(I,fichero);
    end
end