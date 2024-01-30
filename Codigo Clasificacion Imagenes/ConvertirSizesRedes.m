
%% Redimensionado de las imágenes
% Dimensión: 224x224x3: GoogleNet, VGG16, VGG19, ResNet18, ResNet50, ResNet101, densenet201  
% Dimensión: 227x227x3: AlexNet, squeezenet
% Dimensión: 299x299x3: inceptionresnetv2, inceptionv3


%cd 'C:\TFG\'
imds = imageDatastore('..\Monumentos',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

idx = size(imds.Files,1);

for i=1:1:idx
  D = cell2mat(imds.Files(i));
  Img = imread(D);
  
  I = imresize(Img, [299 299]);
  
  [a,b] = find(D =='\');

  S1 = D(1:b(numel(b)-2));
  %S2 = 'DATASET224x224';
  %S2 = 'DATASET227x227';
  S2 = 'DATASET299x299';
  S3 = D(b(numel(b)-1):size(D,2));
  [pathstr, name, ext] = fileparts(S3);
  NewDir = [S1, S2, pathstr];
  if exist(NewDir, 'dir') ~= 7
    mkdir(NewDir);
  end
  fichero = [S1,S2,S3];
  imwrite(I,fichero);
end
