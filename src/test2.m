NewDir = fullfile('..\ModelosEntrenados\',red);

if exist(NewDir, 'dir') ~= 7
    mkdir(NewDir);
end
S1 = '\netTransferMonumentos';
S2 = red;
fichero = [NewDir,S1,S2];
fichero = convertCharsToStrings(fichero);
disp(fichero)