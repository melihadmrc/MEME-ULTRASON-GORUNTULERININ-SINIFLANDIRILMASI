
OrgfolderPath = 'C:\Users\İREM\Desktop\veriarttirimi\normal\'; 
% Klasördeki tüm dosyalar alınır.
fileList = dir(fullfile(OrgfolderPath, '*.png'));

% Veri artırma için ImageDataAugmenter nesnesini oluşturulur.
augmenter = imageDataAugmenter( ...
    'RandRotation', [-10, 10], ...
    'RandXTranslation', [-20, 20], ...
    'RandYTranslation', [-20, 20], ...
    'RandXScale', [0.8, 1.2], ...
    'RandYScale', [0.8, 1.2], ...
    'RandXShear', [-10, 10], ...
    'RandYShear', [-10, 10] ...
);

% Her bir dosya üzerinde döngü
for i = 1:length(fileList)
    % Dosya yolu ve adıyla birleştirilir.
    imagePath = fullfile(OrgfolderPath, fileList(i).name);

    originalImage = imread(imagePath);

    % Veri artırma işlemini uygulanır.
    augmentedImage = augment(augmenter, originalImage);

    % Artırılmış görüntüyü kaydedilir.
    augmentedImagePath = fullfile(OrgfolderPath, ['1-' fileList(i).name]);
    imwrite(augmentedImage, augmentedImagePath);
end




