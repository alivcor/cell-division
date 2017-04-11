imgPath = 'imp-crf/dataset/test/x/';
imgType = '*.png';
images  = dir([imgPath imgType]);
N = length(images);

if( ~exist(imgPath, 'dir') || N<1 )
    display('Directory not found or no matching images found.');
end

Seq{N,1} = [];

for idx = 1:N
    I = imread([imgPath images(idx).name]);
    I = I/max(max(I));
    I = histeq(I);
    I = imgaussfilt(I,2);
    BWs = edge(I, 'Canny',0.32);
    se90 = strel('disk', 2, 4);
    se0 = strel('disk', 2, 0);
    BWsdil = imdilate(BWs, [se90 se0]);
    BWdfill = imfill(BWsdil, 'holes');
    seD = strel('disk',3);
    BWfinal = imerode(BWdfill,seD);
    BWfinal = imerode(BWfinal,seD);
    BWfinal = bwareaopen(BWfinal,80);
    Seq{idx} = BWfinal;
    fname = strcat('erosion_results/', images(idx).name);
    imwrite(BWfinal, fname);
end

for idx = 1:266
    im = Seq{idx,1};
    Seq{idx,1} = strrep(Seq{idx,2},'.png','');
    Seq{idx,2} = im;
end

