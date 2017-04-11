dirName_x = 'dcore/train/x/';
dirName_y = 'dcore/train/y/';
rindices = randi([1 6000],1,700);
for i=1:size(rindices,2)
    index_num = rindices(i);
    fname_x = strcat(dirName_x, num2str(index_num),'.png');
    img_x = mat2gray(squeeze(trX(index_num,:,:,1)));
    imwrite(img_x, fname_x);
    fname_y = strcat(dirName_y, num2str(index_num),'.png');
    img_y = squeeze(trY(index_num,:,:,1))./255;
    imwrite(img_y, fname_y);
end

dirName_x = 'dcore/test/x/';
dirName_y = 'dcore/test/y/';
rindices = randi([1 1200],1,300);
for i=1:size(rindices,2)
    index_num = rindices(i);
    fname_x = strcat(dirName_x, num2str(index_num),'.png');
    img_x = mat2gray(squeeze(teX(index_num,:,:,1)));
    imwrite(img_x, fname_x);
    fname_y = strcat(dirName_y, num2str(index_num),'.png');
    img_y = squeeze(teY(index_num,:,:,1))./255;
    imwrite(img_y, fname_y);
end
