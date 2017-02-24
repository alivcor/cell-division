% Generate data
% cDIC_without_histeq_full
% cDIC_without_histeq_individual
% cDIC_with_histeq_full
% cDIC_with_histeq_individual

dirName = 'cell_profiler_input/';
camera_num = 1;

subDirName = 'without_histeq_full/';
for frame_num = 1:50
    fname = strcat(dirName, subDirName, 'im_H0_F_C',num2str(camera_num),'_F',num2str(frame_num),'.png')
    cell_img = squeeze(cDIC(:,:,camera_num,frame_num));
    cell_img = cell_img/max(max(max((cell_img))));
%     imshow(cell_img);
    imwrite(cell_img, fname)
end


subDirName = 'without_histeq_individual/';
for frame_num = 1:50
    cell_img = squeeze(cDIC(:,:,camera_num,frame_num));
    cell_img = cell_img/max(max(max((cell_img))));
    y = squeeze(data(:,frame_num,3));
    x = squeeze(data(:,frame_num,4));
    for cell_num=1:49
        fname = strcat(dirName, subDirName, 'im_H0_I_C',num2str(camera_num),'_F',num2str(frame_num),'_T',num2str(cell_num),'.png')
        xv = x(cell_num, 1) - 32;
        yv = y(cell_num, 1) - 32;
        x1 = max([xv,0]);
        y1 = max([yv,0]);
        class_1 = imcrop(cell_img,[x1, y1, 63, 63]);
        imwrite(class_1,fname);
        % imshow(class_1);
    end
end


subDirName = 'with_histeq_full/';
for frame_num = 1:50
    fname = strcat(dirName, subDirName, 'im_H1_F_C',num2str(camera_num),'_F',num2str(frame_num),'.png')
    cell_img = squeeze(cDIC(:,:,camera_num,frame_num));
    cell_img = histeq(im2single(cell_img/max(max(max((cell_img))))));
%     imshow(cell_img);
    imwrite(cell_img, fname)
end


subDirName = 'with_histeq_individual/';
for frame_num = 1:50
    cell_img = squeeze(cDIC(:,:,camera_num,frame_num));
    cell_img = histeq(im2single(cell_img/max(max(max((cell_img))))));
    y = squeeze(data(:,frame_num,3));
    x = squeeze(data(:,frame_num,4));
    for cell_num=1:49
        fname = strcat(dirName, subDirName, 'im_H1_I_C',num2str(camera_num),'_F',num2str(frame_num),'_T',num2str(cell_num),'.png')
        xv = x(cell_num, 1) - 32;
        yv = y(cell_num, 1) - 32;
        x1 = max([xv,0]);
        y1 = max([yv,0]);
        class_1 = imcrop(cell_img,[x1, y1, 63, 63]);
        imwrite(class_1,fname);
        % imshow(class_1);
    end
end