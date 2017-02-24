% class_0 : NON-ROIs (BLANK SAMPLES)
% class_1 : ROIs (CELL SAMPLES)
frame_num = 5;
camera_num = 1;
nframe = squeeze(c488(:,:,camera_num,frame_num));
nframe = nframe/max(max(max((nframe))));
% imshow(nvideo(:,:,5));
y = squeeze(data(:,frame_num,3));
x = squeeze(data(:,frame_num,4));
imshow(nframe(:,:));
hold on;
plot(x,y,'r.','Color',[1 0 0],'MarkerSize',20);
hold off;

mkdir('class_1');
%class 1
for cell_num=1:49
    fname = strcat('class_1/im_0_C',num2str(camera_num),'_F',num2str(frame_num),'_T',num2str(cell_num),'.png')
    xv = x(cell_num, 1) - 32;
    yv = y(cell_num, 1) - 32;
    x1 = max(xv,0);
    y1 = max(yv,0);
    class_1 = imcrop(nframe,[x1, y1, 63, 63]);
    imwrite(class_1,fname);
    % imshow(class_1);
end

%class_0
mkdir('class_0');

random_points = 448.*rand(100,2);
D = pdist2(random_points,[x y]);

id_num = 0;
for i=1:100
    useful = 1;
    for j=1:49
        if(D(i,j)<50)
            useful = 0;
            break
        end
    end
    if(useful == 1)
        id_num = id_num + 1;
        fname = strcat('class_0/im_0_C',num2str(camera_num),'_F',num2str(frame_num),'_ID',num2str(id_num),'.png')
        xv = random_points(i, 1) - 32;
        yv = random_points(i, 2) - 32;
        x1 = max(xv,0);
        y1 = max(yv,0);
        class_0 = imcrop(nframe,[x1, y1, 63, 63]);
        imwrite(class_0,fname);
    end
end

strcat('Number of Useful Samples Found : ', num2str(id_num))



