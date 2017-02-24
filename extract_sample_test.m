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

cell_num = 5
xv = x(cell_num, 1) - 32;
yv = y(cell_num, 1) - 32;
x1 = max(xv,0)
y1 = max(yv,0)
class_1 = imcrop(nframe,[x1, y1, 63, 63]);
imshow(class_1);



