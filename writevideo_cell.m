
nvideo = squeeze(cDIC(:,:,1,:));
nvideo = nvideo/max(max(max((nvideo))));
imshow(nvideo(:,:,5)*10);

v = VideoWriter('cell_division.avi');
open(v);
for i=1:241
   img = histeq(im2single(nvideo(:,:,i))/max(max(max((nvideo)))));
   writeVideo(v,img);
end

close(v);