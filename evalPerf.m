img1 = imread('duskswift/cellprofoutput/1.png');
img2 = imread('imp-crf/dataset/test/y/1.png');

imgPath1 = 'duskswift/cellprofoutput/';
imgType = '*.png';
imgPath2 = 'imp-crf/dataset/test/y/';

images  = dir([imgPath1 imgType]);

N = length(images);

if( ~exist(imgPath1, 'dir') || N<1 )
    display('Directory not found or no matching images found.');
end

miou = 0
for idx = 1:N
   img_pred = imfill(imread([imgPath1 images(idx).name]));
   img_gt = imread([imgPath2 images(idx).name]);
   tp = 0;
   tn = 0;
   fp = 0;
   fn = 0;
   for i = 1:224
       for j = 1:224
           if(img_gt(i,j)==0)
              if(img_pred(i,j)==img_gt(i,j))
                  tp = tp + 1;
              else
                  fp = fp + 1;
              end
           elseif(img_gt(i,j)==255)
              if(img_pred(i,j)==img_gt(i,j))
                  tn = tn + 1;
              elseif(img_pred(i,j))
                  fn = fn + 1;
              end
           else
               a  = 'Value Corrupt'
           end
       end
   end
   iou = tp/(tp + fp + fn);
   miou = miou + iou
end

miou = miou/266
   
   
   
   
   
   
   
   
   
   