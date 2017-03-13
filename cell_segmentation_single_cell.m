%% Detecting a Cell Using Image Segmentation
% This example shows how to detect a cell using edge detection and basic
% morphology.  An object can be easily detected in an image if the object has
% sufficient contrast from the background. In this example, the cells are
% prostate cancer cells.
%
% Copyright 2004-2013 The MathWorks, Inc.


close all;
%% Step 1: Read Image
% Read in the |cell.tif| image, which is an image of a prostate cancer
% cell.

%I = imread('cell.tif');


%I = squeeze(trX(1,:,:));

I = squeeze(cDIC(:,:,1,10)); % was 5,20

I = I/max(max(I));

I = histeq(I);

I = imgaussfilt(I,2);

%I = wiener2(I,[5 5]);

% trick one: histeq


% trick two: convolving with a gaussian filter


%figure, imshow(I), title('original image');

%% Step 2: Detect Entire Cell
% Two cells are present in this image, but only one cell can be seen in its
% entirety. We will detect this cell. Another word for object detection is
% segmentation. The object to be segmented differs greatly in contrast from
% the background image. Changes in contrast can be detected by operators that
% calculate the gradient of an image.  The gradient image can be calculated
% and a threshold can be applied to create a binary mask containing the segmented
% cell.  First, we use |edge| and the Sobel operator to calculate the threshold
% value. We then tune the threshold value and use |edge| again to obtain a
% binary mask that contains the segmented cell.

BWs = edge(I, 'Canny',0.32); % Canny, 0.32 was the best.
%fudgeFactor = 1.2; % was .5
%BWs = edge(I,'sobel', threshold * fudgeFactor);
figure, imshow(BWs), title('binary gradient mask');

%% Step 3: Dilate the Image
% The binary gradient mask shows lines of high contrast in the image. These
% lines do not quite delineate the outline of the object of interest.
% Compared to the original image, you can see gaps in the lines surrounding
% the object in the gradient mask. These linear gaps will disappear if the
% Sobel image is dilated using linear structuring elements, which we can
% create with the |strel| function.

se90 = strel('disk', 2, 4); % was line,3,90
se0 = strel('disk', 2, 0); % was line,3,0

%%
% The binary gradient mask is dilated using the vertical structuring
% element followed by the horizontal structuring element. The |imdilate|
% function dilates the image.

BWsdil = imdilate(BWs, [se90 se0]);
figure, imshow(BWsdil), title('dilated gradient mask');

%% Step 4: Fill Interior Gaps 
% The dilated gradient mask shows the outline of the cell quite nicely, but
% there are still holes in the interior of the cell. To fill these holes we
% use the imfill function.

BWdfill = imfill(BWsdil, 'holes');
%figure, imshow(BWdfill);
%title('binary image with filled holes');

%% Step 5: Remove Connected Objects on Border
% The cell of interest has been successfully segmented, but it is not the
% only object that has been found. Any objects that are connected to the
% border of the image can be removed using the imclearborder function. The
% connectivity in the imclearborder function was set to 4 to remove
% diagonal connections.

%BWnobord = imclearborder(BWdfill, 4);
%figure, imshow(BWnobord), title('cleared border image');

%% Step 6: Smoothen the Object
% Finally, in order to make the segmented object look natural, we smoothen
% the object by eroding the image twice with a diamond structuring element.
% We create the diamond structuring element using the |strel| function.

seD = strel('disk',3); % was 1
BWfinal = imerode(BWdfill,seD);
BWfinal = imerode(BWfinal,seD);

BWfinal = bwareaopen(BWfinal,80);



figure, imshow(BWfinal), title('segmented image');


%% splitting the cells

% improving the results by splitting of cells
%splitth = 9;
%plane = 1;
% cells above this threshold are split (all cells here)
%n = prmout.minvolvox;
%h = [0.01 0.01 0.01];   % was h = [0.5 0.5 1.5];
%BWfinal = cellsegm.splitcells(BWfinal,splitth,n,h);
%cellsegm.show(BWfinal,2);title('Cell segmentation by ADTH with splitting');axis off;





%%
% An alternate method for displaying the segmented object would be to place
% an outline around the segmented cell. The outline is created by the
% |bwperim| function.

BWoutline = bwperim(BWfinal);
Segout = I; 
Segout(BWoutline) = 255; 
figure, imshow(Segout), title('outlined original image');






%displayEndOfDemoMessage(mfilename)
