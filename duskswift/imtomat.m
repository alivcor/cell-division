% files = dir('*.png');
% imgs = {};
% for file = files'
%     img = imread(file.name);
%     imgs = [imgs, img];
% end

% imgPath = 'cellprofoutput/';
% imgType = '*.png'; % change based on image type
% images  = dir([imgPath imgType]);
% N = length(images);
% 
% % check images
% if( ~exist(imgPath, 'dir') || N<1 )
%     display('Directory not found or no matching images found.');
% end
% 
% % preallocate cell
% Seq{N,1} = []
% 
% for idx = 1:N
%     Seq{idx} = imread([imgPath images(idx).name]);
% end

