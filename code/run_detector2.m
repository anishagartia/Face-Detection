% Starter code prepared by James Hays for CS 4476, Georgia Tech
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confidences, image_ids] = run_detector2(test_scn_path, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (default 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression. Err
% on the side of having a low confidence threshold (even less than zero) to
% achieve high enough recall.

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));
threshold = -0.5;

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

for i = 1:length(test_scenes)     
  fprintf('Detecting faces in %s\n', test_scenes(i).name)
  img = imread( fullfile( test_scn_path, test_scenes(i).name ));
  img = single(img)/255;
  if(size(img,3) > 1)
    img = rgb2gray(img);
  end
  orig_img = img;
  scl = [1,0.9,0.7,0.5,0.3,0.1];
  
  cur_x_min = zeros(0,1); 
  cur_y_min = zeros(0,1);
  cur_x_max = zeros(0,1); 
  cur_y_max = zeros(0,1);
  cur_confidences = zeros(0,1);
  cur_bboxes = zeros(0,4);
    
  for sc_it = scl
    img = imresize(orig_img,sc_it);
    this_feat = get_hog_feat(img,feature_params.hog_cell_size);
    
    %fun = @(x) w'*x+b;
    %B = colfilt(this_feat,[feature_params.hog_cell_size, feature_params.hog_cell_size, 31],'distinct',fun);
    
    for j = 1:feature_params.hog_cell_size:size(this_feat,1)-feature_params.hog_cell_size
        for k = 1:feature_params.hog_cell_size*9:size(this_feat,2)-feature_params.hog_cell_size*9
            this_patch = this_feat( j:j+5 , k:k+53 , : );
%     for j = 1:feature_params.template_size:size(img,1)-feature_params.template_size
%         for k = 1:feature_params.template_size:size(img,2)-feature_params.template_size
%             this_patch = img( j:j+35 , k:k+35 );
%             this_feat = get_hog_feat(this_patch,feature_params.hog_cell_size);
            this_conf = w'*(reshape(this_patch,1,[]))' + b;
            if (this_conf >= threshold)
                cur_x_min = [cur_x_min; ceil(((k-1)*(36/54) + 1)/sc_it)];
                cur_y_min = [cur_y_min; ceil(((j-1)*6 + 1)/sc_it)];
                cur_x_max = [cur_x_max; ceil(((k-1)*(36/54) + 1 + 35)/sc_it)];
                cur_y_max = [cur_y_max; ceil(((j-1)*6 + 1 + 35)/sc_it)];
                cur_confidences = [cur_confidences; this_conf];                        
            end
        end
    end
  end 
    
    cur_bboxes = [cur_x_min, cur_y_min, cur_x_max, cur_y_max];
    cur_image_ids(1:numel(cur_x_min),1) = {test_scenes(i).name};
   
    %You can delete all of this below.
    % Let's create 15 random detections per image
    %cur_x_min = rand(15,1) * size(img,2);
    %cur_y_min = rand(15,1) * size(img,1);
    %cur_bboxes = [cur_x_min, cur_y_min, cur_x_min + rand(15,1) * 50, cur_y_min + rand(15,1) * 50];
    %cur_confidences = rand(15,1) * 4 - 2; %confidences in the range [-2 2]
    %cur_image_ids(1:15,1) = {test_scenes(i).name};
    
    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(orig_img));

    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);
 
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
  
    
end
end
    
   
    
    
    




