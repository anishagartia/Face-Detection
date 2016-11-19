
function [features_neg] = hard_mining(non_face_scn_path, w, b, feature_params, threshold)

image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);
hog_feats = zeros(0,6*6*31);

for i = 1:length(image_files)     
  fprintf('Detecting faces in %s\n', image_files(i).name)
  img = imread( fullfile( non_face_scn_path, image_files(i).name ));
  img = single(img)/255;
  if(size(img,3) > 1)
    img = rgb2gray(img);
  end
  orig_img = img;
  scl = [1,0.9,0.7, 0.5, 0.3, 0.1];
  
  block_size = feature_params.template_size/feature_params.hog_cell_size;
  
  cur_x_min = zeros(0,1); 
  cur_y_min = zeros(0,1);
  cur_x_max = zeros(0,1); 
  cur_y_max = zeros(0,1);
  D = block_size * block_size * 31;
  cur_hog_feat = zeros(0,D);
  cur_confidences = zeros(0,1);
  cur_bboxes = zeros(0,4);
    
  for sc_it = scl
    img = imresize(orig_img,sc_it);
    this_feat = vl_hog(img,feature_params.hog_cell_size);
    
    %fun = @(x) w'*x+b;
    %B = colfilt(this_feat,[feature_params.hog_cell_size, feature_params.hog_cell_size, 31],'distinct',fun);
=   
    for j = 1:size(this_feat,1)-block_size
        for k = 1:size(this_feat,2)-block_size
            this_patch = this_feat( j:j+block_size-1 , k:k+block_size-1 , : );
            this_conf = w'*(reshape(this_patch,1,[]))' + b;
            if (this_conf >= threshold)
                cur_x_min = [cur_x_min; ceil(((k-1)*feature_params.hog_cell_size + 1)/sc_it)];
                cur_y_min = [cur_y_min; ceil(((j-1)*feature_params.hog_cell_size + 1)/sc_it)];
                cur_x_max = [cur_x_max; ceil(((k-1)*feature_params.hog_cell_size + 1 + feature_params.template_size)/sc_it)];
                cur_y_max = [cur_y_max; ceil(((j-1)*feature_params.hog_cell_size + 1 + feature_params.template_size)/sc_it)];
                cur_hog_feat = [cur_hog_feat; im2double(reshape(this_patch,1,D))];
                cur_confidences = [cur_confidences; this_conf];                        
            end
        end
    end
  end 
    
    cur_bboxes = [cur_x_min, cur_y_min, cur_x_max, cur_y_max];
    cur_image_ids(1:numel(cur_x_min),1) = {image_files(i).name};
   
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

    cur_confidences = cur_confidences(  is_maximum,:);
    cur_bboxes      = cur_bboxes(       is_maximum,:);
    cur_image_ids   = cur_image_ids(    is_maximum,:);
    cur_hog_feat    = cur_hog_feat(     is_maximum,:);
 
    bboxes      = [bboxes;      cur_bboxes];
    hog_feats   = [hog_feats;   cur_hog_feat];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
  
    
end

features_neg = hog_feats;


end
    
   
    
    
    




