
function [bboxes, confidences, image_ids] = run_detector_nnet(test_scn_path, w, b, feature_params)

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));
threshold = 0.5;

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
    this_feat = vl_hog(img,feature_params.hog_cell_size);
    
    %fun = @(x) w'*x+b;
    %B = colfilt(this_feat,[feature_params.hog_cell_size, feature_params.hog_cell_size, 31],'distinct',fun);
    
    for j = 1:size(this_feat,1)-feature_params.hog_cell_size
        for k = 1:size(this_feat,2)-feature_params.hog_cell_size
            this_patch = this_feat( j:j+5 , k:k+5 , : );
            this_conf = w'*(reshape(this_patch,1,[]))' + b;
            if (this_conf >= threshold)
                cur_x_min = [cur_x_min; ceil(((k-1)*6 + 1)/sc_it)];
                cur_y_min = [cur_y_min; ceil(((j-1)*6 + 1)/sc_it)];
                cur_x_max = [cur_x_max; ceil(((k-1)*6 + 1 + 35)/sc_it)];
                cur_y_max = [cur_y_max; ceil(((j-1)*6 + 1 + 35)/sc_it)];
                cur_confidences = [cur_confidences; this_conf];                        
            end
        end
    end
  end 
    
    cur_bboxes = [cur_x_min, cur_y_min, cur_x_max, cur_y_max];
    cur_image_ids(1:numel(cur_x_min),1) = {test_scenes(i).name};
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(orig_img));

    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);
 
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
  
    
end
    
   to_file = [image_ids, num2cell(bboxes)]; 
   cell2csv('bboxes_for_nnet.csv', to_file);
   
end
    
   
    
    
    




