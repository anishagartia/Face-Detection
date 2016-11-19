
function [bboxes, confidences, image_ids] = run_detector_crop_lfw(lfw_file_paths, w, b, feature_params)

threshold = -0.01;

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

lfw_folder_path = '../data/lfw';
lfw_folder_crop_path = '../data/lfw_crop';
mkdir(lfw_folder_crop_path);

for i = 1:length(lfw_file_paths)     
  fprintf('Detecting faces in %d\n', i )
  img = imread(fullfile(lfw_folder_path,lfw_file_paths{i,:}));
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
    cur_image_ids(1:numel(cur_x_min),1) = {lfw_file_paths{i,:}};
   
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size( img )); % orig_img));

    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);
 
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
  

    diffs = bboxes(:,3) - bboxes(:,1);
    [mag,ind] = max(diffs);
    extr = 30;
    
    if abs(bboxes(ind,3) - (mag/2) - size(orig_img,2)/2) < 20        
        %new_img = imcrop(orig_img, [bboxes(ind,1), bboxes(ind,2), mag,mag]);
        new_img = imcrop(orig_img, [bboxes(ind,1)-extr, bboxes(ind,2), mag+extr*2,mag+extr]);
        tempname = strsplit(lfw_file_paths{i,:},'/');
        new_img = imresize(new_img, [feature_params.template_size feature_params.template_size]);
        imwrite(new_img, fullfile(lfw_folder_crop_path,tempname{2}));
        %imshow(new_img);
    end
  
    
end
end
    
   
    
    
    




