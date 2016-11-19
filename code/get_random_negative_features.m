% Starter code prepared by James Hays for CS 4476, Georgia Tech
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (default 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);

D = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
features_neg = zeros(num_samples,D);
num_of_patchs = ceil(num_samples/num_images);
fprintf('Getting negative features: image       ');
k = 0;
for i = 1:num_images
    
    fprintf('\b\b\b\b\b\b%6.0f',i);
    this_img = rgb2gray(imread(fullfile(non_face_scn_path,char(image_files(i).name))));
    
    rnd_1 = randi([1,size(this_img,1)-feature_params.template_size],num_of_patchs,1);%choose num_of_patchs random unique points on 1st dimension
    rnd_2 = randi([1,size(this_img,2)-feature_params.template_size],num_of_patchs,1);%choose num_of_patchs random unique points on 2nd dimension
    for j = 1:num_of_patchs
       k = k+1;
       %this_patch = this_img((rnd_1(j):(rnd_1(j)+feature_params.template_size-1)),(rnd_2(j):(rnd_2(j)+feature_params.template_size-1)));%Convert chosen numbers to image pieces
       this_patch = imcrop(this_img,[rnd_2(j) rnd_1(j) (feature_params.template_size-1) (feature_params.template_size-1)]);
       this_feat =  vl_hog(im2single(this_patch), feature_params.hog_cell_size);
       features_neg(k,:) = im2double(reshape(this_feat,1,D));
    end            
end

fprintf('\n');
end

% placeholder to be deleted. 100 random features.
%features_neg = rand(100, (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);