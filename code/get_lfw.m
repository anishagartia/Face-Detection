function [file_names, features_lfw] = get_lfw(feature_params)
    fileID = fopen('../data/lfw/lfw-names.txt');
    Y = textscan(fileID,'%s\t%d'); 
    names = Y{:,1};
    times = Y{:,2};
    Z = {};
    
    lfw_folder_path = '../data/lfw';
    for i = 1:size(names,1)
        for j = 1:times(i,1)
            temp_name = strcat(names{i,1},'_',num2str(j,'%04i'),'.jpg');
            temp_name = fullfile(names{i,1},temp_name);
            Z = vertcat(Z(:,:), temp_name);
        end
    end
    file_names = Z;
    
    %Get HOG features
    num_images = size(file_names,1);
    D = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
    features_lfw = zeros(num_images,D);
    fprintf('Getting LFW features: image       ');
    
%     num_images = size(file_names,1);
%     for i = 1:num_images
%         fprintf('\b\b\b\b\b\b%6.0f',i);
%         this_feat = vl_hog(im2single(rgb2gray(imread(fullfile(lfw_folder_path, file_names{i,:})))), feature_params.hog_cell_size);
%         features_lfw(i,:) = im2double(reshape(this_feat,1,D));
%     %Error for reshape here
%       end
features_lfw = 0;
%         
end

