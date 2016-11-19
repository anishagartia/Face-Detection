
close all
clear
run('../vlfeat-0.9.20/toolbox/vl_setup')

[~,~,~] = mkdir('visualizations');

data_path = '../data/'; 
train_path_pos = fullfile(data_path, 'caltech_faces/Caltech_CropFaces'); %Positive training examples. 36x36 head crops
non_face_scn_path = fullfile(data_path, 'train_non_face_scenes'); %We can mine random or hard negatives from here
test_scn_path = fullfile(data_path,'test_scenes/test_jpg'); %CMU+MIT test scenes
% test_scn_path = fullfile(data_path,'extra_test_scenes'); %Bonus scenes
label_path = fullfile(data_path,'test_scenes/ground_truth_bboxes.txt'); %the ground truth face locations in the test set

%The faces are 36x36 pixels, which works fine as a template size. You could
%add other fields to this struct if you want to modify HoG default
%parameters such as the number of orientations, but that does not help
%performance in our limited test.
feature_params = struct('template_size', 36, 'hog_cell_size', 6);


%% Step 1. Load positive training crops and random negative examples
%YOU CODE 'get_positive_features' and 'get_random_negative_features'

features_pos = get_positive_features( train_path_pos, feature_params );
%features_pos = get_positive_features2( train_path_pos, feature_params );

num_negative_examples = 10000; %Higher will work strictly better, but you should start with 10000 for debugging
features_neg = get_random_negative_features( non_face_scn_path, feature_params, num_negative_examples);
%features_neg = get_random_negative_features2( non_face_scn_path, feature_params, num_negative_examples);

% LFW Dataset
%[lfw_file_paths, features_lfw] = get_lfw(feature_params);
    

%% step 2. Train Classifier
% Use vl_svmtrain on your training features to get a linear classifier
% specified by 'w' and 'b'
% [w b] = vl_svmtrain(X, Y, lambda) 
% http://www.vlfeat.org/sandbox/matlab/vl_svmtrain.html
% 'lambda' is an important parameter, try many values. Small values seem to
% work best e.g. 0.0001, but you can try other values
lambda = 0.006;
X = [features_pos ; features_neg];
Y = [ones(size(features_pos,1),1); -1*ones(size(features_neg,1),1)];
[w,b] = vl_svmtrain(X', Y, lambda); 

%YOU CODE classifier training. Make sure the outputs are 'w' and 'b'.
%w = rand((feature_params.template_size / feature_params.hog_cell_size)^2 * 31,1); %placeholder, delete
%b = rand(1); %placeholder, delete

% step 3. Examine learned classifier
% You don't need to modify anything in this section. The section first
% evaluates _training_ error, which isn't ultimately what we care about,
% but it is a good sanity check. Your training error should be very low.

fprintf('Initial classifier performance on train data:\n')
confidences = [features_pos; features_neg]*w + b;
label_vector = [ones(size(features_pos,1),1); -1*ones(size(features_neg,1),1)];
[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy( confidences, label_vector );

% Visualize how well separated the positive and negative examples are at
% training time. Sometimes this can idenfity odd biases in your training
% data, especially if you're trying hard negative mining. This
% visualization won't be very meaningful with the placeholder starter code.
non_face_confs = confidences( label_vector < 0);
face_confs     = confidences( label_vector > 0);
figure(2); 
plot(sort(face_confs), 'g'); hold on
plot(sort(non_face_confs),'r'); 
plot([0 size(non_face_confs,1)], [0 0], 'b');
hold off;

% Visualize the learned detector. This would be a good thing to include in
% your writeup!
n_hog_cells = sqrt(length(w) / 31); %specific to default HoG parameters
imhog = vl_hog('render', single(reshape(w, [n_hog_cells n_hog_cells 31])), 'verbose') ;
figure(3); imagesc(imhog) ; colormap gray; set(3, 'Color', [.988, .988, .988])

pause(0.1) %let's ui rendering catch up
hog_template_image = frame2im(getframe(3));
% getframe() is unreliable. Depending on the rendering settings, it will
% grab foreground windows instead of the figure in question. It could also
% return a partial image.
imwrite(hog_template_image, 'visualizations/hog_template.png')
    
 
% step 4. (optional extra credit) Mine hard negatives
% Mining hard negatives is graduate credit / extra credit. You can get very
% good performance by using random negatives, so hard negative mining is
% somewhat unnecessary for face detection. If you implement hard negative
% mining, you probably want to modify 'run_detector', run the detector on
% the images in 'non_face_scn_path', and keep all of the features above
% some confidence level. Hard negative mining would probably be more
% important if you had a strict budget of negative training examples or a
% more expressive, non-linear classifier that can benefit from more
% trianing data.


%% Hard Negative Mining

% lambda2 = 0.0005;
% %0.0001 = 70%
% % 0.0005 = 63.5%
% % 0.001 = 78%
% % 0.005 = 71.1
% % 0.01 = 74.8
% threshold1 = -1.5;
% [features_neg, w, b] = hard_mining(features_pos, features_neg, non_face_scn_path, w, b, feature_params, lambda2, threshold1);

%% Step 5. Run detector on test set.
tic
% YOU CODE 'run_detector'. Make sure the outputs are properly structured!
% They will be interpreted in Step 6 to evaluate and visualize your
% results. See run_detector.m for more details.

[lfw_file_paths, features_lfw] = get_lfw(feature_params);
lfw_folder_path = '../data/lfw';
[bboxes, confidences, image_ids] = run_detector_crop_lfw(lfw_file_paths, w, b, feature_params);
%[bboxes, confidences, image_ids] = run_detector2(test_scn_path, w, b, feature_params);
toc
