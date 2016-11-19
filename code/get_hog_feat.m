function [ feats ] = get_hog_feat( img, cell_dim )
    %Gamma Normalization and Compression
    norm_img = sqrt(img);             
    
    %Compute Gradients
    fx = [-1, 0, 1];
    fy = [-1; 0; 1];    
    fx_img = imfilter(norm_img, fx);
    fy_img = imfilter(norm_img, fy);
    angle_grad_img = atan2d(real(fy_img), real(fx_img)) ;
    mag_grad_img = sqrt( (fy_img.*fy_img) + (fx_img.*fx_img));
    
    %Binning Orientations
    cell_size = [cell_dim cell_dim];
    edges = [-180:360/9:180];
    angle_index = reshape(1:numel(angle_grad_img), size(angle_grad_img));    
    fun = @(block_struct) weighted_hist(angle_grad_img(block_struct.data), mag_grad_img(block_struct.data), edges);
    w_hists = blockproc(angle_index, cell_size, fun, 'UseParallel', true);    
       
   % Conside 2x2 cells for contrast normalization
    block_size = [1 9];
    fun3 = @(bs) const_norm(bs.data); 
        
     
    %feats = zeros( size(img,1)/cell_dim, size(img,1)/cell_dim, cell_dim*9); 
    %feats = blockproc(w_hists, block_size, fun3, 'BorderSize', [1 9], 'PadPartialBlocks', true, 'PadMethod', 'symmetric', 'UseParallel', true, 'TrimBorder', false);
    feats = blockproc(w_hists, [2 18], fun3,'UseParallel', true);

end

function [v] = const_norm(v_mat) 
    
    ep = 0.001;
    norm_mat1 = norm(v_mat);
    v = v_mat ./ (norm_mat1 + ep) ;
    

%     ep = 0.01;
%     norm_mat1 = norm(v_mat(1:2 , 1:18));
%     v1 = v_mat(2,10:18) ./ (norm_mat1 + ep) ;
%     
%     norm_mat1 = norm(v_mat(1:2 , 10:27));
%     v2 = v_mat(2,10:18) ./ (norm_mat1 + ep);
%     
%     norm_mat1 = norm(v_mat(2:3 , 1:18));
%     v3 = v_mat(2,10:18) ./ (norm_mat1 + ep);
%     
%     norm_mat1 = norm(v_mat(2:3 , 10:27));
%     v4 = v_mat(2,10:18) ./ (norm_mat1 + ep);  
%     
%     v = zeros(1,1,36);
%     v(1,1,:) = [v1, v2, v3, v4];
%     
end