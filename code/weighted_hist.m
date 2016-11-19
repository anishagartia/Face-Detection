function [N] = weighted_hist(ang, mag, edges)
    %edges = [-180:360/9:180];
    bin_size = (max(edges) - min(edges))/(numel(edges)-1);
    N = zeros(1,numel(edges)-1);
    for i = 1:numel(ang)
        %[N,edges1] = histcounts(ang,edges);
        centers = edges + (bin_size/2);
        centers = centers(1:end-1);
        i_higher = find(centers >= ang(i),1,'first');
        if (i_higher == 1)
            N(i_higher) = N(i_higher) + mag(i);
            continue;
        elseif (ang(i) >= centers(end))
            N(end) = N(end) + mag(i);
        end 
        contrib_low = (centers(i_higher) - ang(i))/bin_size;       
        contrib_high = (ang(i) - centers(i_higher-1))/bin_size;
        N(i_higher) = N(i_higher) + contrib_high*mag(i);
        N(i_higher-1)= N(i_higher-1) + contrib_low*mag(i);        
    end
end

