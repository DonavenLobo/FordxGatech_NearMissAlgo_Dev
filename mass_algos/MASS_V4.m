% This code is created by Sheng Zhong and Abdullah Mueen
% The overall time complexity of the code is O(n log n). The code is free to use for research purposes.
% The code does not produce imaginary numbers due to numerical errors 
% k should greater than or equals to floor((3m+1)/2)

function [dist] = MASS_V4(T, Q, k)
    n = length(T);
    m = length(Q);
    Q = zNorm(Q);
    dist = [];
    batch=get_batch_size(k, m);
    
    for j = 1 : batch-m+1 : n-m+1
        right = j + batch - 1;
        if right >= n
            right = n;
        end 
        dot_p = DCT_dot_product(T(j:right), Q);
        sigmaT = movstd(T(j:right),[m-1 0],1);
        d = sqrt(2*(m-(dot_p./sigmaT(m:end)))) ;        
        dist = [dist; d];
    end
    
end

function dct_batch_size = get_batch_size(target_batch_size, m)
    dct_batch_size = floor((2*target_batch_size-2)/3) -1;
    if dct_batch_size < m
        dct_batch_size = m;
    end
    pad_length = dct_batch_size + floor((dct_batch_size-m+1)/2) + floor((m+1)/2);
    while pad_length < target_batch_size
        dct_batch_size = dct_batch_size + 1;
        pad_length = dct_batch_size + floor((dct_batch_size-m+1)/2) + floor((m+1)/2);
    end
    if (pad_length > target_batch_size)
        dct_batch_size = dct_batch_size - 1;
    end
end


function dot_p = DCT_dot_product(x, y)
    n = length(x);
    m = length(y);
    [x_pad, y_pad, si] = DCT_padding(x,y);
    
    N = length(x_pad);
    
    Xc = dct(x_pad, 'Type', 2);
    Yc = dct(y_pad, 'Type', 2);
    
    dct_product = Xc .* Yc;
    dct_product(N + 1) = 0;
    dct_product(1) = dct_product(1) * sqrt(2);
    dot_p = dct(dct_product, 'Type', 1);
    dot_p(1) = dot_p(1) * 2;
    dot_p = sqrt(2*N) * dot_p(si: si+n-m);
end

function [x_pad, y_pad, start_index] = DCT_padding(x, y)
    n = length(x);
    m = length(y);
    p2=floor((n-m+1)/2);
    p1=p2+floor((m+1)/2);
    p4=n-m+p1-p2;
    x_pad = zeros(n+p1, 1);
    x_pad(1+p1:end) = x;
    y_pad = zeros(m+p2+p4, 1);
    y_pad(1+p2:1+p2+m-1) = y;
    start_index = p1-p2+1;
end