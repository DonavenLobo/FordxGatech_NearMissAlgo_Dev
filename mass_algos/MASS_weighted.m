% This code is created by Abdullah Mueen, Krishnamurthy Viswanathan, Chetan Kumar Gupta and Eamonn Keogh.
% The overall time complexity of the code is O(n log n). The code is free to use for research purposes.
% The code may produce imaginary numbers due to numerical errors for long time series where batch processing on short segments can solve the problem.

function dist = MASS_weighted(x,y,w,varargin)

    if nargin>2
        
        %x is the data, y is the query
        n = length(x);
        y = (y-mean(y))./std(y,1);                      %Normalize the query. If you do not want to normalize just comment this line.
        m = length(y);
        
        %compute y stats -- O(n)
        sumy = sum(w.*y);
        sumy2 = sum(w.*(y.^2));
        sumw = sum(w);
        x(n+1:2*n) = 0;                                 %Append zeros
        y = y(end:-1:1);                                %Reverse the query
        y(m+1:2*n) = 0;                                 %Append zeros
        w = w(end:-1:1);                                %Reverse the weights
        w(m+1:2*n) = 0;                                 %Append zeros
        
        
        %The main trick of getting dot products in O(n log n) time. The algorithm is described in [a].
        X = fft(x);                                     %Change to Frequency domain
        Y = fft(w.*y);                                  %Change to Frequency domain

        Z = X.*Y;                                       %Do the dot product
        z = ifft(Z);                                    %Come back to Time domain

        %compute x stats -- O(n)
        cum_sumx = cumsum(x);                           %Cumulative sums of x
        cum_sumx2 = cumsum(x.^2);                       %Cumulative sums of x^2
        sumx2 = cum_sumx2(m+1:n)-cum_sumx2(1:n-m);      %Sum of x^2 of every subsequences of length m
        sumx = cum_sumx(m+1:n)-cum_sumx(1:n-m);         %Sum of x of every subsequences of length m
        meanx = sumx./m;                                %Mean of every subsequences of length m
        sigmax2 = (sumx2./m)-(meanx.^2);
        sigmax = sqrt(sigmax2);                         %Standard deviaiton of every subsequences of length m
        
        %compute x stats -- O(n log n)
        W = fft(w);
        S = X.*W;
        s = ifft(S);
        X = fft(x.^2);
        S2  = X.*W;
        s2 = ifft(S2);
        sumx2 = s2(m+1:n);                              %Sum of x^2 of every subsequences of length m
        sumx = s(m+1:n);                                %Sum of x of every subsequences of length m
    
        
        %computing the distances -- O(n) time. The formula is described in [b].
        dist = (sumx2 - 2*sumx.*meanx + sumw*(meanx.^2))./sigmax2 - 2*(z(m+1:n) - sumy.*meanx)./sigmax + sumy2;
        dist = abs(sqrt(dist));

    else

    %x is the data, y is the query
    n = length(x);
    y = (y-mean(y))./std(y,1);                      %Normalize the query. If you do not want to normalize just comment this line.
    m = length(y);
    x(n+1:2*n) = 0;                                 %Append zeros
    y = y(end:-1:1);                                %Reverse the query
    y(m+1:2*n) = 0;                                 %Append zeros

    %The main trick of getting dot products in O(n log n) time. The algorithm is described in [a].
    X = fft(x);                                     %Change to Frequency domain
    Y = fft(y);                                     %Change to Frequency domain
    Z = X.*Y;                                       %Do the dot product
    z = ifft(Z);                                    %Come back to Time domain

    %compute y stats -- O(n)
    sumy = sum(y);
    sumy2 = sum(y.^2);

    %compute x stats -- O(n)
    cum_sumx = cumsum(x);                           %Cumulative sums of x
    cum_sumx2 = cumsum(x.^2);                       %Cumulative sums of x^2
    sumx2 = cum_sumx2(m+1:n)-cum_sumx2(1:n-m);      %Sum of x^2 of every subsequences of length m
    sumx = cum_sumx(m+1:n)-cum_sumx(1:n-m);         %Sum of x of every subsequences of length m
    meanx = sumx./m;                                %Mean of every subsequences of length m
    sigmax2 = (sumx2./m)-(meanx.^2);
    sigmax = sqrt(sigmax2);                         %Standard deviaiton of every subsequences of length m


    %computing the distances -- O(n) time. The formula is described in [b].
    dist = (sumx2 - 2*sumx.*meanx + m*(meanx.^2))./sigmax2 - 2*(z(m+1:n) - sumy.*meanx)./sigmax + sumy2;
    dist = abs(sqrt(dist));

    %If you want Pearson's correlation coefficients instead of Euclidean
    %Distance use uncomment the next line. The formula is described in [c].
    %CorrCoef = 1-dist./(2*m);
    end
end

%[a] Abdullah Mueen, Hossein Hamooni, Trilce Estrada: Time Series Join on Subsequence Correlation. ICDM 2014: 450-459
%[b] Abdullah Mueen, Eamonn J. Keogh, Neal Young: Logical-shapelets: an expressive primitive for time series classification. KDD 2011: 1154-1162
%[c] Abdullah Mueen, Suman Nath, Jie Liu: Fast approximate correlation for massive time-series data. SIGMOD Conference 2010: 171-182