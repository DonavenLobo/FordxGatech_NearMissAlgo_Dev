% This code is created by Abdullah Mueen, Yan Zhu, Michael Yeh, Kaveh Kamgar, Krishnamurthy Viswanathan, Chetan Kumar Gupta and Eamonn Keogh.
% The overall time complexity of the code is O(n log n). The code is free to use for research purposes.
% The code may produce imaginary numbers due to numerical errors for long time series where batch processing on short segments can solve the problem.

function [dist] = MASS_absolute(x, y)
%x is the data, y is the query
m = length(y);
n = length(x);

%compute y stats -- O(n)
sumy2 = sum(y.^2);


%compute x stats -- O(n)
sumx2 = movsum(x.^2,[m-1 0]);

y = y(end:-1:1);%Reverse the query
y(m+1:n) = 0; %append zeros

%The main trick of getting dot products in O(n log n) time
X = fft(x);
Y = fft(y);
Z = X.*Y;
z = ifft(Z);

dist = sumx2(m:n)-2*z(m:n)+sumy2;

%dist = 2*(m-(z(m:n)-m*meanx(m:n)*meany)./(sigmax(m:n)*sigmay));
dist = sqrt(dist);