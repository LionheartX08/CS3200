%CS 3200 Assingment 1
%Auther - Jake Betenson | u0624782

vectorTiming = RandVectorsDP(100)

%Format from https://www.mathworks.com/help/matlab/ref/tic.html
%Specifically Take Measurements using Multiple tic Calls
%Function determines the amount of time taken to perform N length vector
%dot product from size 1,N
function out = RandVectorsDP(n)
    T = zeros(1,n); %timing array
    for i = 1:10:n
        A = rand(i,1); %some random column vector between 0,1
        B = rand(1,i); %some random row vector between 0,1
        tic %start timing dot product of vectors
        dot(A,B);
        T(i) = toc; %store elapsed time in timing array
    end
    out = T;
end

%Format from https://www.mathworks.com/help/matlab/ref/tic.html
%Specifically Take Measurements using Multiple tic Calls
%Function determines the amount of time taken to perform N x N matrix
%dot product from size 1,N
function out = RandNbyNDP(n)
    T = zeros(1,n);
    for i = 1:10:n
        A = rand(i,i); %some random n x n array [0,1]
        B = rand(i,i); %some random n x n array [0,1]
        tic %start timing dot product of vectors
        dot(A,B);
        T(i) = toc; %store elapsed time in timing array
    end
    out = T;
end




