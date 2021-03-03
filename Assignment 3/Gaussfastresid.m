clear all
t = zeros(1,200);
normb = zeros(1,200);
norm2step=zeros(1,200); %holds 2 steps of refinement residuals
norm2stepSP=zeros(1,200); %holds 2 steps of single precision refined residuals
normbsp=zeros(1,200); %holds 2 steps of single precision refined residuals
mormbb= normb;
normbre= normb;
model = zeros(1,200);
for m = 1:200
    %Ax = b
    n = m*10;
    A = rand(n,n);
    b = rand(n,1);
    x = A\b; %original result
    xsp = single(x); %single precision result
    r = A*x-b; %first residual
    rsp = single(A*xsp-b); %single precision residual
    
    %iterative refinement
    y = A\-r;
    ysp = single(A\-rsp);    
    x = x+y;
    xsp = single(xsp + ysp);    
    r2 = A*x-b;
    rsp2 = single(A*xsp-b);    
    %second iterative refinement
    y = A\-r2;
    ysp = single(A\-rsp2);    
    x = x+y;
    ysp = single(xsp + ysp);    
    r3 = A*x-b;
    rsp3 = single(A*xsp-b);    
    %plot stuff
    normb(m) = norm(r,inf);
    normbsp(m)=norm(rsp,inf);
    nt(m) = n;
    norm2step(m)=norm(r3,inf);
    norm2stepSP(m)=norm(rsp3,inf);
end 
semilogy(nt,normb,nt,norm2step,nt,norm2stepSP,nt,normbsp)
title('Comparison of Various Residual Methods')
xlabel('Size of Matrix (n x n)')
ylabel('Norm of Residual')
legend('double precsion residual','double precision twice refined','single precision twice refined','single precision residual')

