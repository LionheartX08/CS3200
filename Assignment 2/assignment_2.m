% CS 3200 - Assignment 2
% Author: Jake Betenson | u0624782
% Compute polynomial interpolants for the function exp(x) on [0,2] using
% evenly spaced and chebyshev points and with interpolants contructed using
% the Vandermonde equations, the polyinterp.m routine, the barycentric 
% interpolation routine barylag.m and finally the matlab cubic spline routine. 
% This process will enable you to understand which method is the  most robust 
% and accurate and how this accuracy varies with the choice of points and also 
% what the cost of the methods is.

% % 1 - Write a simple function to plot the expoential function on the
% % interval [0,2] using 1001 evenly spaced points
% 
% part1a = evenspace(1001);
% part1b = exp12(part1a);
% figure
% plot(part1a,part1b);
% xlabel('x');
% ylabel('exp(x)');
% title('Plot 1001 points of Exp(x) on [0,2]');

% % 2 - Modify your program to use the Vandermonde matrix to calculate an interpolating 
% % polynomial to exp(x) on [0,2] with 6,11,21,41,81,161,321 and 641 points and to 
% % evaluate the accuracy at 1001 sample points using the infinity and 2 norms.  
% % Use both evenly spaced points and Chebyshev spaced points mapped to [0,2]
% % as given by x(i) = 1 –cos(π(i-1)/(n-1)) for i = 1,...,n. Comment on and 
% % describe how the accuracy changes with the choice and number of points.
% % general code format provided by Dr. Martin Berzins, lecture INTERPOLATION
% % 2021_4
% 
% %Vandermonde Error Calcs of exp(x)
% fprintf('\nVandermonde Method\n');
% timing_ve = zeros(8,1);
% acc_ve = zeros(8,1);
% for i = 1:8
%     %get x values
%     tic;
%     part2a = evenspace(1+5*(2)^(i-1))';
%     n = 1+5*(2)^(i-1);
%     nm1 = n-1;   
%     
%     expx = exp(part2a);
%     %get polynomial coefficients
%     A = fliplr(vander(part2a));
%     a = A\expx;
%     %interpolation
%     nplot = 1001;
%     sqrth = 1.0/sqrt(n);
%     xplot = linspace(1,2,nplot);
%     expplot = exp(xplot);
%     yplot = ones(1,nplot)*a(n);
%     for j = 1:nm1
%         yplot = yplot.*xplot+a(n-j);
%     end
%     timing_ve(i) = toc;
%     %error calc
%     errinf = norm((yplot-expplot),inf);
%     acc_ve(i) = errinf;
%     err2 = sqrth*norm((yplot-expplot),2);
%     fprintf('n=%i, infinity error = %8.2e, 2-norm error = %8.2e \n', n ,errinf, err2);   
% end
% 
% %Chebyshev Error Calcs of exp(x)
% fprintf('\nChebyshev Method\n');
% timing_vc = zeros(8,1);
% acc_vc = zeros(8,1);
% for i = 1:8
%     %get x values
%     tic;
%     n = 1+5*(2)^(i-1);
%     nm1 = n-1;
%     x = zeros(n,1); y = zeros(n,1);
%     for k = 1:n
%         x(k) = 1-cos(pi*(k-1)/(n-1));
%         y(k) = exp(x(k));
%     end        
%     
%     %get polynomial coefficients
%     A = fliplr(vander(x));
%     a = A\y;
%     %interpolation
%     nplot = 1001;
%     sqrth = 1.0/sqrt(n);
%     xplot = linspace(1,2,nplot);
%     expplot = exp(xplot);
%     yplot = ones(1,nplot)*a(n);
%     for j = 1:nm1
%         yplot = yplot.*xplot+a(n-j);
%     end
%     timing_vc(i)=toc;
%     %error calc
%     errinf = norm((yplot-expplot),inf);
%     acc_vc(i) = errinf;
%     err2 = sqrth*norm((yplot-expplot),2);
%     fprintf('n=%i, infinity error = %8.2e, 2-norm error = %8.2e \n', n ,errinf, err2);   
% end

% 3 - Extend the program to use the  Matlab  routines  polyinterp  and barylag 
% that use Lagrange  polynomial interpolation . Again comment on how the accuracy 
% varies with the different choice of 6,11,21,41,81,161,321 and 641 points, 
% both Chebyshev and evenly spaced. general code format provided by Dr.
% Martin Berzins in lecture

% even spaced Lagrange Polynomial Interpolation
% fprintf('Even Spaced Langrange\n')
% timing_le = zeros(8,1);
% acc_le = zeros(8,1);
% for i = 1:8
%     tic;
%     n = 1+5*(2^(i-1));
%     x = evenspace(n);
%     y = exp(x);
%     nplot = 1001;
%     sqrth = 1.0/sqrt(n);
%     u = evenspace(1001);
%     xplot = evenspace(1001);
%     yplot = exp(xplot);
%     %interp step
%     v = polyinterp(x,y,u);
%     %error calc
%     timing_le(i) = toc;
%     errinf = norm((v-yplot),inf);
%     acc_le(i) = errinf;
%     err2 = sqrth*norm((v-yplot),2);
%     fprintf('n=%i, infinity error = %8.2e, 2-norm error = %8.2e \n', n ,errinf, err2);
% end


% % Chebyshev spaced Langrange Polynomial Interpolation
% % even spaced Lagrange Polynomial Interpolation
% fprintf('Chebyshev Langrange\n')
% timing_lc = zeros(8,1);
% acc_lc = zeros(8,1);
% for i = 1:8
%     tic;
%     n = 1+5*(2^(i-1));
%     x = zeros(n,1); y = zeros(n,1);
%     for k = 1:n
%         x(k) = 1-cos(pi*(k-1)/(n-1));
%         y(k) = exp(x(k));
%     end 
%    
%     nplot = 1001;
%     sqrth = 1.0/sqrt(n);
%     u = evenspace(1001);
%     %real values
%     xplot = evenspace(1001);
%     yplot = exp(xplot);
%     %interp step
%     v = polyinterp(x,y,u);
%     %error calc
%     timing_lc(i) = toc;
%     errinf = norm((v-yplot),inf);
%     acc_lc(i) = errinf;
%     err2 = sqrth*norm((v-yplot),2);
%     fprintf('n=%i, infinity error = %8.2e, 2-norm error = %8.2e \n', n ,errinf, err2);
% end

% % Barycentric Lagrange
% fprintf('Barylag Even Spaced\n')
% timing_ble = zeros(8,1);
% acc_ble = zeros(8,1);
% for i = 1:8
%     tic;
%     n = 1+5*(2^(i-1));
%     x = evenspace(n)';
%     y = exp(x);
%    
%     nplot = 1001;
%     %real values
%     xplot = evenspace(nplot)';
%     yplot = exp(xplot);
%     
%     %interp step
%     data = [x(:), y(:)];
%     bl = barylag(data, xplot);
%     timing_ble(i) = toc;
%     %error calc
%     sqrth = 1.0/sqrt(n);
%     errinf = norm((bl-yplot),inf);
%     acc_ble(i) = errinf;
%     err2 = sqrth*norm((bl-yplot),2);
%     fprintf('n=%i, infinity error = %8.2e, 2-norm error = %8.2e \n', n ,errinf, err2);
% end
% 
% plot(x, bl, '*', x, yplot);
% 
% Barycentric Lagrange
% fprintf('Barylag Chebyshev\n')
% timing_blc = zeros(8,1);
% acc_blc = zeros(8,1);
% for i = 1:8
%     tic;
%     n = 1+5*(2^(i-1));
%     x = zeros(n,1); y = zeros(n,1);
%     for k = 1:n
%         x(k) = 1-cos(pi*(k-1)/(n-1));
%         y(k) = exp(x(k));
%     end 
%    
%     nplot = 1001;
%     %real values
%     xplot = evenspace(nplot)';
%     yplot = exp(xplot);
%     
%     %interp step
%     data = [x(:), y(:)];
%     bl = barylag(data, xplot);
%     timing_blc(i) = toc;
%     %error calc
%     sqrth = 1.0/sqrt(n);
%     errinf = norm((bl-yplot),inf);
%     acc_blc(i) = errinf;
%     err2 = sqrth*norm((bl-yplot),2);
%     fprintf('n=%i, infinity error = %8.2e, 2-norm error = %8.2e \n', n ,errinf, err2);
% end

% 4 - Further  extend your program to use the Matlab cubic spline routine with 
% both even and Chebyshev points.  Again use both sets of points and comment 
% on the outcome. Is the choice of points so critical for the spline routine 
% as it was for the Lagrange polynomials? general code format provided by Dr. Martin Berzins,

fprintf('Spline Evenly Spaced\n');
timing_se = zeros(1,8);
acc_se = zeros(1,8);
for i = 1:8
    tic;
    n = 1+5*(2^(i-1));    
    %Real Values
    nplot = 1001;
    x = evenspace(n)';
    y = exp(x);
    %for plotting later
    xplot = evenspace(nplot)';
    yplot = exp(xplot);
    %Interpolation
    p = splinetx(x,y,xplot);
    timing_se(i) = toc;
    %error calc
    sqrth = 1.0/sqrt(n);
    errinf = norm((p-yplot),inf);
    acc_se(i) = errinf;
    err2 = sqrth*norm((p-yplot),2);
    fprintf('n=%i, infinity error = %8.2e, 2-norm error = %8.2e \n', n ,errinf, err2);
end


fprintf('Spline Chebyshev Spaced\n');
timing_sc = zeros(8,1);
acc_sc = zeros(8,1);
for i = 1:8
    tic;
    n = 1+5*(2^(i-1));
    
    %Real Values
    nplot = 1001;
    x = zeros(n,1); y = zeros(n,1);
    for k = 1:n
        x(k) = 1-cos(pi*(k-1)/(n-1));
        y(k) = exp(x(k));
    end 
    %for plotting later
    xplot = evenspace(nplot)';
    yplot = exp(xplot);
    %Interpolation
    p = splinetx(x,y,xplot);
    timing_sc(i) = toc;
    %error calc
    sqrth = 1.0/sqrt(n);
    errinf = norm((p-yplot),inf);
    acc_sc(i) = errinf;
    err2 = sqrth*norm((p-yplot),2);
    fprintf('n=%i, infinity error = %8.2e, 2-norm error = %8.2e \n', n ,errinf, err2);
end

% 5 - Accuracy and Timing Comparisons

% error
% figure
% x = [6,11,21,41,81,161,321,641];
% plot(x,acc_blc,x,acc_ble,x,acc_lc,x,acc_le,x,acc_sc,x,acc_se,x,acc_vc,x,acc_ve);
% legend('BaryLag C','BaryLag E','Lag C','Lag E', 'Spline C', 'Spline E', 'Vand C', 'Vand E');
% xlim([0 641]);
% ylim([0 5e-13]);
% title('Error Comparison')
% xlabel('N')
% ylabel('Error (lower is better)')
% % Timing
% figure
% x = [6,11,21,41,81,161,321,641];
% plot(x,timing_blc,x,timing_ble,x,timing_lc,x,timing_le,x,timing_sc,x,timing_se,x,timing_vc,x,timing_ve);
% legend('BaryLag C','BaryLag E','Lag C','Lag E', 'Spline C', 'Spline E', 'Vand C', 'Vand E');
% xlim([0 641]);
% title('Timing Comparison')
% xlabel('N')
% ylabel('Time (lower is better)')
% ylim([0 1e-1]);

% Part 4 functions
% Given from Class
function v = splinetx(x,y,u)
    h = diff(x);
    delta = diff(y)./h;
    d=splineslopes(h,delta);
    
    n = length(x);
    c = (3*delta-2*d(1:n-1) - d(2:n))./h;
    b = (d(1:n-1) - 2*delta + d(2:n))./h.^2;
    
    k = ones(size(u)); 
    for j = 2:n-1      
        k(x(j) <= u) = j;   
    end%  Evaluate spline
    s = u - x(k);   
    v = y(k) + s.*(d(k) + s.*(c(k) + s.*b(k)));
end

% Given from Class
function d = splineslopes(h,delta)
    n = length(h)+1;
    a = zeros(size(h)); b = a; c = a; r = a;
    a(1:n-2) = h(2:n-1);
    a(n-1) = h(n-2)+h(n-1);
    b(1) = h(2);
    b(2:n-1) = 2*(h(2:n-1)+h(1:n-2));
    b(n) = h(n-2);
    c(1) = h(1)+h(2);
    c(2:n-1) = h(1:n-2);
    r(1) = ((h(1)+2*c(1))*h(2)*delta(1)+ ...
        h(1)^2*delta(2))/c(1);   
    r(2:n-1) = 3*(h(2:n-1).*delta(1:n-2)+ ... 
        h(1:n-2).*delta(2:n-1));
    r(n) = (h(n-1)^2*delta(n-2)+ ...          
        (2*a(n-1)+h(n-1))*h(n-2)*delta(n-1))/a(n-1);
    
    d = tridisolve(a,b,c,r);
end

% Function from 
% https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/37976/versions/7/previews/tridisolve.m/index.html
function x = tridisolve(a,b,c,d)
%   TRIDISOLVE  Solve tridiagonal system of equations.
%     x = TRIDISOLVE(a,b,c,d) solves the system of linear equations
%     b(1)*x(1) + c(1)*x(2) = d(1),
%     a(j-1)*x(j-1) + b(j)*x(j) + c(j)*x(j+1) = d(j), j = 2:n-1,
%     a(n-1)*x(n-1) + b(n)*x(n) = d(n).
%
%   The algorithm does not use pivoting, so the results might
%   be inaccurate if abs(b) is much smaller than abs(a)+abs(c).
%   More robust, but slower, alternatives with pivoting are:
%     x = T\d where T = diag(a,-1) + diag(b,0) + diag(c,1)
%     x = S\d where S = spdiags([[a; 0] b [0; c]],[-1 0 1],n,n)
    x = d;
    n = length(x);
    for j = 1:n-1
        mu = a(j)/b(j);
        b(j+1) = b(j+1) - mu*c(j);
        x(j+1) = x(j+1) - mu*x(j);
    end
    x(n) = x(n)/b(n);
    for j = n-1:-1:1
        x(j) = (x(j)-c(j)*x(j+1))/b(j);
    end
end

%part 1 functions
function v = evenspace(n)
    %EVENSPACE generates an evenly spaced vector between 1 and 2 with n
    %points.
    v = linspace(0,2,n);
end

function y = exp12(v)
    %exp12 returns exp(x) with vector v inputs
    y = exp(v);
end

