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
% for i = 1:9
%     %get x values
%     if (i ~= 9)
%         part2a = evenspace(1+5*(2)^(i-1))';
%         n = 1+5*(2)^(i-1);
%         nm1 = n-1;
%     else
%         part2a = evenspace(1001)'; 
%         n = 1001;
%         nm1 = n -1;
%     end
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
%     %error calc
%     errinf = norm((yplot-expplot),inf);
%     err2 = sqrth*norm((yplot-expplot),2);
%     fprintf('n=%i, infinity error = %8.2e, 2-norm error = %8.2e \n', n ,errinf, err2);   
% end
% 
% %Chebyshev Error Calcs of exp(x)
% fprintf('\nChebyshev Method\n');
% for i = 1:9
%     %get x values
%     if (i ~= 9)
%         n = 1+5*(2)^(i-1);
%         nm1 = n-1;
%         x = zeros(n,1); y = zeros(n,1);
%         for k = 1:n
%             x(k) = 1-cos(pi*(k-1)/(n-1));
%             y(k) = exp(x(k));
%         end        
%     else
%         n = 1001;
%         nm1 = n -1;
%         x = zeros(n,1); y = zeros(n,1);
%         for k = 1:n
%             x(k) = 1-cos(pi*(k-1)/(n-1));
%             y(k) = exp(x(k));
%         end
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
%     %error calc
%     errinf = norm((yplot-expplot),inf);
%     err2 = sqrth*norm((yplot-expplot),2);
%     fprintf('n=%i, infinity error = %8.2e, 2-norm error = %8.2e \n', n ,errinf, err2);   
% end

% 3 - Extend the program to use the  Matlab  routines  polyinterp  and barylag 
% that use Lagrange  polynomial interpolation . Again comment on how the accuracy 
% varies with the different choice of 6,11,21,41,81,161,321 and 641 points, 
% both Chebyshev and evenly spaced. 




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

