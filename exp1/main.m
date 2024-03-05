% 导入数据
x = load('ex1x.dat');
y = load('ex1y.dat');
m=length(y);%store the number of training examples
x=[ones(m,1) x];%add column of ones to x

J_vals = zeros(100, 100);
theta0_vals = linspace(-3, 3, 100);
theta1_vals = linspace(-1, 1, 100);
for i = 1 :length(theta0_vals)
    for j = 1:length(theta1_vals)
        t = [theta0_vals(i); theta1_vals(j)];
        J_vals(i, j) = 1/(2 * m) * (x * t - y)' * (x * t - y);
    end
end

J_vals = J_vals'

figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0');
ylabel('\theta_1')

figure;%contours
contour(theta0_vals, theta1_vals, J_vals, logspace(-3, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1')


