function result = hansen(theta, x, lambda)
    t = eye(28);
    t(1,1) = 0;
    result = 1 / 117 * x' * diag(h(theta, x)) * diag(1 - h(theta, x)) * x + lambda / 117 * t;
%     ans = 0;
%     for i = 1:117
%         xi = x(i,:)'; % 28 * 1
%         ans = ans +   xi * xi'* h2(theta,xi) *(1 - h2(theta, xi));
%     end
%     ans = ans / 117;
%     result = ans + lambda / 117 * t;
end