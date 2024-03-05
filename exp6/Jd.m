function result = Jd(x, y, theta, lambda)
%H x,theta, lambda
%     a = 1 / 117;
%     b = a * x';
%     c = h(theta, x) - y;
%     d = b * c;
    
    result = 1/ 117 * x' * (h(theta, x) - y) +lambda / 117 * theta ;
    
%     m = 117;
%     n = 28; 
%     ans = zeros(n,1);
%     for j = 1 : n
%         for i = 1 : m
%             xi = x(i,:)'; % 28 * 1
%             ans(j) = ans(j) +  (h2(theta, xi) - y(i)) * xi(j) ;
%         end
%         ans(j) = ans(j) / m + lambda / m * theta(j);
%     end
%     result = ans;
end