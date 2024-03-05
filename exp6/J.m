function result = J(x, y, theta, lambda)
%H x, y, theta, lambda
    result = 1/ 117 * (- y' * log(h(theta, x)) - (1 - y') * log(1 - h(theta, x))) + lambda / (2*117) * theta'*theta;
    
%     ans = 0;
%     for i = 1 : 117
%         xi = x(i,:)';
%         ans = ans + (y(i)*log(h2(theta, xi) + 0.00000001) + (1 - y(i)) * log(1 - h2(theta, xi)+ 0.00000001));
%     end 
%     ans = ans * (-1/117);
%     t = 0;
%     for j = 1 : 28
%         t = t + theta(j).^2;
%     end 
%     t = t * lambda / (2 * 117);
%     result = ans + t;
end
