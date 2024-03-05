
function result = h2(theta, x)  % h¾ØÕó
    %H theta,x
    %t = exp(x * theta);
    %result = 1 ./ (1 + t);
    % theta: 28 * 1
    % x: 28 * 1
    result = 1 / (1 + exp(-x' * theta));
end
