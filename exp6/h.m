function result = h(theta, x)  % h����
    %H theta,x
    t = exp(-x * theta);
    result = 1 ./ (1 + t);
end
