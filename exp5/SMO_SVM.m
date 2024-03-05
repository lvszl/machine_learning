function [alphas, bias] = SMO_SVM(train_x, train_y, C, tol, max_passes)
    [m, n] = size(train_x);
    alphas = zeros(m, 1);
    bias = 0;
    passes = 0;

    while passes < max_passes
        num_changed_alphas = 0;
        for i = 1:m
            Ei = bias + sum(alphas .* train_y .* (train_x * train_x(i, :)')) - train_y(i);
            if ((train_y(i) * Ei < -tol && alphas(i) < C) || (train_y(i) * Ei > tol && alphas(i) > 0))
                j = select_second_alpha(i, m);
                Ej = bias + sum(alphas .* train_y .* (train_x * train_x(j, :)')) - train_y(j);

                alpha_i_old = alphas(i);
                alpha_j_old = alphas(j);

                [L, H] = calculate_bounds(train_y, alphas, C, i, j);

                if L == H
                    continue;
                end

                eta = 2 * train_x(i, :) * train_x(j, :)' - train_x(i, :) * train_x(i, :)' - train_x(j, :) * train_x(j, :)';
                if eta >= 0
                    continue;
                end

                alphas(j) = alphas(j) - (train_y(j) * (Ei - Ej)) / eta;
                alphas(j) = min(H, alphas(j));
                alphas(j) = max(L, alphas(j));

                if abs(alphas(j) - alpha_j_old) < tol
                    alphas(j) = alpha_j_old;
                    continue;
                end

                alphas(i) = alphas(i) + train_y(i) * train_y(j) * (alpha_j_old - alphas(j));

                b1 = bias - Ei - train_y(i) * (alphas(i) - alpha_i_old) * train_x(i, :) * train_x(i, :)' - train_y(j) * (alphas(j) - alpha_j_old) * train_x(i, :) * train_x(j, :)';
                b2 = bias - Ej - train_y(i) * (alphas(i) - alpha_i_old) * train_x(i, :) * train_x(j, :)' - train_y(j) * (alphas(j) - alpha_j_old) * train_x(j, :) * train_x(j, :)';

                if 0 < alphas(i) && alphas(i) < C
                    bias = b1;
                elseif 0 < alphas(j) && alphas(j) < C
                    bias = b2;
                else
                    bias = (b1 + b2) / 2;
                end

                num_changed_alphas = num_changed_alphas + 1;
            end
        end

        if num_changed_alphas == 0
            passes = passes + 1;
        else
            passes = 0;
        end
    end
end