function [next_pop, next_fit] = selection(next_size, pop1, fit1, pop2, fit2)
    imm_pop = [pop1; pop2];
    imm_fit = [fit1; fit2];
    
    n = size(imm_pop, 1);
    idx = (1:n);
    
    for i=1:n-1
        for j=i+1:n
            if imm_fit(idx(i)) < imm_fit(idx(j))
                tmp = idx(i);
                idx(i) = idx(j);
                idx(j) = tmp;
            end
        end
    end
    
    next_pop = zeros(next_size, size(pop1, 2));
    next_fit = zeros(next_size, 1);
    for i=1:next_size
        next_pop(i, :) = imm_pop(idx(i), :);
        next_fit(i) = imm_fit(idx(i));
    end
end