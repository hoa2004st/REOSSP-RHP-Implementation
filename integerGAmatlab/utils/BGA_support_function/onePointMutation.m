function p = onePointMutation(p)
    rand_idx = randi([1, size(p, 2)]);
    p(rand_idx) = 1 - p(rand_idx);
end

