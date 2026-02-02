function offspring = operatePBest1Bin(pop, target_index, best_index, cr, sf, archieve, arc_count)
    % V = X_best + F(X_rand1 - X_rand2)
    LB = 0;
    UB = 1;
    
    pop_size = size(pop, 1);
    dim = size(pop, 2);
    
    r1 = randi([1, pop_size]);
    
    if rand < arc_count / (arc_count + pop_size)
        arc = randi([1, arc_count]);
        v = pop(best_index, :) + sf * (pop(r1, :) - archieve(arc, :));
    else
        r2 = randi([1, pop_size]);
        while r1 == r2
            r2 = randi([1, pop_size]);
        end
        v = pop(best_index, :) + sf * (pop(r1, :) - pop(r2, :));
    end
    
    offspring = pop(target_index, :);
    j_rand = randi([1, dim]);
    
    for j=1:dim
        if rand <= cr || j == j_rand
            offspring(j) = v(j);
            
            if rand <= 0.05
                offspring(j) = offspring(j) + normrnd(0, 0.05);
            end
            
            if offspring(j) > UB
                offspring(j) = UB;
            elseif offspring(j) < LB
                offspring(j) = LB;
            end
        end
    end
end


