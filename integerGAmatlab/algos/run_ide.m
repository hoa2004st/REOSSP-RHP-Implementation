function [best_solution, convergence, best_objective] = run_ide(dim, obj_func, max_fes, pop_size, pbest_rate, mem_size, arc_rate, verbose)
    disp("IDE running for maximization, generation = " + max_fes/pop_size + ", n=" + pop_size + " ...");

    count_fes = 0;
    convergence = zeros(round(max_fes / pop_size), 1);
    
    % init population and best solution
    pop = rand(pop_size-1, dim);
    new_pop = ones(1,dim);
    pop = [pop; new_pop];
    
    fitness = -1e9 * ones(pop_size, 1);
    best_solution = rand(1, dim);
    best_objective = -1e9;
    
    % evaluate the init population
    for i=1:pop_size
        fitness(i) = obj_func(pop(i, :));
        count_fes = count_fes+1;
        if fitness(i) > best_objective
            best_objective = fitness(i);
            best_solution = pop(i, :);
        end
    end
    
    % for parameters sampling
    M = 2;
    success_sf = zeros(M, pop_size);
    success_cr = zeros(M, pop_size);
    dif_fitness = zeros(M, pop_size);
    
    mem_sf = 0.2 * ones(M, mem_size);
    mem_cr = 1.0 * ones(M, mem_size);
    mem_pos = ones(M, 1);
    
    % for best operator
    best_op = 1;
    lambda = 0.2;
    consumed_fes = ones(M, 1);
    sum_improv = zeros(M, 1);
    
    arc_size = round(pop_size * arc_rate);
    archieve = zeros(arc_size, dim);
    arc_count = 0;
    
    generation = 1;
    disp("Generation " + generation + ", best objective = " + best_objective);
    convergence(generation) = best_objective;
    
    offspring = zeros(pop_size, dim);
    offspring_fit = -1e9 * ones(pop_size, 1);

    p_num = round(pbest_rate * pop_size);
    p_num = max(p_num, 2);
    
    while count_fes < max_fes
        generation = generation + 1;
        
        % for current-to-pbest/1
        [sorted_fitness, sorted_idx] = sort(fitness, 'descend');
        
        % reproduction
        next_pop = zeros(size(pop));
        next_pop_fitness = zeros(size(fitness));
        
        success_count = zeros(M, 1);
        
        for i=1:pop_size
            p_best_idx = sorted_idx(randi([1, p_num]));
            
            opcode = -1;
            rand_op = rand;
            for j=1:M
                if (j-1) * lambda <= rand_op && rand_op < j * lambda
                    opcode = j;
                    break;
                end
            end
            if opcode == -1
                opcode = best_op;
            end
            consumed_fes(opcode) = consumed_fes(opcode) + 1;
            
            % sample F and CR
            rand_pos = randi([1, mem_size]);
            mu_cr = mem_cr(opcode, rand_pos);
            mu_f = mem_sf(opcode, rand_pos);
            
            if mu_cr == -1
                CR = 0;
            else
                CR = normrnd(mu_cr, 0.1);
                CR = max(min(CR, 1), 0);
            end
            
            F = cauchy(mu_f, 0.1);
            while F <= 0
                F = cauchy(mu_f, 0.1);
            end
            F = min(F, 1.0);
            
            if opcode == 1
                offspring(i, :) = operatePBest1Bin(pop, i, p_best_idx, CR, F, archieve, arc_count);
            elseif opcode == 2
                offspring(i, :) = operateCurrentToPBest1Bin(pop, i, p_best_idx, CR, F, archieve, arc_count);
            end
            
            offspring_fit(i) = obj_func(offspring(i, :));
            
            if fitness(i) <= offspring_fit(i)
                next_pop(i, :) = offspring(i, :);
                next_pop_fitness(i) = offspring_fit(i);
                
                if offspring_fit(i) > best_objective
                    best_objective = offspring_fit(i);
                    best_solution = offspring(i, :);
                end
                
                if fitness(i) < offspring_fit(i)
                    pos = success_count(opcode) + 1;
                    success_count(opcode) = pos;
                    success_sf(opcode, pos) = F;
                    success_cr(opcode, pos) = CR;
                    dif_fitness(opcode, pos) = offspring_fit(i) - fitness(i);
                    
                    sum_improv(opcode) = sum_improv(opcode) + offspring_fit(i) - fitness(i);
                end
                
                if arc_count < arc_size
                    arc_count = arc_count + 1;
                    archieve(arc_count, :) = pop(i, :);
                else
                    rm_pos = randi([1, arc_size]);
                    archieve(rm_pos, :) = pop(i, :);
                end
            else
                next_pop(i, :) = pop(i, :);
                next_pop_fitness(i) = fitness(i);
            end
            
        end
        count_fes = count_fes + pop_size;
        convergence(generation) = best_objective;
        
        
        pop = next_pop;
        fitness = next_pop_fitness;
        
        % update the parameter memory
        for m=1:M
            sc = success_count(m);
            if sc > 0
                sum_improvement = sum(dif_fitness(m, 1:sc));
                weight = dif_fitness(m, 1:sc) / sum_improvement;
                mem_sf(m, mem_pos(m)) = sum(weight.*(success_sf(m, 1:sc).^2)) / sum(weight.*(success_sf(m, 1:sc)));
                
                mem_cr(m, mem_pos(m)) = sum(weight.*(success_cr(m, 1:sc).^2));
                tmp = sum(weight.*(success_cr(m, 1:sc)));
                if tmp == 0 || mem_cr(m, mem_pos(m)) == -1
                    mem_cr(m, mem_pos(m)) = -1;
                else
                    mem_cr(m, mem_pos(m)) = mem_cr(m, mem_pos(m)) / tmp;
                end
                
                mem_pos(m) = mem_pos(m) + 1;
                if mem_pos(m) == mem_size + 1
                    mem_pos(m) = 1;
                end
            end
        end
        
        % update best operator
        if rem(generation, 10) == 0 
            new_best_op = -1;
            best_improve_rate = 0;
            for m=1:M
                improve_rate = sum_improv(m) / consumed_fes(m);
                if improve_rate > best_improve_rate
                    best_improve_rate = improve_rate;
                    new_best_op = m;
                end
                
                sum_improv(m) = 0;
                consumed_fes(m) = 1;
            end
            if new_best_op == -1
                best_op = randi([1, M]);
            else
                best_op = new_best_op;
            end
        end
        
        if verbose
        %if generation == max_fes / pop_size
        if mod(generation, 100) == 0
            disp("Generation " + generation + ", best objective = " + best_objective);
        end
    end

end

