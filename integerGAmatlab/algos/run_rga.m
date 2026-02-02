function [best_solution, convergence, best_obj] = run_rga(dim, obj_func, max_fes, pop_size, verbose)
    disp("RGA running for maximization, generation = " + max_fes/pop_size + ", n=" + pop_size + " ...");

    indivs = rand(pop_size-1, dim);
    new_pop = ones(1,dim);
    indivs = [indivs; new_pop];
    
    best_indiv = rand(1, dim);
    best_obj = -1e9;
    count_fes = 0;
    objective = -1e9 * ones(pop_size, 1);
    
    % Đánh giá quần thể khởi tạo
    for i=1:pop_size
        objective(i) = obj_func(indivs(i, :));
        count_fes = count_fes+1;
        if objective(i) > best_obj
            best_obj = objective(i);
            best_indiv = indivs(i, :);
        end
    end
    
    generation = 1;

    convergence = zeros(round(max_fes / pop_size), 1);
    convergence(generation) = best_obj;
    
    while count_fes < max_fes
        generation = generation + 1;
        
        % ============ BEGIN REPRODUCTION ==================
        offspring = zeros(pop_size, dim);
        offspring_fitness = zeros(pop_size, 1);
        count = 1;
        while count < pop_size
            p1 = randi([1, size(indivs, 1)]);
            p2 = randi([1, size(indivs, 1)]);
            while p1 == p2
                p2 = randi([1, size(indivs, 1)]);
            end
            
            if rand <= 0.9
                child = sb_crossover(indivs(p1, :), indivs(p2, :));
                offspring(count, :) = child(1, :);
                offspring(count+1, :) = child(2, :);
                count = count+2;
            end
        end
        
        for i=1:size(offspring, 1)
            if rand < 0.01
                offspring(i, :) = polynomial_mutation(offspring(i, :));
                % offspring(i, :) = gaussian_mutation(offspring(i, :));
            end
            
            offspring_fitness(i) = obj_func(offspring(i, :));
            count_fes = count_fes+1;
        end
        % ============== END REPRODUCTION ===================
        
        % Survival selection
        [indivs, objective] = selection(pop_size, indivs, objective, offspring, offspring_fitness);
        
        % Update best solution
        if best_obj < objective(1)
            best_obj = objective(1);
            best_indiv = indivs(1, :);
        end
        
        % --- GIỮ NGUYÊN ĐOẠN NÀY: Chỉ hiển thị nếu chia hết cho 50 ---
        if verbose
            if mod(generation, 100) == 0
                disp("Generation " + generation + ", best objective = " + best_obj);
            end
        end
        
        convergence(generation) = best_obj;
    end

    best_solution = best_indiv;
end