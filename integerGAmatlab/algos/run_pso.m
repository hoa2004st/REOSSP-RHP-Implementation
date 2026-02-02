function [best_solution, convergence, best_obj] = run_pso(dim, obj_func, max_fes, pop_size, w, c1, c2, verbose)
    % Chuyển đổi số lần đánh giá hàm (FEs) thành số thế hệ (Generations)
    iters = round(max_fes / pop_size);
    
    if verbose
        disp("PSO running for maximization, generation = " + iters + ", n=" + pop_size + " ...");
    end

    %% Initialize swarm
    % Cấu trúc hạt nhân: Gene (Vị trí), Velocity (Vận tốc), Fitness
    empty_particle.Gene = [];
    empty_particle.Velocity = [];
    empty_particle.Fitness = -inf;
    empty_particle.BestGene = [];
    empty_particle.BestFitness = -inf;

    Swarm = repmat(empty_particle, pop_size, 1);
    
    % Global Best
    GlobalBest.Gene = [];
    GlobalBest.Fitness = -inf;

    % Khởi tạo quần thể
    for i = 1:pop_size
        % Tạo ngẫu nhiên vị trí trong khoảng [0, 1]
        % Lưu ý: Gene được để dạng hàng (1 x dim) để khớp với obj_func
        Swarm(i).Gene = rand(1, dim); 
        Swarm(i).Velocity = zeros(1, dim);
        
        % Đánh giá Fitness
        Swarm(i).Fitness = obj_func(Swarm(i).Gene);
        
        % Cập nhật Personal Best
        Swarm(i).BestGene = Swarm(i).Gene;
        Swarm(i).BestFitness = Swarm(i).Fitness;
        
        % Cập nhật Global Best
        if Swarm(i).Fitness > GlobalBest.Fitness
            GlobalBest.Fitness = Swarm(i).Fitness;
            GlobalBest.Gene = Swarm(i).Gene;
        end
    end

    Convergence = zeros(iters, 1);
    Convergence(1) = GlobalBest.Fitness;

    %% Main PSO loop
    for iter = 1:iters
        for i = 1:pop_size
            % 1. Update Velocity
            % v = w*v + c1*r1*(pBest - x) + c2*r2*(gBest - x)
            r1 = rand(1, dim);
            r2 = rand(1, dim);
            
            Swarm(i).Velocity = w * Swarm(i).Velocity ...
                + c1 * r1 .* (Swarm(i).BestGene - Swarm(i).Gene) ...
                + c2 * r2 .* (GlobalBest.Gene - Swarm(i).Gene);
            
            % 2. Update Position
            Swarm(i).Gene = Swarm(i).Gene + Swarm(i).Velocity;
            
            % 3. Clamp position to [0, 1]
            Swarm(i).Gene = max(0, min(1, Swarm(i).Gene));
            
            % 4. Evaluation
            Swarm(i).Fitness = obj_func(Swarm(i).Gene);
            
            % 5. Update Personal Best
            if Swarm(i).Fitness > Swarm(i).BestFitness
                Swarm(i).BestGene = Swarm(i).Gene;
                Swarm(i).BestFitness = Swarm(i).Fitness;
            end
            
            % 6. Update Global Best
            if Swarm(i).Fitness > GlobalBest.Fitness
                GlobalBest.Fitness = Swarm(i).Fitness;
                GlobalBest.Gene = Swarm(i).Gene;
            end
        end
        
        Convergence(iter) = GlobalBest.Fitness;
        
        % In kết quả (chỉ in mỗi 50 thế hệ)
        if verbose && mod(iter, 50) == 0
            fprintf('Generation %d, best objective = %.4f\n', iter, GlobalBest.Fitness);
        end
    end
    
    % Trả về kết quả cuối cùng
    best_solution = GlobalBest.Gene;
    best_obj = GlobalBest.Fitness;
    convergence = Convergence;
end