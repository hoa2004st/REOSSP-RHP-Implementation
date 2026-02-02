function child = sb_crossover(p1, p2)
    % follow the paper of Kalyanmoy Deb: An Efficient Constraint Handling Method for Genetic Algorithms
    
    EPS = 1E-6;
    eta_c = 2;
    UB = 1.0;
    LB = 0.0;
    
    dim = size(p1, 2);
    o1 = zeros(1, dim);
    o2 = zeros(1, dim);
    
    for i=1:dim
        if rand <= 0.5 && abs(p1(i) - p2(i)) >= EPS
            % perform crossover
            y1 = p1(i);
            y2 = p2(i);
            if p1(i) > p2(i)
                y1 = p2(i);
                y2 = p1(i);
            end
            
            rand_num = rand;
            
            % first offspring
            beta = 1.0 + (2.0*(y1-LB)/(y2-y1));
            alpha = 2.0 - beta^(-(eta_c+1.0));
            
            if rand_num <= (1.0/alpha)
                betaq = (rand_num*alpha)^(1.0/(eta_c+1.0));
            else
                betaq = (1.0/(2.0 - rand_num*alpha))^(1.0/(eta_c+1.0));
            end
            
            o1(i) = 0.5*((y1+y2) - betaq*(y2-y1));
            if o1(i) > UB
                o1(i) = UB;
            elseif o1(i) < LB
                o1(i) = LB;
            end
            
            % second offspring
            beta = 1.0 + (2.0*(UB-y2)/(y2-y1)); 
            alpha = 2.0 - beta^(-(eta_c+1.0));
            
            if rand_num <= (1.0/alpha)
                betaq = (rand_num*alpha)^(1.0/(eta_c+1.0));
            else
                betaq = (1.0/(2.0 - rand_num*alpha))^(1.0/(eta_c+1.0));
            end
            
            o2(i) = 0.5*((y1+y2)+betaq*(y2-y1));
            if o2(i) > UB
                o2(i) = UB;
            elseif o2(i) < LB
                o2(i) = LB;
            end

        else
            o1(i) = p1(i);
            o2(i) = p2(i);
        end
    end
    
    child = [o1; o2];
end

