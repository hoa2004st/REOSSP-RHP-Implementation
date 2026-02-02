function p = polynomial_mutation(p)

    distributionIndex = 5;
    UB = 1;
    LB = 0;
    
    dim = size(p, 2);
    mutProb = 1.0 / dim;
    
    for i=1:dim
        if rand <= mutProb
            delta1 = (p(i) - LB) / (UB - LB);
            delta2 = (UB - p(i)) / (UB - LB);
            rand_num = rand;
            mutPow = 1.0 / (distributionIndex + 1.0);

            if rand_num <= 0.5
                val = 2.0 * rand_num + (1.0 - 2.0 * rand_num) * ((1.0 - delta1)^(distributionIndex + 1.0));
                deltaq = val^mutPow - 1.0;
            else
                val = 2.0 * (1.0 - rand_num) + 2.0 * (rand_num - 0.5) * ((1.0 - delta2)^(distributionIndex + 1.0));
                deltaq = 1.0 - val^mutPow;
            end

            p(i) = p(i) + deltaq * (UB - LB);
            if p(i) > UB
                p(i) = UB;
            elseif p(i) < LB
                p(i) = LB;
            end
        end
    end
end

