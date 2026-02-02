function [c1,c2] = twoPointCrossover(p1, p2)
    %size_parent = size(p1);

    point1 = randi([1, size(p1, 2)]);
    point2 = randi([point1, size(p1, 2)]);
    
    c1 = p1;
    c2 = p2;
    
    c1(point1:point2) = p2(point1:point2);
    c2(point1:point2) = p1(point1:point2);
    
    %child = [c1,c2];
end
