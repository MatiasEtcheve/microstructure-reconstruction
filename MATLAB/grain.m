classdef grain
    properties
        TR % triangulation object of the grain
        P % points of the triangulation
        CL % connectivity list of the triangulation
        coeff % vectors of the PCA whose features are the triangulation points
        barycenter
    end

    methods
        function obj = grain(TR)
            % Constructor of the grain
            obj.TR = TR;
            obj.P = obj.TR.Points;
            obj.CL = obj.TR.ConnectivityList;
            [obj.coeff, ~, ~] = pca(obj.P);
            obj.barycenter = mean(obj.P);
        end

        function grain_size_along_axis = compute_grain_size_along_axis(obj, vector)
            % Computes the grain size along a direction given by a vector
            projections = zeros(size(obj.P(:, 1)));
            for index = 1:size(obj.P)
                point = obj.P(index, :);
                projections(index) = dot(vector, point);
            end
            grain_size_along_axis = max(projections) - min(projections);
        end

        function grain_volume = compute_grain_volume(obj)
            % Computes the grain volume
            grain_volume = 0;
            for row = obj.TR.ConnectivityList.'
                points = obj.P(row, :);
                p1 = points(1, :);
                p2 = points(2, :);
                p3 = points(3, :);
                grain_volume = grain_volume + dot(p1, cross(p2, p3)) / 6;
            end
        end

        function grain_convex_volum = compute_grain_convex_volume(obj)
            % Computes the convex grain volume
            [~, grain_convex_volum] = convhull(obj.P(:, 1), obj.P(:, 2), obj.P(:, 3));
        end

        function grain_angles = compute_grain_angles(obj)
            major_vector = obj.coeff(:, 1);
            theta = acos(major_vector(3));
            phi = acos(major_vector(2)/ sin(theta));
            grain_angles = horzcat(theta, phi);
        end

        function fabrics = compute_fabrics(obj)
            grain_volume = obj.compute_grain_volume();
            grain_convex_volum = obj.compute_grain_convex_volume();
            
            % angles = obj.compute_grain_angles();
            size_grain = obj.compute_grain_size_along_axis(obj.coeff(:, 1));
            sphere_volume = 4 / 3 * pi * (size_grain / 2)^3;
            roundness = grain_volume / sphere_volume;
            aspect_ratio = [obj.compute_grain_size_along_axis(obj.coeff(:, 2)), obj.compute_grain_size_along_axis(obj.coeff(:, 3))] / size_grain;
            solidity = grain_volume / grain_convex_volum;

            fabrics = horzcat(obj.coeff(:, 1).', aspect_ratio, size_grain, solidity, roundness, grain_volume);
        end
    end
end