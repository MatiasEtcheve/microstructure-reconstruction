classdef rev
    properties
        TR % triangulation object of the rev
        P % points of the triangulation
        CL % connectivity list of the triangulation
        AdjMat % adjency matrix: used to know the grain graphs
        number_grains % number of grains
        point_index_grain % vector of length = number of points and the values are the index of the grain
        grains % cell of grain objects
        total_volume % total volume of the rev
        x_min
        y_min
        z_min
        x_max
        y_max
        z_max
    end
    methods
        function obj = rev(filename)
            obj.TR = stlread(filename);
            obj.P = obj.TR.Points;
            obj.CL = obj.TR.ConnectivityList;

            obj.x_min = min(obj.P(:, 1));
            obj.y_min = min(obj.P(:, 2));
            obj.z_min = min(obj.P(:, 3));
            obj.x_max = max(obj.P(:, 1));
            obj.y_max = max(obj.P(:, 2));
            obj.z_max = max(obj.P(:, 3));

            obj.total_volume = prod(max(obj.P)-min(obj.P));

            obj.AdjMat = false(size(obj.P, 1));
            for kk = 1:size(obj.TR, 1)
                obj.AdjMat(obj.TR(kk, 1), obj.TR(kk, 2)) = true;
                obj.AdjMat(obj.TR(kk, 2), obj.TR(kk, 3)) = true;
                obj.AdjMat(obj.TR(kk, 3), obj.TR(kk, 1)) = true;
            end
            obj.AdjMat = obj.AdjMat | obj.AdjMat';
            [obj.point_index_grain, bin_sizes] = conncomp(graph(obj.AdjMat));
            [~, obj.number_grains] = size(bin_sizes);

            % we create a obj.grains list, whose values are grain object
            % to know which point of the rev belongs to which grain, we us the adjency matrix
            for grain_index = 1:obj.number_grains
                grain_point_indexes = find(obj.point_index_grain(:, :) == grain_index).';
                cl_index = ismember(obj.CL(:, 1), grain_point_indexes);
                grain_triangulation = triangulation(obj.CL(cl_index, :)-min(obj.CL(cl_index, :), [], 'all')+1, obj.P(grain_point_indexes, :));
                obj.grains{end+1} = grain(grain_triangulation);
            end
        end

        function fabrics = compute_fabrics(obj)
            % computes the fabrics of each grain and append it to a list
            fabrics = zeros(obj.number_grains, 12);
            for index = 1:obj.number_grains
                fabrics(index, :) = obj.grains{index}.compute_fabrics();
            end
        end

        function input_fabrics = compute_input_fabrics(obj)
            % creates a well-formed input fabrics with mean and std
            fabrics = obj.compute_fabrics();
            average = mean(fabrics(:, 1:end-1));
            deviation = std(fabrics(:, 1:end-1));

            aggregates_volume = sum(fabrics(:, 12));
            global_volume_fraction = aggregates_volume / obj.total_volume;

            % six first values are orientation (mean and std)
            % following 2 values are aspect ratios (mean and std)
            input_fabrics = [average(1:6), deviation(1:6), average(7:8), deviation(7:8)];
            % following 3 values are size, solidity and roundness (mean and std)
            for i = 9:11
                input_fabrics(end+1:end+2) = [average(i), deviation(i)];
            end
            % last value is the global volume fraction
            input_fabrics(end+1) = global_volume_fraction;
        end

        function image = binary_image_from_points(obj, points, save, filename)
            width = size(points);
            width = width(3);
            points_as_vector = reshape(permute(points, [2, 1, 3]), size(points, 2), [])';
            in = -inpolyhedron(struct("faces", obj.CL, "vertices", obj.P), points_as_vector) + 1;
            image = reshape(in, [width, width]);
            if save
                if ~exist("saved_images", 'dir')
                    mkdir("saved_images")
                end
                imwrite(image, strcat("saved_images/", filename), "WriteMode", "overwrite");
            end

        end

        function images = take_slice_images_along_x(obj, n, width, eps, save)
            fixed_x = linspace(obj.x_min+eps*(obj.x_max - obj.x_min), obj.x_max-eps*(obj.x_max - obj.x_min), n);
            images = zeros(n*width+3*(n - 1), width);
            for i = 1:length(fixed_x)
                y = linspace(obj.y_min, obj.y_max, width);
                z = linspace(obj.z_min, obj.z_max, width);
                [X, Y, Z] = meshgrid(fixed_x(i), y, z);
                A = horzcat(X, Y, Z);
                %     points is a array of size (N, 3). Each line is a point to study
                image = obj.binary_image_from_points(A, save, strcat("image_slice_x_", int2str(i), "-", int2str(length(fixed_x)), "_", int2str(width), "x", int2str(width), ".png"));
                images((i - 1)*(width + 3)+1:(i - 1)*(width + 3)+width, :) = image;
            end
        end

        function images = take_slice_images_along_y(obj, n, width, eps, save)
            fixed_y = linspace(obj.y_min+eps*(obj.y_max - obj.y_min), obj.y_max-eps*(obj.y_max - obj.y_min), n);
            images = zeros(n*width+3*(n - 1), width);
            for i = 1:length(fixed_y)
                x = linspace(obj.x_min, obj.x_max, width);
                z = linspace(obj.z_min, obj.z_max, width);
                [Y, X, Z] = meshgrid(fixed_y(i), x, z);
                A = horzcat(X, Y, Z);
                %     points is a array of size (N, 3). Each line is a point to study
                image = obj.binary_image_from_points(A, save, strcat("image_slice_y_", int2str(i), "-", int2str(length(fixed_y)), "_", int2str(width), "x", int2str(width), ".png"));
                images((i - 1)*(width + 3)+1:(i - 1)*(width + 3)+width, :) = image;
            end
        end

        function images = take_slice_images_along_z(obj, n, width, eps, save)
            fixed_z = linspace(obj.z_min+eps*(obj.z_max - obj.z_min), obj.z_max-eps*(obj.z_max - obj.z_min), n);
            images = zeros(n*width+3*(n - 1), width);
            for i = 1:length(fixed_z)
                x = linspace(obj.x_min, obj.x_max, width);
                y = linspace(obj.y_min, obj.y_max, width);
                [Z, X, Y] = meshgrid(fixed_z(i), x, y);
                A = horzcat(X, Y, Z);
                %     points is a array of size (N, 3). Each line is a point to study
                image = obj.binary_image_from_points(A, save, strcat("image_slice_z_", int2str(i), "-", int2str(length(fixed_z)), "_", int2str(width), "x", int2str(width), ".png"));
                images((i - 1)*(width + 3)+1:(i - 1)*(width + 3)+width, :) = image;
            end
        end

        function images = take_slice_images(obj, n, width, eps, save)
            images = zeros(n*width+3*(n - 1), 3*width+3*2);
            images(:, 1:width) = obj.take_slice_images_along_x(n, width, eps, save);
            images(:, (width + 3)+1:(width + 3)+width) = obj.take_slice_images_along_y(n, width, eps, save);
            images(:, 2*(width + 3)+1:2*(width + 3)+width) = obj.take_slice_images_along_z(n, width, eps, save);
        end
    end
end