classdef rev
    properties
        path % path to the stl file. Useful to save the input fabrics and slice_images in the current folder with pretty name
        TR % triangulation object
        P % points of the triangulation
        CL % connectivity list of the triangulation
        grains % cell of grain objects
        total_volume % total volume of the rev
        barycenters
        x_min
        y_min
        z_min
        x_max
        y_max
        z_max
    end
    methods
        function obj = rev(filename, rotation_matrix)
            obj.path = filename;
            TR = stlread(filename);
               
            if isa(rotation_matrix, "double")
                TR = triangulation(TR.ConnectivityList, TR.Points * rotation_matrix);
            end
            
            obj.TR = TR;
            obj.P = TR.Points;
            obj.CL = TR.ConnectivityList;

            obj.x_min = min(obj.P(:, 1));
            obj.y_min = min(obj.P(:, 2));
            obj.z_min = min(obj.P(:, 3));
            obj.x_max = max(obj.P(:, 1));
            obj.y_max = max(obj.P(:, 2));
            obj.z_max = max(obj.P(:, 3));

            obj.total_volume = prod(max(obj.P)-min(obj.P));

            AdjMat = false(size(obj.P, 1));
            for kk = 1:size(TR, 1)
                AdjMat(TR(kk, 1), TR(kk, 2)) = true;
                AdjMat(TR(kk, 2), TR(kk, 3)) = true;
                AdjMat(TR(kk, 3), TR(kk, 1)) = true;
            end
            [point_index_grain, bin_sizes] = conncomp(graph(AdjMat));
            [~, number_grains] = size(bin_sizes);

            % we create a obj.grains list, whose values are grain object
            % to know which point of the rev belongs to which grain, we us the adjency matrix
            grains = cell(number_grains, 1);
            barycenters = zeros(number_grains, 3);
            parfor grain_index = 1:number_grains
                grain_point_indexes = find(point_index_grain(:, :) == grain_index).';
                cl_index = ismember(TR.ConnectivityList(:, 1), grain_point_indexes);
                grain_triangulation = triangulation(TR.ConnectivityList(cl_index, :)-min(TR.ConnectivityList(cl_index, :), [], 'all')+1, TR.Points(grain_point_indexes, :));
                grains{grain_index} = grain(grain_triangulation);
                barycenters(grain_index, :) = grains{grain_index}.barycenter;
            end
            obj.grains = grains;
            obj.barycenters = barycenters;
        end
        
        function [m, s] = compute_distance_to_nearest(obj)
            D = squareform(pdist(obj.barycenters));
            D = D + max(D(:))*eye(size(D)); % ignore zero distances on diagonals 
            distance_to_nearest = min(D, [],2);
            m = mean(distance_to_nearest);
            s = std(distance_to_nearest);
        end

        function fabrics = compute_fabrics(obj)
            % computes the fabrics of each grain and append it to a list
            fabrics = zeros(length(obj.grains), 12);
            for index = 1:length(obj.grains)
                fabrics(index, :) = obj.grains{index}.compute_fabrics();
            end
        end

        function input_fabrics = compute_input_fabrics(obj, save)
            % creates a well-formed input fabrics with mean and std
            [mean_nearest_distance, std_nearest_distance] = obj.compute_distance_to_nearest();
            fabrics = obj.compute_fabrics();
            average = mean(fabrics(:, 1:end-1));
            deviation = std(fabrics(:, 1:end-1));
            aggregates_volume = sum(fabrics(:, end));
            global_volume_fraction = aggregates_volume / obj.total_volume;
            % 2 first values are angles and std
            % six next values are orientation (mean and std)
            % following 2 values are aspect ratios (mean and std)
            % input_fabrics = [average(1:2), deviation(1:2), average(3:8), deviation(3:8), average(9:10), deviation(9:10)];
            input_fabrics = [mean_nearest_distance, std_nearest_distance, average(1:6), deviation(1:6), average(7:8), deviation(7:8)];

            % following 3 values are size, solidity and roundness (mean and std)
            for i = 9:11
                input_fabrics(end+1:end+2) = [average(i), deviation(i)];
            end
            % last value is the global volume fraction
            input_fabrics(end+1) = global_volume_fraction;
            if save
                [filepath, name, ~] = fileparts(obj.path);
                fabrics_file = strcat(filepath, "\fabrics_", name, ".txt");
                writematrix(input_fabrics, fabrics_file, 'Delimiter', ',')
            end
        end

        function image = binary_image_from_points(obj, points, save, dir_path, filename)
            width = size(points);
            width = width(3);
            points_as_vector = reshape(permute(points, [2, 1, 3]), size(points, 2), [])';
            in = -inpolyhedron(struct("faces", obj.CL, "vertices", obj.P), points_as_vector) + 1;
            image = reshape(in, [width, width]);

            if save
                [filepath, name, ~] = fileparts(obj.path);
                dir_path = append(dir_path, strcat(name, "_Imgs")); 
%                 dir_path = strcat(filepath, "/", int2str(n), "_", name, "_Imgs");
                if ~exist(dir_path, 'dir')
                    mkdir(dir_path)
                end
                imwrite(image, strcat(dir_path, "/", filename));
            end

        end

        function images = take_slice_images_along_x(obj, n, width, eps, save, dir_path)
            fixed_x = linspace(obj.x_min+eps*(obj.x_max - obj.x_min), obj.x_max-eps*(obj.x_max - obj.x_min), n);
%             images = zeros(n*width+3*(n - 1), width);
            images = cell(n, 1);
            for i = 1:length(fixed_x)
                y = linspace(obj.y_min, obj.y_max, width);
                z = linspace(obj.z_min, obj.z_max, width);
                [X, Y, Z] = meshgrid(fixed_x(i), y, z);
                A = horzcat(X, Y, Z);
                %     points is a array of size (N, 3). Each line is a point to study
                image = obj.binary_image_from_points(A, save, dir_path, strcat("image_slice_x_", int2str(i), "-", int2str(length(fixed_x)), ".png"));
%                 images((i - 1)*(width + 3)+1:(i - 1)*(width + 3)+width, :) = image;
                images{i} = image;

            end
        end

        function images = take_slice_images_along_y(obj, n, width, eps, save, dir_path)
            fixed_y = linspace(obj.y_min+eps*(obj.y_max - obj.y_min), obj.y_max-eps*(obj.y_max - obj.y_min), n);
%             images = zeros(n*width+3*(n - 1), width);
            images = cell(n, 1);
            for i = 1:length(fixed_y)
                x = linspace(obj.x_min, obj.x_max, width);
                z = linspace(obj.z_min, obj.z_max, width);
                [Y, X, Z] = meshgrid(fixed_y(i), x, z);
                A = horzcat(X, Y, Z);
                %     points is a array of size (N, 3). Each line is a point to study
                image = obj.binary_image_from_points(A, save, dir_path, strcat("image_slice_y_", int2str(i), "-", int2str(length(fixed_y)), ".png"));
%                 images((i - 1)*(width + 3)+1:(i - 1)*(width + 3)+width, :) = image;
                images{i} = image;

            end
        end

        function images = take_slice_images_along_z(obj, n, width, eps, save, dir_path)
            fixed_z = linspace(obj.z_min+eps*(obj.z_max - obj.z_min), obj.z_max-eps*(obj.z_max - obj.z_min), n);
            images = cell(n, 1);
%             images = zeros(n*width+3*(n - 1), width);
            for i = 1:length(fixed_z)
                x = linspace(obj.x_min, obj.x_max, width);
                y = linspace(obj.y_min, obj.y_max, width);
                [Z, X, Y] = meshgrid(fixed_z(i), x, y);
                A = horzcat(X, Y, Z);
                %     points is a array of size (N, 3). Each line is a point to study
                image = obj.binary_image_from_points(A, save, dir_path, strcat("image_slice_z_", int2str(i), "-", int2str(length(fixed_z)), ".png"));
%                 images((i - 1)*(width + 3)+1:(i - 1)*(width + 3)+width, :) = image;
                images{i} = image;
            end
        end

        function images = take_slice_images(obj, n, width, eps, save, dir_path)
            images(:, 1) = obj.take_slice_images_along_x(n, width, eps, save, dir_path);
            images(:, 2) = obj.take_slice_images_along_y(n, width, eps, save, dir_path);
            images(:, 3) = obj.take_slice_images_along_z(n, width, eps, save, dir_path);
%             images = zeros(n*width+3*(n - 1), 3*width+3*2);
%             images(:, 1:width) = obj.take_slice_images_along_x(n, width, eps, save);
%             images(:, (width + 3)+1:(width + 3)+width) = obj.take_slice_images_along_y(n, width, eps, save);
%             images(:, 2*(width + 3)+1:2*(width + 3)+width) = obj.take_slice_images_along_z(n, width, eps, save);
        end
    end
end