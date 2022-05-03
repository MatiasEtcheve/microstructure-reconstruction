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
        
        function distance_to_nearest = compute_distance_to_nearest(obj)
            D = squareform(pdist(obj.barycenters));
            D = D + max(D(:))*eye(size(D)); % ignore zero distances on diagonals 
            distance_to_nearest = min(D, [],2);
        end

        function fabrics = compute_fabrics(obj)
            % computes the fabrics of each grain and append it to a list
            fabrics = zeros(length(obj.grains), 9);
            for index = 1:length(obj.grains)
                fabrics(index, :) = obj.grains{index}.compute_fabrics();
            end
        end

        function [invariants, orientation_vectors] = compute_invariants(obj, unit_vectors)
            F11 = unit_vectors(:, 1).*unit_vectors(:, 1);
            F22 = unit_vectors(:, 2).*unit_vectors(:, 2);
            F33 = unit_vectors(:, 3).*unit_vectors(:, 3);
            F12 = unit_vectors(:, 1).*unit_vectors(:, 2);
            F13 = unit_vectors(:, 1).*unit_vectors(:, 3);
            F23 = unit_vectors(:, 2).*unit_vectors(:, 3);
            orientation_vectors = horzcat(F11, F22, F33, F23, F13, F12);
            f11 = mean(F11);
            f22 = mean(F22);
            f33 = mean(F33);
            f12 = mean(F12);
            f13 = mean(F13);
            f23 = mean(F23);
            A1 = f11 + f22 + f33;
            A2 = (f11.*f22 - f12.*f12) + (f22.*f33 - f23.*f23) + (f11.*f33 - f13.*f13);
            A3 = f11.*f22.*f33 + 2.*f12.*f23.*f13 - f22.*f13.*f13 - f11.*f23.*f23 - f33.*f12.*f12;
            invariants = horzcat(A1, A2, A3);
        end

        function input_fabrics = compute_input_fabrics(obj, save)
            % creates a well-formed input fabrics with mean and std
            fabrics = obj.compute_fabrics();
            distance_to_nearest = obj.compute_distance_to_nearest();
            [invariants, orientation_vectors] = obj.compute_invariants(fabrics(:, 1:3));
            custom_fabrics = horzcat(distance_to_nearest, orientation_vectors, fabrics(:, 4:end-1));
            avg = mean(custom_fabrics);
            deviation = std(custom_fabrics);
            aggregates_volume = sum(fabrics(:, end));
            global_volume_fraction = aggregates_volume / obj.total_volume;

            input_fabrics = zeros(1, size(invariants, 2)+size(avg, 2)+size(deviation, 2)+1);
            input_fabrics(1, 1:2) = horzcat(avg(1), deviation(1));
            input_fabrics(1, 3:5) = invariants;
            input_fabrics(1, 6:17) = horzcat(avg(2:7), deviation(2:7));
            input_fabrics(1, 18:21) = horzcat(avg(8:9), deviation(8:9));
            input_fabrics(1, 22:2:end-1) = avg(10:end);
            input_fabrics(1, 23:2:end-1) = deviation(10:end);
            input_fabrics(1, end) = global_volume_fraction;

            if save
                [filepath, name, ~] = fileparts(obj.path);
                fabrics_file = strcat(filepath, "\fabrics_", name, ".txt");
                writematrix(input_fabrics, fabrics_file, 'Delimiter', ',')
            end
        end

        function inside_matrix = compute_inside_matrix(obj, ti, X, Y, Z)
            in = inpolyhedron(struct("faces", obj.grains{1}.CL, "vertices", obj.grains{1}.P), X, Y, Z) * 1;
            for index = 2:length(obj.grains)
                current_in = inpolyhedron(struct("faces", obj.grains{index}.CL, "vertices", obj.grains{index}.P), X, Y, Z) * index;
                in = in + current_in;
            end
            inside_matrix = squeeze(in);
        end
        
        function [cl, points] = compute_mesh(obj, ti, X)
            Y = obj.y_min:ti:obj.y_max+ti;
            Z = obj.z_min:ti:obj.z_max+ti;
            inside_matrix = obj.compute_inside_matrix(ti, X, Y, Z);
            
            nb_rows = length(Y);
            nb_cols = length(Z);
            elt_id = 1;
            array_index = 1;

            cl = cell((nb_rows-1)*(nb_cols-1)+1, 7);
            cl(1, :) = {"elt_id", "material_id", "object_id", "upleft_node", "downleft_node", "downright_node", "upright_node"};
            points = cell(nb_rows*nb_cols+1, 3);
            points(1, :) = {"point_id", "x", "y"};
            for i = 1:nb_rows
                for j = 1:nb_cols
                    grain_id = inside_matrix(i, j);
                    material_id = 0;
                    if grain_id > 0
                        material_id = 1;
                    end
                    up_left = array_index;
                    up_right = array_index+1;
                    down_left = array_index + nb_cols;
                    down_right = array_index + nb_cols +1;
                    
                    if j<nb_cols && i<nb_rows
                        cl(elt_id+1, :) = {elt_id, material_id, grain_id, up_left, down_left, down_right, up_right};
                        points(up_left+1, :) = {up_left, Y(i), Z(j)};
                        points(down_left+1, :) = {down_left, Y(i+1), Z(j)};
                        points(down_right+1, :) = {down_right, Y(i+1), Z(j+1)};
                        points(up_right+1, :) = {up_right, Y(i), Z(j+1)};
                                   
                        elt_id = elt_id + 1;
                    end
                    array_index = array_index + 1;
                end
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