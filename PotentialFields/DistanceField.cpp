#include "DistanceField.h"
#include <cmath>
#include <functional>
#include <iostream>
#include <omp.h>
#include <chrono>

using namespace std;


vector<vector<double>> edt(std::vector<std::vector<unsigned char>> occupancy_grid) {
    int rows = occupancy_grid.size();
    int cols = occupancy_grid[0].size();

    vector<vector<double>> col_res = vector<vector<double>>();
    vector<vector<double>> result = vector<vector<double>>();
    for (int i = 0; i < rows; i++) {
        vector<double> row = vector<double>();
        vector<double> row2 = vector<double>();
        for (int j = 0; j < cols; j++) {
            row.push_back(rows + cols);
            row2.push_back(rows + cols);
        }
        col_res.push_back(row);
        result.push_back(row2);
    }

    // Perform the 1d edt on each column going up and down
    auto start = chrono::system_clock::now();
    #pragma omp parallel for
    for (int j = 0; j < cols; j++) {
        double updist = rows * cols;
        double downdist = rows * cols;
        for (int i = 0; i < rows; i++) {
            if (occupancy_grid[i][j]) {
                downdist = 0;
            } else {
                downdist++;
            }

            if (occupancy_grid[rows - 1 - i][j]) {
                updist = 0;
            } else {
                updist++;
            }

            col_res[i][j] = min(col_res[i][j], downdist);
            col_res[rows - 1 - i][j] = min(col_res[rows - 1 - i][j], updist);
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        vector<double> row = col_res[i];
        vector<double> result_row = result[i];
        
        int x_pos = 0;
        int max_offset;
        for (int start_x = 0; start_x < cols; start_x++) {
            result_row[start_x] = sqrt(row[x_pos]*row[x_pos] + (start_x-x_pos)*(start_x-x_pos));
            max_offset = ceil(result_row[start_x]);
            for (int j = 0; j < min(max_offset, cols - start_x); j++) {
                double v = sqrt(row[start_x+j]*row[start_x+j] + j*j);
                if (v < result_row[start_x]) {
                    result_row[start_x] = v;
                    x_pos = start_x + j;
                    max_offset = ceil(sqrt(v));
                }
            }
        }
        result[i] = result_row;
    }
    auto end = chrono::system_clock::now();

    cout << "EDT Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;
    return result;
}

DistanceField::DistanceField(std::vector<std::vector<unsigned char>> occupancy_grid) {

    value = edt(occupancy_grid);

    rows = value.size();
    cols = value[0].size();
}

double DistanceField::get_value(std::vector<double> x) const {
    if (x[1] < 2 || x[1] >= cols - 2) {
        return 0;
    }
    if (x[0] < 2 || x[0] >= rows - 2) {
        return 0;
    }

    int i = x[0] - 0.5;
    int j = x[1] - 0.5;

    double tx = x[1] - j - 0.5;
    double ty = x[0] - i - 0.5;

    double lerpx0 = (1 - tx) * value[i][j] + tx * value[i][j+1];
    double lerpx1 = (1 - tx) * value[i+1][j] + tx * value[i+1][j+1];
    //cout << x[0] << "," << x[1] << "|" << value[i][j] << "," << value[i][j+1] << "," << value[i+1][j] << "," << value[i+1][j+1] << endl;

    return (1 - ty) * lerpx0 + ty * lerpx1;
}