#ifndef DISTANCE_FIELD_H
#define DISTANCE_FIELD_H

#include <vector>
#include <functional>

class DistanceField {
private:
public:
    std::vector<std::vector<double>> value;
    int rows;
    int cols;
    DistanceField(std::vector<std::vector<unsigned char>> occupancy_grid);

    double get_value(std::vector<double> x) const;
};
#endif