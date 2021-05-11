#include "DistanceField.h"
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <chrono>
#include <functional>
#include <queue>
#include <numeric>
#include <algorithm>

using namespace std;

template <typename T>
vector<int> sort_indexes(const vector<T> &v) {

  vector<int> idx = vector<int>();
  for(int i = 0; i < v.size(); i++)
    idx.push_back(i);

  stable_sort(idx.begin(), idx.end(),
       [&v](int i1, int i2) {return v[i1] < v[i2];});

  return idx;
}

float randomf() {
    return 2 * (((float)rand() / RAND_MAX) - 0.5);
}

vector<double> grad(const function<double(vector<double>)> &f, vector<double> x) {
    double h = 1e-5;
    double f0 = f(x);
    x[0] += h;
    double fx = f(x);
    x[0] -= h; x[1] += h;
    double fy = f(x);
    x[1] -= h;
    return vector<double>({(fx - f0) / h, (fy - f0) / h});
}

vector<vector<double>> gradient_descent(const function<double(vector<double>)> &f, vector<double> start, int max_iter = 5000) {
    auto x = start;
    auto trajectory = vector<vector<double>>();
    trajectory.push_back(x);
    vector<double> gd;
    int i = 0;

    do {
        gd = grad(f, x);
        double l = sqrt(pow(gd[0], 2) + pow(gd[1], 2));
        if (l < 1e-6)
            break;
        gd[0] *= min(l, 10.0) / l; gd[1] *= min(l, 10.0) / l;
        x = vector<double>({x[0] - gd[0], x[1] - gd[1]});
        trajectory.push_back(x);
        i++;
    } while(i < max_iter);
    
    return trajectory;
}

vector<vector<double>> bacterial_foraging(const function<double(vector<double>)> &f, const DistanceField & distanceField, vector<double> start, vector<double> end, double R = 10, int n = 10, int max_iter = 100) {
    auto x = start;
    vector<vector<double>> trajectory = vector<vector<double>>();
    trajectory.push_back(x);
    vector<vector<double>> bacteria = vector<vector<double>>();
    vector<double> errors = vector<double>();
    for (int i = 0; i < n; i++) {
        bacteria.push_back(vector<double>({0,0}));
        errors.push_back(0);
    }

    double delta = 0;

    int i = 0;
    while (pow(end[0] - x[0],2) + pow(end[1] - x[1],2) > R * R && i < max_iter) {
        for(int i = 0; i < n; i++) {
            double theta = ((double)rand() / RAND_MAX) * M_PI * 2;

            bacteria[i][0] = x[0] + R * cos(theta);
            bacteria[i][1] = x[1] + R * sin(theta);
        }

        #pragma omp parallel for
        for(int i = 0; i < n; i++) {
            auto traj = gradient_descent(f, bacteria[i], 10);
            bacteria[i] = traj.back();
            errors[i] = pow(bacteria[i][0] - end[0],2) + pow(bacteria[i][1] - end[1],2);
        }

        double best_error = 1e10;
        int best_b = -1;
        for(int i = 0; i < n; i++) {
            if (distanceField.get_value(bacteria[i]) > 1 && errors[i] < best_error) {
                best_b = i;
                best_error = errors[i];
            }
        }
        if (best_b > 0) {
            if (pow(bacteria[best_b][0] - x[0],2) + pow(bacteria[best_b][1] - x[1],2) < 1) {
                break;
            }
            x = bacteria[best_b];

            trajectory.push_back(x);
        }
        i++;
    }
    return trajectory;
}

vector<vector<double>> genetic_descent(const DistanceField & distanceField, const DistanceField &interiorDistanceField, vector<double> start, vector<double> end, int max_parents = 6, int max_iter = 10) {
    auto cost_function = [&distanceField, &end, &interiorDistanceField](vector<double> x, double goal_gain, double collision_gain) {
        auto f = [&distanceField, &interiorDistanceField, &goal_gain, &collision_gain, &end](vector<double> x) {
            double collision_distance = 100;
            double d = distanceField.get_value(x);
            double id = interiorDistanceField.get_value(x);
            double gd = pow(x[0] - end[0], 2) + pow(x[1] - end[1], 2);
            double v = goal_gain * 0.01 * gd;
            if (d < collision_distance)
                v += collision_gain * 1000 * pow(1 / (d + 1) - 1 / (collision_distance + 1), 2);
            v += id*id;
            return v;
        };
        vector<vector<double>> traj = gradient_descent(f, x);

        vector<double> last_x = traj.back();
        return sqrt(pow(last_x[0] - end[0], 2) + pow(last_x[1] - end[1], 2));
    };

    vector<double> achieved_pos;
    vector<vector<double>> population = vector<vector<double>>();
    double mutation_scale = 0.2;
    for(int i = 0; i < max_parents; i++) {
        population.push_back(vector<double>({
            randomf() * mutation_scale + 1,
            randomf() * mutation_scale + 1
        }));
    }
    vector<vector<double>> next_population = vector<vector<double>>();

    double best_cost = 1e10;
    vector<double> best_params = vector<double>({1,1});
    int i = 0;
    do {
        vector<double> evaluation = vector<double>();
        for(int j = 0; j < population.size(); j++)
            evaluation.push_back(0);
        #pragma omp parallel for
        for(int j = 0; j < population.size(); j++) {
            evaluation[j] = cost_function(start, population[j][0], population[j][1]);
        }
        for(int j = 0; j < evaluation.size(); j++){
            if (evaluation[j] < best_cost) {
                best_cost = evaluation[j];
                best_params[0] = population[j][0];
                best_params[1] = population[j][1];
            }
        }
        // Determine the best parents
        vector<int> parents = sort_indexes(evaluation);
        while(parents.size() > max_parents) {
            parents.pop_back();
        }

        // Breed the parents
        for(int j = 0; j < parents.size(); j++) {
            int mate = ((float)rand() / RAND_MAX) * parents.size();
            next_population.push_back(vector<double>({
                randomf() * mutation_scale + (population[parents[j]][0] + population[parents[mate]][0]) / 2,
                randomf() * mutation_scale + (population[parents[j]][1] + population[parents[mate]][1]) / 2,
            }));
        }

        i++;
        population = next_population;
        next_population = vector<vector<double>>();
    } while (best_cost > 1 && i < max_iter);

    auto f = [&](vector<double> x) {
        double collision_distance = 100;
        double d = distanceField.get_value(x);
        double id = interiorDistanceField.get_value(x);
        double gd = pow(x[0] - end[0], 2) + pow(x[1] - end[1], 2);
        double v = best_params[0] * 0.01 * gd;
        if (d < collision_distance)
            v += best_params[1] * 1000 * pow(1 / (d + 1) - 1 / (collision_distance + 1), 2);
        v += id*id;
        return v;
    };
    return gradient_descent(f, start);
}

double distance_to_trajectories(vector<double> x, vector<vector<vector<double>>> trajectories) {
    double d = 1e10;
    for (int j = 0; j < trajectories.size(); j++) {
        auto traj = trajectories[j];
        double curr_d = pow(x[0] - traj[0][0], 2) + pow(x[1] - traj[0][1], 2);
        d = min(d, curr_d);
        continue;
        for (int i = 0; i < traj.size(); i++) {
            double curr_d = pow(x[0] - traj[i][0], 2) + pow(x[1] - traj[i][1], 2);
            d = min(d, curr_d);
        }
    }
    return sqrt(d);
}

vector<vector<double>> find_path(vector<vector<double>> &final_traj, vector<double> end, vector<vector<vector<double>>> trajectories) {
    auto last_pos = final_traj.back();
    vector<int> used_trajectories = vector<int>();
    while (sqrt(pow(last_pos[0] - end[0], 2) + pow(last_pos[1] - end[1], 2)) > 3) {
        double d = 1e10;
        int traj_id;
        int pos_id;
        for (int j = 0; j < trajectories.size(); j++) {
            bool used = false;
            for (int k = 0; k < used_trajectories.size(); k++){
                if (used_trajectories[k] == j){
                    used = true;
                    break;
                }
            }
            if (used) continue;
            auto traj = trajectories[j];
            double curr_d = sqrt(pow(last_pos[0] - traj[0][0], 2) + pow(last_pos[1] - traj[0][1], 2));
            if (curr_d < d) {
                d = curr_d;
                traj_id = j;
                pos_id = 0;
            }
            // for (int i = 0; i < traj.size(); i++) {
            //     double curr_d = sqrt(pow(last_pos[0] - traj[i][0], 2) + pow(last_pos[1] - traj[i][1], 2));
            //     if (curr_d < d) {
            //         d = curr_d;
            //         traj_id = j;
            //         pos_id = i;
            //     }
            // }
        }
        if (d > 3) {
            break;
        }
        last_pos = trajectories[traj_id].back();
        used_trajectories.push_back(traj_id);
        for(int i = pos_id; i < trajectories[traj_id].size(); i++) {
            final_traj.push_back(trajectories[traj_id][i]);
        }
    }
    return final_traj;
}

vector<vector<double>> random_descent(const DistanceField & distanceField, const DistanceField & interiorDistanceField, vector<double> start, vector<double> end, int spread = 5, int R = 25, int max_iter = 10) {
    vector<vector<vector<double>>> good_trajectories = vector<vector<vector<double>>>();
    
    auto f = [&](vector<double> x) {
        double collision_distance = 100;
        double d = distanceField.get_value(x);
        double id = interiorDistanceField.get_value(x);
        double gd = sqrt(pow(x[0] - end[0], 2) + pow(x[1] - end[1], 2));
        gd = min(distance_to_trajectories(x, good_trajectories), gd);
        if (gd < 3) {
            return 0.0;
        }
        double v = 0.01 * gd * gd;
        if (d < collision_distance)
            v += 1000 * pow(1 / (d + 1) - 1 / (collision_distance + 1), 2);
        v += id*id;
        return v;
    };

    vector<vector<double>> traj;
    for (int iter = 0; iter < max_iter; iter++) {
        cout << iter << "|" << good_trajectories.size() << endl;
        traj = gradient_descent(f, start);
        auto x = traj.back();
        if (sqrt(pow(x[0] - end[0], 2) + pow(x[1] - end[1], 2)) < 3) {
            return traj;
        }
        if (distance_to_trajectories(x, good_trajectories) < 3) {
            break;
        }

        vector<vector<vector<double>>> next_trajectories = vector<vector<vector<double>>>();

        double r = R * (iter / 2.5 + 1);
        #pragma omp parallel for
        for (int i = 0; i < spread; i++) {
            vector<double> s;
            do {
                s = vector<double>({(double)(rand() % distanceField.rows), (double)(rand() % distanceField.cols)});
            } while(distanceField.get_value(s) < 1);
            auto t = gradient_descent(f, s);

            #pragma omp critical
            {
                next_trajectories.push_back(t);
            }
        }

        for (int i = 0; i < next_trajectories.size(); i++) {
            auto end_pos = next_trajectories[i].back();
            double d = sqrt(pow(end_pos[0] - end[0], 2) + pow(end_pos[1] - end[1], 2));
            d = min(distance_to_trajectories(end_pos, good_trajectories), d);
            if (d < 3) {
                good_trajectories.push_back(next_trajectories[i]);
            } else {
                //bad_trajectories.push_back(next_trajectories[i]);
            }
        }
    }
    return find_path(traj, end, good_trajectories);
}

vector<vector<unsigned char>> read_file(string filename) {
    vector<vector<unsigned char>> result = vector<vector<unsigned char>>();

    ifstream collisionFile(filename);
    string temp;

    int rows = 0;
    int cols = 0;
    if(collisionFile.good()){
        getline(collisionFile, temp, ',');
        rows = atoi(temp.c_str());
        getline(collisionFile, temp);
        cols = atoi(temp.c_str());
    }
    
    for(int i = 0; i < rows; i++) {
        vector<unsigned char> row = vector<unsigned char>();
        for (int j = 0; j < cols - 1; j++) {
            getline(collisionFile,temp,',');
            row.push_back(atof(temp.c_str()));
        }
        getline(collisionFile,temp);
        row.push_back(atof(temp.c_str()));
        result.push_back(row);
    }

    collisionFile.close();

    return result;
}

vector<vector<vector<double>>> read_planning_goals(string filename) {
    vector<vector<vector<double>>> result = vector<vector<vector<double>>>();

    ifstream file(filename);
    string temp;

    vector<double> goal = vector<double>();

    getline(file, temp, ',');
    goal.push_back(stof(temp.c_str()));
    getline(file, temp);
    goal.push_back(stof(temp.c_str()));
    
    while (file.peek() != EOF) {
        vector<double> start = vector<double>();
        getline(file, temp, ',');
        start.push_back(stof(temp.c_str()));
        getline(file, temp);
        start.push_back(stof(temp.c_str()));

        result.push_back(vector<vector<double>>({start, goal}));
    }
    file.close();

    return result;
}

void write_trajectory(string filename, vector<double> start, vector<double> end, vector<vector<double>> traj) {
    ofstream fout(filename, ios_base::app);
    fout << start[0] << "," << start[1] << "|" << end[0] << "," << end[1] << "|";
    for(int i = 0; i < traj.size(); i++) {
        fout << traj[i][0] << "," << traj[i][1] << ";";
    }
    fout << endl;
    fout.close();
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        cout << "Enter the binary image CSV to process followed by CSV of planning goals and the results folder" << endl;
        return -1;
    }

    auto array = read_file(argv[1]);
    auto inverse = array;
    for (int i = 0; i < inverse.size(); i++){
        for (int j = 0; j < inverse[j].size(); j++){
            inverse[i][j] = 1 - inverse[i][j];
        }
    }
    auto distancefield = DistanceField(array);

    ofstream fout("file.csv");
    for(int i = 0; i < distancefield.rows; i++) {
        for(int j = 0; j < distancefield.cols; j++) {
            fout << distancefield.get_value(vector<double>({(double)i,(double)j})) << ",";
        }
        fout << endl;
    }
    fout.close();

    auto interiorDistanceField = DistanceField(inverse);
    auto tasks = read_planning_goals(argv[2]);

    string result_folder = argv[3];

    double collision_distance = 500;
    vector<vector<double>> traj;
    for(int i = 0; i < tasks.size(); i++) {
        auto start = tasks[i][0];
        auto end = tasks[i][1];

        if (distancefield.get_value(start) < 1 || distancefield.get_value(end) < 1) {
            cout << "Invalid goal. Start: " << start[0] << "," << start[1] << "; End: " << end[0] << "," << end[1] << endl;
            continue; 
        }

        auto f = [&](vector<double> x) {
            double d = distancefield.get_value(x);
            double id = interiorDistanceField.get_value(x);
            double gd = pow(x[0] - end[0], 2) + pow(x[1] - end[1], 2);
            double v = 0.01 * gd;
            if (d < collision_distance)
                v += 100 * pow(1 / (d + 1) - 1 / (collision_distance + 1), 2);
            v += id*id;
            return v;
        };
        cout << "Gradient Descent" << endl;
        traj = gradient_descent(f, start);
        write_trajectory(result_folder + "GradientDescent.csv", start, end, traj);

        cout << "Bacterial Foraging" << endl;
        traj = bacterial_foraging(f, distancefield, start, end);
        write_trajectory(result_folder + "BacterialForaging.csv", start, end, traj);

        cout << "Genetic Descent" << endl;
        traj = genetic_descent(distancefield, interiorDistanceField, start, end);
        write_trajectory(result_folder + "GeneticDescent.csv", start, end, traj);

        cout << "Random Descent" << endl;
        traj = random_descent(distancefield, interiorDistanceField, start, end);
        write_trajectory(result_folder + "RandomDescent.csv", start, end, traj);
    }

    return 0;
}