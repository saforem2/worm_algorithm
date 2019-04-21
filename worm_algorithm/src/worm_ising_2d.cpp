/**
 *
 * @file worm_ising_2d.cpp
 *
 * @brief Main method for Monte-Carlo implementation
 * of worm algorithm as defined for 2D Ising spin lattice.
 */
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <stdio.h>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <time.h>
#include <stdlib.h>
#include "mt19937ar.c"

bool fexists(const std::string& filename) {
  std::ifstream ifile(filename.c_str());
  return (bool)ifile;
}

std::tuple<int, int, int, int, int> bond_number(
    int i1, int i2, int *x, int *y, int L
);

int main()
{
    //read parameters from the input file
    int L;
    unsigned long int num_steps;
    double T;
    double seed;
    int bond_flag;
    // int write_obs;
    FILE *in_file = fopen("../data/setup/input.txt","r");
    if (in_file == NULL ){
        printf("Error! Could not open input file (input.txt: L, T, num_steps)\n");
        return 2;
    }
    fscanf(in_file,"%i %lf %li %lf %i", &L, &T, &num_steps, &seed, &bond_flag);
    fclose(in_file);

    init_genrand(seed);
    // double T_crit = 2/log(1+sqrt(2));
    double K = 1.0 / T, P_acc;
    double inv_K = 1.0 / K;
    int N = L * L;
    int tail = 0, head = 0, new_head;
    int nb, ibond, delta_nb, Nb = 0, Nb_tot = 0;
    unsigned long int step_num = 0, Z = 0;
    unsigned long int therm_steps = 0.1 * num_steps;
    unsigned long int write_steps = 50;
    double E_av = 0.0;
    double Z_av = 0.0;
    double Nb_av = 0.0;
    int bond_num, x1, x2, y1, y2;
    // double gamma = 1.75;
    // double zc = 2.17;
    std::setprecision(2);
    std::string T_str = std::to_string(T);
    std::string L_str = std::to_string(L);
    std::string T_str_sub = T_str.substr(0,5);

    std::string bond_map_file = std::string("../data/bond_map/lattice_")
      + L_str + std::string("/bond_map_") + L_str + std::string(".txt");

    std::string bonds_file = std::string("../data/bonds/lattice_") + L_str
      + std::string("/bonds_") + T_str_sub + std::string(".txt");

    std::string num_bonds_file = std::string("../data/num_bonds/lattice_")
      + L_str + std::string("/num_bonds_") + L_str + std::string(".txt");

    std::string observables_file = std::string("../data/observables/lattice_")
      + L_str + std::string("/observables_") + L_str + std::string(".txt");

    // build site->x and site->y tables
    int x[N], y[N];
    for (int site=0; site<N; site++)
    {
        x[site] = site - (site / L) * L;
        y[site] = site / L;
    }
    // build neighbors table
    int nbr[N][4];
    for (int site=0; site<N; site++)
    {
        nbr[site][0] = (site / L) * L + (site + 1 + N) % L;
        nbr[site][1] = (site + L) % N;
        nbr[site][2] = (site / L) * L + (site - 1 + N) % L;
        nbr[site][3] = (site - L + N) % N;
    }
    // initialize bond weights
    int bonds[N * 2];
    for (int b=0; b<N * 2; b++)
    {
        bonds[b] = 0;
    }


    std::ofstream observables_out;
    // main Monte Carlo loop
    observables_out.open(observables_file, std::ofstream::app);
    // if (bond_flag != 0) {
    std::ofstream bonds_out;
    bonds_out.open(bonds_file, std::ofstream::app);
    // }
    while(1) {
      if (tail == head) {
          if (step_num > therm_steps) {
              if (step_num % write_steps == 0) {
                  Z_av = Z / (step_num * 1.0);
                  E_av = Nb_tot * T / (Z * N);
                  Nb_av = Nb_tot / (Z * N);
                  observables_out << T << " " << E_av << " " << Z_av << " " 
                      << Nb_av << " " << step_num << std::endl;
                  if (bond_flag != 0) {
                      for (int b=0; b < N * 2; b++) {
                          bonds_out << b << " " << bonds[b] << std::endl;
                      }
                  } 
              }
          }
          // randomly choose new head, tail
          tail = (int)floor(genrand() * N);
          head = tail;
          Z += 1;   // new kick
          // remove Nb bonds from previous worm configuration
          Nb_tot -= Nb;
          if (step_num >= num_steps)
              break;
      }
      // shift move -- start
      new_head = nbr[head][(int)floor(genrand() * 4)];
      std::tie(ibond, x1, x2, y1, y2) = bond_number(head, new_head, x, y, L);
      nb = bonds[ibond];
      
      if (genrand() < 0.5) {
        delta_nb = 1;
        P_acc = K / (nb + 1.0);
      }
      
      else {
        delta_nb = - 1;
        P_acc = nb * inv_K;
      }
      
      if (genrand() < P_acc) {
        bonds[ibond] += delta_nb;
        Nb += delta_nb;
        head = new_head;
      }
    ++step_num; // shift move -- end
    } // end MC loop
    observables_out.close();

    if (bond_flag != 0) {
        std::ofstream bonds_out;
        bonds_out.open(bonds_file, std::ofstream::app);
        for (int b=0; b < N * 2; b++) {
            bonds_out << b << " " << bonds[b] << std::endl;
        } 
        bonds_out.close();
    }

    if (!fexists(bond_map_file)) {
        std::ofstream bonds_map_out;
        bonds_map_out.open(bond_map_file);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < 4; ++j) {
                int neighbor = nbr[i][j];
                std::tie(bond_num, x1, x2, y1, y2) = bond_number(i, neighbor, x, y, L);
                bonds_map_out << bond_num << " " << x1 << " " 
                    << y1 << " " << x2 << " " << y2 << std::endl;
            }
        }
        bonds_map_out.close();
    }

    Z_av = Z / (step_num * 1.0);
    //E_av = - 1.0 * tanh(K) * (2*L*L + (Nb_tot/(sinh(K)*sinh(K))));
    E_av = Nb_tot * T / (Z * N);
    Nb_av = Nb_tot / (Z * N);

    // print output
    FILE *out_file = fopen("../data/setup/output.txt","w");
    fprintf(out_file, "%4i %.12f %.12f %.12lf %.12lf %.12lf %lu\n",
            L, T, K, Z_av, E_av, Nb_av, step_num);
        //Nb_tot * T / (Z * N * 1.0));
    fclose(out_file);

    std::ofstream num_bonds_out;
    num_bonds_out.open(num_bonds_file, std::ofstream::app);
    num_bonds_out << L << " " << T << " " << Nb << "\n";
    num_bonds_out.close();
} //end main

/**
 * @brief method for computing the index of the bond connecting
 * the points i1, i2.
 *
 * @param i1: Integer specifying start point of bond
 * @param i2: Integer specifying end point of bond
 * @param x: Pointer to integer specifying horizontal coordinate of lattice
 * site.
 * @param y: Pointer to integer specifying vertical coordinate of lattice site
 *
 * @return bond_number: Integer specifying index of bond connecting the sites
 * i1, i2.
 */
std::tuple<int, int, int, int, int> bond_number(
    int i1, int i2, int *x, int *y, int L
)
{
  int x1 = x[i1], x2 = x[i2], y1 = y[i1], y2 = y[i2];
  if (y1 == y2)
  {
      if (x2 == x1 + 1)
        return std::make_tuple(2 * i1, x1, x2, y1, y2);
      else if (x1 == x2 + 1)
        return std::make_tuple(2 * i2, x1, x2, y1, y2);
      else if (x1 == L - 1)
        return std::make_tuple(2 * i1, x1, x2, y1, y2);
      else if (x2 == L - 1)
        return std::make_tuple(2 * i2, x1, x2, y1, y2);
  }
  else if (x1 == x2)
  {
      if (y2 == y1 + 1)
        return std::make_tuple(2 * i1 + 1, x1, x2, y1, y2);
      else if (y1 == y2 + 1)
        return std::make_tuple(2 * i2 + 1, x1, x2, y1, y2);
      else if (y1 == L - 1)
        return std::make_tuple(2 * i1 +1, x1, x2, y1, y2);
      else if (y2 == L - 1)
        return std::make_tuple(2 * i2 + 1, x1, x2, y1, y2);
  }
  // if you got here, something went wrong
  printf("ERROR!\n");
  return std::make_tuple(2 * L * L + 1, 0, 0, 0, 0);
}
