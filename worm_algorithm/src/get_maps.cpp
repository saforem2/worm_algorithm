/**
 *
 * @file worm_ising_2d.cpp
 *
 * @author Sam Foreman
 *
 * @brief Main method for Monte-Carlo implementation
 * of worm algorithm as defined for 2D Ising spin lattice.
 */
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include "mt19937ar.c"

int bond_number(int i1, int i2, int *x, int *y, int L);
//std::vector<int> bonds_history;
//std::vector<int> head_history;

int main()
{
  int L = 6;
  int N = L * L;

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

  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      int neighbor = nbr[i][j];
      bond_number(i, neighbor, x, y, L);
    }

  }
  return 0;
}


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
int bond_number(int i1, int i2, int *x, int *y, int L){
    int x1 = x[i1], x2 = x[i2], y1 = y[i1], y2 = y[i2];
    std::string map_path = std::string("6_bond_map.txt");
    std::ofstream map_out;
    map_out.open(map_path, std::ios::app);

    if (y1 == y2){
        if (x2 == x1 + 1){
          map_out << 2 * i1 << ", " <<
           x1 << "," << y1 << "," << x2 << "," << y2 <<
          std::endl;
          return 2 * i1;
        }
        else if (x1 == x2 + 1){
          map_out << 2 * i2 << ", " <<
            x1 << ", " << y1 << ", " << x2 << ", " << y2 <<
            std::endl;
          return 2 * i2;
        }
        else if (x1 == L - 1){
          map_out << 2 * i1 << ", " <<
            x1 << ", " << y1 << ", " << x2 << ", " << y2 <<
            std::endl;
          return 2 * i1;
        }
        else if (x2 == L - 1){
          map_out << 2 * i2 << ", " <<
            x1 << ", " << y1 << ", " << x2 << ", " << y2 <<
            std::endl;
          return 2 * i2;
        }
    }
    else if (x1 == x2){
        if (y2 == y1 + 1){
          map_out << 2 * i1 + 1 << ", " <<
             x1 << ", " << y1 << ", " << x2 << ", " << y2 <<
            std::endl;
          return 2 * i1 + 1;
        }
        else if (y1 == y2 + 1){
          map_out << 2 * i2 + 1 << ", " <<
            x1 << ", " << y1 << ", " << x2 << ", " << y2 <<
            std::endl;
          return 2 * i2 + 1;
        }
        else if (y1 == L - 1){
          map_out << 2 * i1 + 1 << ", " <<
            x1 << ", " << y1 << ", " << x2 << ", " << y2 <<
            std::endl;
          return 2 * i1 +1;
        }
        else if (y2 == L - 1){
          map_out << 2 * i2 + 1 << ", " <<
            x1 << ", " << y1 << ", " << x2 << ", " << y2 <<
            std::endl;
          return 2 * i2 + 1;
        }
    }
    map_out.close();
    // if you got here, something went wrong
    printf("ERROR!\n");
    return 2 * L * L + 1;
}
